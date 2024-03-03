import argparse
import copy
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
from config import TASK_NER_LABEL_LIST, TASK_REL_LABEL_INFO_DICT
from config import pipe2_re_tokaux_sent as CONFIG
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import batched_index_select, check_and_mkdir, rindex, set_seed

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.DEBUG)
LOGGER = logging.getLogger("re_tokaux")


TOKEN_CONST = {
    "pad_token_id": None,
    "seq_start_token": None,
    "seq_end_token": None,
    "subj_start_marker": "[unused0]",
    "subj_end_marker": "[unused1]",
    "obj_start_marker": "[unused2]",
    "obj_end_marker": "[unused3]",
}


DEVICE = None

LOSS_IGNORE_INDEX = -100


##############################################
# Model class
##############################################


class REModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_rel_labels = config.num_rel_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)

        self.tok_classifier = nn.Linear(config.hidden_size, self.num_rel_labels)

        self.re_classifier_subj = nn.Linear(config.hidden_size * 2, self.num_rel_labels)
        self.re_classifier_obj = nn.Linear(config.hidden_size * 2, self.num_rel_labels)

        self.init_weights()

    def _get_span_embeddings(self, input_dict):
        # torch.Size: [batch_size, num_toks, 768(tok_emb)]
        output = self.bert(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["input_masks"],
            position_ids=input_dict["position_ids"],
        )
        last_hidden_state = output.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)

        span_pairs_info = input_dict["rel_info"]
        obj_markers_start = span_pairs_info[:, :, 0]
        obj_markers_start_embs = batched_index_select(last_hidden_state, obj_markers_start)
        obj_markers_end = span_pairs_info[:, :, 1]
        obj_markers_end_embs = batched_index_select(last_hidden_state, obj_markers_end)

        obj_embs = torch.cat((obj_markers_start_embs, obj_markers_end_embs), dim=-1)

        subj_indices = input_dict["subj_indices"]
        subj_markers_start = subj_indices[:, 0]
        subj_markers_start_embs = last_hidden_state[torch.arange(subj_indices.shape[0]), subj_markers_start]
        subj_markers_end = subj_indices[:, 1]
        subj_markers_end_embs = last_hidden_state[torch.arange(subj_indices.shape[0]), subj_markers_end]

        subj_embs = torch.cat([subj_markers_start_embs, subj_markers_end_embs], dim=-1)

        aux_tok_info = input_dict["aux_tok_info"]
        aux_tok_indices = aux_tok_info[:, :, 0]
        aux_tok_embs = batched_index_select(last_hidden_state, aux_tok_indices)

        return subj_embs, obj_embs, aux_tok_embs

    def forward(self, input_dict, mode):
        subj_embs, obj_embs, aux_tok_embs = self._get_span_embeddings(input_dict)
        rel_labels = input_dict["rel_info"][:, :, 2]
        ner_labels = input_dict["rel_info"][:, :, 5]
        aux_tok_labels = input_dict["aux_tok_info"][:, :, 1:]

        obj_ner_logits = self.ner_classifier(obj_embs)  # 对subj marker进行ent分类; (bsz, ent_len, hidd_dim*2)

        subj_rel_logits = self.re_classifier_subj(subj_embs)  # bsz, num_label
        obj_rel_logits = self.re_classifier_obj(obj_embs)  # bsz, ent_len, num_label
        rel_logits = subj_rel_logits.unsqueeze(1) + obj_rel_logits  # bsz, ent_len, num_label

        aux_tok_logits = self.tok_classifier(aux_tok_embs)
        active_aux_tok_bool = aux_tok_labels[:, :, 0].view(-1) >= 0  # aux_tok_labels的最后一维是onehot label，而padding部分全为-1，正常的部分为0/1
        active_aux_tok_logits = aux_tok_logits.view(-1, aux_tok_logits.size(-1))[active_aux_tok_bool]
        active_aux_tok_labels = aux_tok_labels.view(-1, aux_tok_labels.size(-1))[active_aux_tok_bool]

        if mode == "train":
            ce_loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=LOSS_IGNORE_INDEX)

            rel_loss = ce_loss_fct(rel_logits.view(-1, self.num_rel_labels), rel_labels.view(-1))
            ner_loss = ce_loss_fct(obj_ner_logits.view(-1, self.num_ner_labels), ner_labels.view(-1))

            bce_loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            aux_tok_loss = bce_loss_fct(active_aux_tok_logits, active_aux_tok_labels.float())

            return {
                "rel_loss": rel_loss,
                "ner_loss": ner_loss,
                "aux_tok_loss": aux_tok_loss,
            }

        elif mode == "eval":
            return {
                "rel_logits": rel_logits,
                "ner_logits": obj_ner_logits,
                "tok_logits": active_aux_tok_logits,
                "tok_labels": active_aux_tok_labels,
            }


##############################################
# Load dataset class
##############################################


class TaskDataset(Dataset):
    """Load the dataset and return the tensor features of each data sample."""

    # 构造函数
    def __init__(self, tokenizer, file_path, do_eval):
        """Args:
        span_source: ner|pred_ner, Indicates which field is used to build candidate span pairs.
        """
        LOGGER.info("Loading data from %s", file_path)

        self.tokenizer = tokenizer
        self.do_eval = do_eval
        self.docs = [json.loads(line) for line in open(file_path, encoding="UTF-8")]
        self.src_path = file_path

        self.ner_label_list = TASK_NER_LABEL_LIST[CONFIG.task]
        self.ner_label2id_dict = {label: i for i, label in enumerate(self.ner_label_list)}  # ner_label_map

        self.rel_label_infodict = TASK_REL_LABEL_INFO_DICT[CONFIG.task]
        self.rel_label_infolist = sorted(self.rel_label_infodict.values(), key=lambda x: x["label_id"])
        self.inverse_rel_labelid_list = [i["inverse_label_id"] for i in self.rel_label_infolist]  # [0,1,2,8-12,3-7]

        # 用于eval的信息
        self.key2sample_dict = {}  # dict[sample_key_x3, subj_indices_x2] = sample

        # Each sentence in a doc is an sample
        self.samples = []

        for doc in tqdm(self.docs):
            self._convert_doc_to_samples(doc)

    def _convert_doc_to_samples(self, doc):
        """
        1.There is only one subj per sample. That is, each ent in ner will form a sample. There will be an additional empty ent with the "X" label as subj, which is placed behind the SEP.
        2. There are multiple objs per sample. All ent in ner/pred_ner are candidate obj. But no additional empty ent with X label will be created.
        """
        # Some metadata used to construct the sample
        doctoks, subtoks = [], []
        subtok_idx2doctok_idx, subtok2sentidx = [], []
        doctok_idx_indoc = 0
        for sent_idx, sent_doctoks in enumerate(doc["sentences"]):
            for doctok in sent_doctoks:
                _subwords = self.tokenizer.tokenize(doctok)
                doctoks.append(doctok)
                subtoks.extend(_subwords)
                subtok_idx2doctok_idx.extend([doctok_idx_indoc] * len(_subwords))
                subtok2sentidx.extend([sent_idx] * len(_subwords))
                doctok_idx_indoc += 1

        # 构造数据, 遍历每一个句子
        ners = doc[CONFIG.ner_tag_for_eval] if self.do_eval else doc["ner"]
        for sent_idx, sent_ners in enumerate(ners):
            ### A. 构造input ids

            # A1. 判断需要引入的上下文长度
            max_num_subtoks = CONFIG.max_seq_length - 4  # CLS SEP <subj> </subj>
            sent_subtok_start_idx = subtok2sentidx.index(sent_idx)
            sent_subtok_end_idx = rindex(subtok2sentidx, sent_idx)

            left_length = sent_subtok_start_idx  # 句子左边的长度, 正好是当前句子第一个tok的idx
            right_length = len(subtoks) - (sent_subtok_end_idx + 1)  # 句子右边的长度
            sentence_length = sent_subtok_end_idx - sent_subtok_start_idx + 1  # 句子的长度
            max_half_ctx_length = int((max_num_subtoks - sentence_length) / 2)  # 可用的单侧上下文长度
            assert sentence_length <= max_num_subtoks

            if sentence_length < max_num_subtoks:
                if left_length < right_length:
                    left_context_length = min(left_length, max_half_ctx_length)
                    right_context_length = min(right_length, max_num_subtoks - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, max_half_ctx_length)
                    left_context_length = min(left_length, max_num_subtoks - right_context_length - sentence_length)

            # A2. 在 CONFIG.max_seq_length 的条件下, 判断seq需要截断的位置
            seq_start_idx = sent_subtok_start_idx - left_context_length
            seq_end_idx = sent_subtok_end_idx + right_context_length
            input_tokens = subtoks[seq_start_idx : seq_end_idx + 1]
            # input_tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
            assert len(input_tokens) == max_num_subtoks or len(input_tokens) == len(subtoks)

            ### B. 遍历2次ner/pred_ner, 构造形成rel pair用于train或者eval。如果train, 则用的是ner, 如果是eval, 用的是pred_ner
            # 其中subj在train阶段还引入了一个额外的“标签为X”的ent。该方法会产生更多的训练样本。似乎可以提高结果
            subj_entities = list(sent_ners)
            # for radgraph, a few relations are linked crossing sentences, thus we expand range of candidiate ents
            expanded_sent_start_idx = sent_idx
            expanded_sent_end_idx = sent_idx
            if CONFIG.num_extra_sent_for_candi_ent > 0:
                expanded_sent_start_idx = sent_idx - CONFIG.num_extra_sent_for_candi_ent
                expanded_sent_start_idx = 0 if expanded_sent_start_idx < 0 else expanded_sent_start_idx
                expanded_sent_end_idx = sent_idx + CONFIG.num_extra_sent_for_candi_ent
                expanded_sent_end_idx = len(doc["sentences"]) - 1 if expanded_sent_end_idx >= len(doc["sentences"]) else expanded_sent_end_idx

            obj_entities, gold_rels = [], []
            gold_ner_indices2label_checkdict = {}
            for target_sent_idx in range(expanded_sent_start_idx, expanded_sent_end_idx + 1):
                obj_entities += [[ner[0], ner[1], ner[2], target_sent_idx] for ner in ners[target_sent_idx]]
                gold_rels += [rel for rel in doc["relations"][target_sent_idx]]
                gold_ner_indices2label_checkdict.update({(ner_info[0], ner_info[1]): ner_info[2] for ner_info in doc["ner"][target_sent_idx] if ner_info})

            # 跳过无ner标签的句子
            # if not subj_entities:
            #     continue

            if not self.do_eval:
                subj_entities.append((-1, -1, "X"))  # sepX
                gold_ner_indices2label_checkdict[(-1, -1)] = "X"
                # subj_entities.append((-2, -2, "X"))  # sentX
                # gt_ner_indices2label_checkdict[(-2, -2)] = "X"

            for subj_doctok_start_idx, subj_doctok_end_idx, subj_ner_label_str in subj_entities:
                subj_gold_ner_label_str = gold_ner_indices2label_checkdict.get((subj_doctok_start_idx, subj_doctok_end_idx), "X")
                subj_gold_ner_label_id = self.ner_label2id_dict[subj_gold_ner_label_str]
                subj_ner_label_id = self.ner_label2id_dict[subj_ner_label_str]

                # A3. 构造最终的input seq, 需要在subj subtoks的前后添加subj markers。
                # B1. 根据input seq, 确定subj markers的位置信息。将用于在模型中找到对应的emb
                if subj_doctok_start_idx >= 0:
                    subj_subtok_start_idx_old = subtok_idx2doctok_idx.index(subj_doctok_start_idx) - seq_start_idx
                    subj_subtok_end_idx_old = rindex(subtok_idx2doctok_idx, subj_doctok_end_idx) - seq_start_idx

                    final_input_tokens = [TOKEN_CONST["seq_start_token"]] + input_tokens[:subj_subtok_start_idx_old] + [TOKEN_CONST["subj_start_marker"]] + input_tokens[subj_subtok_start_idx_old : subj_subtok_end_idx_old + 1] + [TOKEN_CONST["subj_end_marker"]] + input_tokens[subj_subtok_end_idx_old + 1 :] + [TOKEN_CONST["seq_end_token"]]

                    # 更新subj marker和subtok的位置
                    subj_marker_start_idx = subj_subtok_start_idx_old + 1  # +1 because of CLS token
                    subj_subtok_start_idx = subj_marker_start_idx + 1
                    subj_subtok_end_idx = subj_subtok_end_idx_old + 2  # +2 because of CLS and <subj>
                    subj_marker_end_idx = subj_subtok_end_idx + 1

                elif subj_doctok_start_idx == -1:  # 在 SEP 后面添加 <subj> </subj>
                    final_input_tokens = [TOKEN_CONST["seq_start_token"]] + input_tokens + [TOKEN_CONST["seq_end_token"]] + [TOKEN_CONST["subj_start_marker"], TOKEN_CONST["subj_end_marker"]]

                    # 更新subj marker的位置。对于subtok来说, 其实是不存在的, 在训练中也不会使用到, 所以这里设为-1, 用于debug
                    subj_marker_start_idx = len(final_input_tokens) - 2
                    subj_subtok_start_idx = -1
                    subj_subtok_end_idx = -1
                    subj_marker_end_idx = subj_marker_start_idx + 1

                # B2. 根据input seq, 确定obj markers的位置信息。以及其他, 比如rel label等信息
                # 除了数据集标注的正向(subj, obj) rel label, 还新增了反向的(obj, subj) rel label
                rel_indices2label_checkdict = {}  # (pair_indices): rel_label_id
                for subj_start, subj_end, obj_start, obj_end, rel_label_str in gold_rels:
                    rel_label_info = self.rel_label_infodict[rel_label_str]

                    pair_indices = (subj_start, subj_end, obj_start, obj_end)
                    rel_indices2label_checkdict[pair_indices] = [rel_label_info["label_str"], rel_label_info["label_id"]]

                    inverse_pair_indices = (obj_start, obj_end, subj_start, subj_end)
                    rel_indices2label_checkdict[inverse_pair_indices] = [rel_label_info["inverse_label_str"], rel_label_info["inverse_label_id"]]  # 注意, 对称标签的反向标签依然是其自身

                rel_instances = []
                for obj_idx, (obj_doctok_start_idx, obj_doctok_end_idx, obj_ner_label_str, obj_sent_idx) in enumerate(obj_entities):
                    obj_gold_ner_label_str = gold_ner_indices2label_checkdict.get((obj_doctok_start_idx, obj_doctok_end_idx), "X")
                    obj_gold_ner_label_id = self.ner_label2id_dict[obj_gold_ner_label_str]
                    obj_ner_label_id = self.ner_label2id_dict[obj_ner_label_str]

                    # 更新obj subtoks的位置。影响因素包括:
                    # 1) 上下文长度限制导致的序列偏移, 2) 添加的CLS tok
                    obj_subtok_start_idx = subtok_idx2doctok_idx.index(obj_doctok_start_idx) - seq_start_idx + 1
                    obj_subtok_end_idx = rindex(subtok_idx2doctok_idx, obj_doctok_end_idx) - seq_start_idx + 1

                    # 3) subj的前后添加了marker。注意, 这里判断的是doctok的位置, 修改的是subtok的位置。比较符号不同的原因: marker插入的是start的左边, 或是end的右边
                    if obj_doctok_start_idx >= subj_doctok_start_idx:
                        obj_subtok_start_idx += 1
                        if obj_doctok_start_idx > subj_doctok_end_idx:
                            obj_subtok_start_idx += 1

                    if obj_doctok_end_idx >= subj_doctok_start_idx:
                        obj_subtok_end_idx += 1
                        if obj_doctok_end_idx > subj_doctok_end_idx:
                            obj_subtok_end_idx += 1

                    # 构建obj markers的位置。该位置为
                    obj_marker_start_idx = len(final_input_tokens) + obj_idx
                    obj_marker_end_idx = len(final_input_tokens) + len(sent_ners) + obj_idx
                    assert obj_marker_end_idx < 512

                    # 获取rel label
                    pair_indices = (subj_doctok_start_idx, subj_doctok_end_idx, obj_doctok_start_idx, obj_doctok_end_idx)
                    rel_label_str, rel_label_id = rel_indices2label_checkdict.get(pair_indices, ("X", 0))  # label "X"

                    # 计算subj 和 obj 之间的 ctx: 这里采用的是包括subj start和obj end之间的所有token
                    pair_ctx_boundries = [subj_marker_start_idx, subj_marker_end_idx, obj_subtok_start_idx, obj_subtok_end_idx]
                    pair_ctx_boundries.sort()
                    if (subj_subtok_start_idx == obj_subtok_start_idx and subj_subtok_end_idx == obj_subtok_end_idx) or subj_doctok_start_idx < 0:  # 排除subjX，排除subj=obj
                        ctx_left_idx = -1
                        ctx_right_idx = -1
                    else:
                        ctx_left_idx = pair_ctx_boundries[0]
                        ctx_right_idx = pair_ctx_boundries[-1]

                    rel_instances.append(
                        {
                            "obj_sent_idx": obj_sent_idx,
                            "marker_indices": (obj_marker_start_idx, obj_marker_end_idx),
                            "doctok_indices": (obj_doctok_start_idx, obj_doctok_end_idx),
                            "subtok_indices": (obj_subtok_start_idx, obj_subtok_end_idx),
                            "pred_ner_label_str_id": (obj_ner_label_str, obj_ner_label_id),  # train时是ner,  eval时是pred_ner
                            "gold_ner_label_str_id": (obj_gold_ner_label_str, obj_gold_ner_label_id),
                            "rel_label_str_id": (rel_label_str, rel_label_id),
                            "ctx_indices": (ctx_left_idx, ctx_right_idx),
                            "debug": [
                                (subj_doctok_start_idx, subj_doctok_end_idx, obj_doctok_start_idx, obj_doctok_end_idx, rel_label_str),
                                (subj_subtok_start_idx, subj_subtok_end_idx, obj_subtok_start_idx, obj_subtok_end_idx, rel_label_id),
                                (subj_marker_start_idx, subj_marker_end_idx, obj_marker_start_idx, obj_marker_end_idx, rel_label_id),
                            ],
                        }
                    )

                # 当rel pair的数量大于CONFIG.max_pair_length时, 就增加一个样本（已取消）
                instance_idx = 0
                sample_key = (doc["doc_key"], sent_idx, instance_idx, subj_doctok_start_idx, subj_doctok_end_idx)
                input_token_len_without_objmarkers = len(final_input_tokens)
                final_input_tokens += [TOKEN_CONST["obj_start_marker"]] * len(rel_instances) + [TOKEN_CONST["obj_end_marker"]] * len(rel_instances)
                assert len(final_input_tokens) < 512

                ### A3. 辅助任务2：根据每个tok所属的rel，构造onehot多标签。只包含目标句子的subtok (不包括subj marker)
                sent_subtok_start_idx_inseq = sent_subtok_start_idx - seq_start_idx + 1  # 1=CLS
                sent_subtok_end_idx_inseq = sent_subtok_end_idx - seq_start_idx + 3  # 3=CLS,<subj>,</subj>
                tok_idx_onehot_rel_label_dict = {}
                for subtok_idx in range(sent_subtok_start_idx_inseq, sent_subtok_end_idx_inseq + 1):
                    if subtok_idx == subj_marker_start_idx or subtok_idx == subj_marker_end_idx:
                        continue
                    tok_idx_onehot_rel_label_dict[subtok_idx] = [1] + [0] * (len(self.rel_label_infodict) - 1)

                for rel_obj_info in rel_instances:
                    obj_subtok_start_idx, obj_subtok_end_idx = rel_obj_info["subtok_indices"]
                    rel_label_id = rel_obj_info["rel_label_str_id"][1]

                    if rel_label_id == 0:
                        continue

                    for subtok_idx in range(obj_subtok_start_idx, obj_subtok_end_idx + 1):
                        if subtok_idx in tok_idx_onehot_rel_label_dict:
                            onehot = tok_idx_onehot_rel_label_dict[subtok_idx]
                            onehot[rel_label_id] = 1
                            onehot[0] = 0

                    if subj_subtok_start_idx >= 0:  # 当数据增强时，没有对应是subj tok，subj_subtok_start_idx为-1，需要排除
                        for subtok_idx in range(subj_subtok_start_idx, subj_subtok_end_idx + 1):
                            if subtok_idx in tok_idx_onehot_rel_label_dict:
                                onehot = tok_idx_onehot_rel_label_dict[subtok_idx]
                                onehot[rel_label_id] = 1
                                onehot[0] = 0

                sample = {
                    "doc_key": doc["doc_key"],
                    "sent_id": sent_idx,
                    "expanded_sent_range": (expanded_sent_start_idx, expanded_sent_end_idx),
                    "instance_id": instance_idx,
                    "input_token_len_without_objmarkers": input_token_len_without_objmarkers,
                    "input_tokens": final_input_tokens,
                    "input_ids": self.tokenizer.convert_tokens_to_ids(final_input_tokens),
                    "info_dict": {
                        "subj": {
                            "marker_indices": (subj_marker_start_idx, subj_marker_end_idx),
                            "doctok_indices": (subj_doctok_start_idx, subj_doctok_end_idx),
                            "subtok_indices": (subj_subtok_start_idx, subj_subtok_end_idx),
                            "pred_ner_label_str_id": (subj_ner_label_str, subj_ner_label_id),
                            "gold_ner_label_str_id": (subj_gold_ner_label_str, subj_gold_ner_label_id),
                        },
                        "obj_list": rel_instances,
                    },
                    "sample_key": sample_key,
                    "tok_idx2onehot_rel_label_dict": tok_idx_onehot_rel_label_dict,
                }
                self.samples.append(sample)
                self.key2sample_dict[sample_key] = sample  # 由三元组组成的sample_key, 唯一定位样本

    # 返回数据集大小
    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch):
    max_input_len = max([len(sample["input_ids"]) for sample in batch])
    max_num_objs = max([len(sample["info_dict"]["obj_list"]) for sample in batch])
    max_num_aux_toks = max([len(sample["tok_idx2onehot_rel_label_dict"]) for sample in batch])

    all_input_ids = []
    all_input_masks = []
    all_position_ids = []
    all_subj_indices = []
    all_rel_info = []
    all_aux_tok_info = []
    for sample in batch:
        ### input ids: (seq + <obj>s + </obj>s + pads)
        input_ids = copy.deepcopy(sample["input_ids"])
        curr_seq_len = len(input_ids)
        num_pad_tokens = max_input_len - curr_seq_len

        input_ids += [TOKEN_CONST["pad_token_id"]] * num_pad_tokens

        ### attention mask: (bsz, inpu_len, input_len)
        seq_without_objmarkers_len = sample["input_token_len_without_objmarkers"]
        attention_mask = np.zeros((max_input_len, max_input_len))
        attention_mask[:seq_without_objmarkers_len, :seq_without_objmarkers_len] = 1

        ### position ids: (bsz, inpu_len)
        position_ids = list(range(curr_seq_len)) + [TOKEN_CONST["pad_token_id"]] * num_pad_tokens

        ### rel info (bsz, 16, n): 包括 ctx indices, obj indices, rel labels, obj ner labels
        ### 同时调整 attention mask and position ids
        rel_info = []
        for obj_info_dict in sample["info_dict"]["obj_list"]:
            # obj marker (start/end) 与对应的 obj subtok (start/end) 共享 position id
            obj_marker_start_idx = obj_info_dict["marker_indices"][0]
            obj_marker_end_idx = obj_info_dict["marker_indices"][1]

            position_ids[obj_marker_start_idx] = obj_info_dict["subtok_indices"][0]
            position_ids[obj_marker_end_idx] = obj_info_dict["subtok_indices"][1]

            # attention mask: 同一个obj的两个marker:
            for marker_idx1 in [obj_marker_start_idx, obj_marker_end_idx]:
                attention_mask[marker_idx1, :seq_without_objmarkers_len] = 1  # 1) att所有非obj markers,
                for marker_idx2 in [obj_marker_start_idx, obj_marker_end_idx]:
                    attention_mask[marker_idx1, marker_idx2] = 1  # 2) start end 之间相互att

            # 构造 rel info
            rel_info.append(
                [
                    obj_info_dict["marker_indices"][0],
                    obj_info_dict["marker_indices"][1],
                    obj_info_dict["rel_label_str_id"][1],
                    obj_info_dict["ctx_indices"][0],
                    obj_info_dict["ctx_indices"][1],
                    obj_info_dict["pred_ner_label_str_id"][1],  # 这个在eval时, 模型是不使用的。仅用于obj的aux cls
                    obj_info_dict["subtok_indices"][0],
                    obj_info_dict["subtok_indices"][1],
                ]
            )

        # rel_info添加padding: 1) label=-100时, 交叉熵会忽略该项；2) marker_indices理论上可以是任何值, 因为不会贡献loss; 3) ctx_indices设为 LOSS_IGNORE_INDEX,
        num_pad_objs = max_num_objs - len(sample["info_dict"]["obj_list"])
        rel_info += [[0, 0, LOSS_IGNORE_INDEX, -1, -1, LOSS_IGNORE_INDEX, 0, 0]] * num_pad_objs

        ### subj_indices
        all_subj_indices.append(sample["info_dict"]["subj"]["marker_indices"] + sample["info_dict"]["subj"]["subtok_indices"])

        ### aux_tok_info: 包括tok的indices和onehot labels
        # 对于padding部分，idx为0，onehot label为全-1。需要再model中计算loss时，对padding进行排除
        aux_tok_info = []
        for tok_idx, onehot_label in sample["tok_idx2onehot_rel_label_dict"].items():
            aux_tok_info.append([tok_idx] + onehot_label)
        curr_num_aux_toks = len(sample["tok_idx2onehot_rel_label_dict"])
        num_pad_aux_toks = max_num_aux_toks - curr_num_aux_toks
        aux_tok_info += [[0] + [-1] * (len(aux_tok_info[0]) - 1)] * num_pad_aux_toks

        all_input_ids.append(input_ids)
        all_input_masks.append(attention_mask)
        all_position_ids.append(position_ids)
        all_rel_info.append(rel_info)
        all_aux_tok_info.append(aux_tok_info)

    return {
        "input_ids": torch.tensor(all_input_ids).to(DEVICE),
        "input_masks": torch.tensor(np.array(all_input_masks), dtype=torch.int64).to(DEVICE),
        "position_ids": torch.tensor(all_position_ids).to(DEVICE),
        "subj_indices": torch.tensor(all_subj_indices, dtype=torch.int64).to(DEVICE),
        "rel_info": torch.tensor(all_rel_info, dtype=torch.int64).to(DEVICE),
        "sample_keys": [i["sample_key"] for i in batch],
        "aux_tok_info": torch.tensor(all_aux_tok_info, dtype=torch.int64).to(DEVICE),
    }


class StableRandomSampler:
    def __init__(self, data_source, num_epoch):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.seeds = iter(torch.empty((num_epoch), dtype=torch.int64).random_().tolist())

    def __iter__(self):
        seed = next(self.seeds)
        generator = torch.Generator()
        generator.manual_seed(seed)
        rand_list = torch.randperm(self.num_samples, generator=generator).tolist()
        LOGGER.debug("Shuffle batches with seed = %d", seed)

        yield from rand_list

    def __len__(self) -> int:
        return self.num_samples


##############################################
# Train and eval
##############################################


def train(model, tokenizer):
    # 加载数据
    train_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "train.json")
    train_dataset = TaskDataset(tokenizer=tokenizer, file_path=train_data_path, do_eval=False)
    train_sampler = StableRandomSampler(train_dataset, CONFIG.num_epoch)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=CONFIG.train_batch_size, drop_last=True)

    dev_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "dev.json")
    dev_dataset = TaskDataset(tokenizer=tokenizer, file_path=dev_data_path, do_eval=True)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)

    test_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "test.json")
    test_dataset = TaskDataset(tokenizer=tokenizer, file_path=test_data_path, do_eval=True)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)

    # 优化器和调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": CONFIG.weight_decay}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG.learning_rate, eps=1e-8)
    total_num_steps = len(train_dataloader) // CONFIG.grad_accum_steps * CONFIG.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * CONFIG.warmup_proportion), num_training_steps=total_num_steps)

    log_result_dict = {
        "best_dev_f1": 0.0,
        "best_dev_test_f1": 0.0,
        "best_dev_at": 0,
        "best_test_f1": 0.0,
        "best_test_dev_f1": 0.0,
        "best_test_at": 0,
    }
    log_info = {
        "batch_trained_examples": 0,
        "global_step": 0,
        "batch_loss": 0,
        "batch_rel_loss": 0,
        "batch_ner_loss": 0,
        "batch_tok_loss": 0,
        "checkpoint_saved_at": 0,
        "rel": log_result_dict.copy(),
        "rel+": log_result_dict.copy(),
    }

    LOGGER.info("****************************** Training ******************************")
    LOGGER.info("Total samples = %d, batch size = %d", len(train_dataset), CONFIG.train_batch_size)
    LOGGER.info("Total epochs = %d, total iterations per epoch = %d", CONFIG.num_epoch, len(train_dataloader))
    LOGGER.info("Total optimization steps = %d", total_num_steps)
    LOGGER.info("Gradient accumulation steps = %d", CONFIG.grad_accum_steps)
    if CONFIG.grad_accum_steps > 1:
        LOGGER.info("Gradient accumulation at iter=%s per epoch", log_info["grad_accum_at_iter_n_per_epoch"])

    model.zero_grad()

    def eval_model_and_save_checkpoint_at(curr_epoch_or_step):
        dev_out = evaluate(model, dev_dataloader)
        test_out = evaluate(model, test_dataloader)

        # print current result
        LOGGER.info("****************************** Checkpoint ******************************")
        LOGGER.info("(curr_epoch_or_step=%d) Current [dev] rel f1: %.3f, rel+ f1: %.3f, avg f1: %.3f,", curr_epoch_or_step, dev_out["rel"] * 100, dev_out["rel+"] * 100, (dev_out["rel"] + dev_out["rel+"]) / 2 * 100)
        LOGGER.debug("(curr_epoch_or_step=%d) Current [test] rel f1: %.3f, rel+ f1: %.3f, avg f1: %.3f,", curr_epoch_or_step, test_out["rel"] * 100, test_out["rel+"] * 100, (test_out["rel"] + test_out["rel+"]) / 2 * 100)

        # save checkpoint
        for eval_type in ["rel", "rel+"]:
            dev_f1 = dev_out[eval_type]
            test_f1 = test_out[eval_type]

            achieved_best_dev = True if dev_f1 > log_info[eval_type]["best_dev_f1"] else False
            achieved_best_test = True if test_f1 > log_info[eval_type]["best_test_f1"] else False

            if achieved_best_dev:
                log_info[eval_type]["best_dev_f1"] = dev_f1
                log_info[eval_type]["best_dev_test_f1"] = test_f1
                log_info[eval_type]["best_dev_at"] = curr_epoch_or_step
                LOGGER.info("(curr_epoch_or_step=%d) !!! Achieved best %s dev f1: %.3f", curr_epoch_or_step, eval_type, dev_f1 * 100)

                if eval_type == "rel+":
                    log_info["checkpoint_saved_at"] = curr_epoch_or_step
                    LOGGER.info("Saving model checkpoint to %s", CONFIG.output_model_dir)
                    model.save_pretrained(CONFIG.output_model_dir)

            if achieved_best_test:
                log_info[eval_type]["best_test_dev_f1"] = dev_f1
                log_info[eval_type]["best_test_f1"] = test_f1
                log_info[eval_type]["best_test_at"] = curr_epoch_or_step
                LOGGER.debug("(curr_epoch_or_step=%d) !!! Achieved best %s test f1: %.3f", curr_epoch_or_step, eval_type, test_f1 * 100)

        # print the best result
        LOGGER.info("------------------------------------")
        for eval_type in ["rel", "rel+"]:
            LOGGER.info(
                "###### Best [dev] %s: dev_f1: %.3f, test_f1: %.3f, at_epoch_or_step=%d",
                eval_type,
                log_info[eval_type]["best_dev_f1"] * 100,
                log_info[eval_type]["best_dev_test_f1"] * 100,
                log_info[eval_type]["best_dev_at"],
            )
            LOGGER.debug(
                "       Best [test] %s: dev_f1: %.3f, test_f1: %.3f, at_epoch_or_step=%d",
                eval_type,
                log_info[eval_type]["best_test_dev_f1"] * 100,
                log_info[eval_type]["best_test_f1"] * 100,
                log_info[eval_type]["best_test_at"],
            )
        LOGGER.info("------------------------------------")

    for curr_epoch in range(CONFIG.num_epoch):
        for curr_iter, batch_inputs_dict in enumerate(train_dataloader):
            model.train()
            out = model(batch_inputs_dict, mode="train")

            rel_loss = out["rel_loss"]
            ner_loss = out["ner_loss"]
            aux_tok_loss = out["aux_tok_loss"]
            loss = rel_loss + ner_loss + aux_tok_loss

            # Gradient accumulation;
            # for the last few mini-batches, the accum step may be smaller than other mini-batches
            if CONFIG.grad_accum_steps > 1:
                loss = loss / CONFIG.grad_accum_steps
                rel_loss = rel_loss / CONFIG.grad_accum_steps
                ner_loss = ner_loss / CONFIG.grad_accum_steps
                aux_tok_loss = aux_tok_loss / CONFIG.grad_accum_steps

            loss.backward()

            curr_batch_size = batch_inputs_dict["input_ids"].shape[0]
            log_info["batch_trained_examples"] += curr_batch_size
            log_info["batch_loss"] += loss.item() * curr_batch_size
            log_info["batch_rel_loss"] += rel_loss.item() * curr_batch_size
            log_info["batch_ner_loss"] += ner_loss.item() * curr_batch_size
            log_info["batch_tok_loss"] += aux_tok_loss.item() * curr_batch_size

            # Update model parameters
            if (curr_iter + 1) % CONFIG.grad_accum_steps == 0:
                if CONFIG.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()  # optimizer.zero_grad()

                log_info["global_step"] += 1

                # print loss
                if log_info["global_step"] == 1 or log_info["global_step"] % CONFIG.print_loss_per_n_step == 0:
                    LOGGER.debug(
                        "Epoch=%d, iter=%d, steps=%d, loss=%.9f (rel=%.6f, aux_ner=%.6f, aux_tok=%.6f)",
                        curr_epoch,
                        curr_iter,
                        log_info["global_step"],
                        log_info["batch_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_rel_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_ner_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_tok_loss"] / log_info["batch_trained_examples"],
                    )
                    log_info["batch_trained_examples"] = 0
                    log_info["batch_loss"] = 0
                    log_info["batch_rel_loss"] = 0
                    log_info["batch_ner_loss"] = 0
                    log_info["batch_tok_loss"] = 0

                # checkpoint, eval at specific steps:
                if log_info["global_step"] % CONFIG.eval_per_steps == 0:
                    eval_model_and_save_checkpoint_at(curr_epoch_or_step=log_info["global_step"])

        # checkpoint, eval at the end of each epoch:
        if curr_epoch >= (CONFIG.num_epoch / 2 - 1) and curr_epoch % CONFIG.eval_per_epoch == 0:
            eval_model_and_save_checkpoint_at(curr_epoch_or_step=curr_epoch)

    LOGGER.info("Best checkpoint saved at: %s", log_info["checkpoint_saved_at"])


def evaluate(model, eval_dataloader):
    eval_dataset = eval_dataloader.dataset

    sum_ner, sum_rel = 0, 0
    for doc in eval_dataset.docs:
        for sent_ner in doc["ner"]:
            sum_ner += len(sent_ner)
        for sent_rel in doc["relations"]:
            sum_rel += len(sent_rel)

    task_f1 = {}
    eval_results = {
        "binary_rel": {  # whether two spans have relation
            "num_gold_label": sum_rel,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "rel": {  # the boundaries of two spans are correct and the predicted relation type is correct
            "num_gold_label": sum_rel,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "rel+": {  # predicted entity types also must be correct
            "num_gold_label": sum_rel,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "aux_ner": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "aux_tok": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
    }

    LOGGER.info("****************************** Evaluation ******************************")
    LOGGER.info("Source = %s", eval_dataset.src_path)
    LOGGER.info("Batch size = %d", CONFIG.eval_batch_size)
    LOGGER.info("Num samples = %d", len(eval_dataset))

    pred_result_dict = defaultdict(dict)  # (doc_key, sent_id, sample_id): ((subj start,end),(obj start,end)): (rel pred logits) (pred subj_ner, obj_ner), (gt subj_ner, obj_ner)

    model.eval()
    for batch_inputs_dict in eval_dataloader:
        with torch.no_grad():
            output = model(batch_inputs_dict, mode="eval")
            rel_logits = output["rel_logits"]
            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)  # torch.nn.functional.softmax(logits, dim=-1)
            rel_logits = rel_logits.cpu().numpy()

            aux_ner_preds = torch.argmax(output["ner_logits"], dim=-1).cpu().numpy()

            # 把所有结果储存到 pred_result_dict 中
            # batch_inputs_dict中, 每个subj是一个样本；但在 pred_result_dict中, 需要合并为: 一个句子为一个item
            for i, sample_key in enumerate(batch_inputs_dict["sample_keys"]):
                doc_key, _, _, _, _ = sample_key  # 后续按照文档进行预测
                src_sample = eval_dataset.key2sample_dict[sample_key]
                sample_info_dict = src_sample["info_dict"]
                for j, obj_info_dict in enumerate(sample_info_dict["obj_list"]):
                    subj_indices = sample_info_dict["subj"]["doctok_indices"]
                    obj_indices = obj_info_dict["doctok_indices"]

                    pred_result_dict[doc_key][(subj_indices, obj_indices)] = {
                        "rel_logits": rel_logits[i, j].tolist(),
                        "rel_gold_label_id": obj_info_dict["rel_label_str_id"][1],
                        "subj_ner_label_id": {
                            "pred": sample_info_dict["subj"]["pred_ner_label_str_id"][1],
                            "gold": sample_info_dict["subj"]["gold_ner_label_str_id"][1],
                        },
                        "obj_ner_label_id": {
                            "pred": obj_info_dict["pred_ner_label_str_id"][1],
                            "gold": obj_info_dict["gold_ner_label_str_id"][1],
                        },
                    }

                    # aux ner 可以直接评估
                    aux_ner_pred, aux_ner_gold = aux_ner_preds[i, j], obj_info_dict["gold_ner_label_str_id"][1]
                    if aux_ner_pred > 0:
                        eval_results["aux_ner"]["num_pred_label"] += 1
                        if aux_ner_gold > 0:
                            eval_results["aux_ner"]["num_gold_label"] += 1
                            if aux_ner_pred == aux_ner_gold:
                                eval_results["aux_ner"]["num_correct_label"] += 1

            # aux_tok可以直接评估，且其结果已经展开为(bsz * num_tok, num_labels)，且padding部分已经去掉
            threshold = 0.5
            aux_tok_probs = torch.sigmoid(output["tok_logits"])
            aux_tok_preds = torch.where(aux_tok_probs > threshold, 1, 0).cpu().numpy()
            aux_tok_golds = output["tok_labels"].cpu().numpy()
            for gold, pred in zip(aux_tok_golds, aux_tok_preds):
                for i, (y_gold, y_pred) in enumerate(zip(gold, pred)):
                    if i == 0:
                        continue
                    if y_gold == 1:
                        eval_results["aux_tok"]["num_gold_label"] += 1
                    if y_pred == 1:
                        eval_results["aux_tok"]["num_pred_label"] += 1
                    if y_gold == 1 and y_pred == 1:
                        eval_results["aux_tok"]["num_correct_label"] += 1

    # ****************************** Post-processing for inference and evaluation ******************************

    for doc_key, pair_dict in pred_result_dict.items():
        doc_candidate_results = []  # [(-9.523009538650513, (43, 46), (34, 34), 4, 'Task', 'Generic'), ...]
        # 遍历一个句子里的所有pairs。pairs是所有pred ent的两两组合。
        # 因为subj obj可以对调位置, 所以要合并2个logits。subj和obj可能相同,如果subj obj相同或者标签为 X, 则跳过。
        # 每一对组合, 只会产生一个结果: subj-obj == obj-subj (因此评估结果是正确的)
        # 对于结果为非对称标签, 需要根据预测的rel标签判断subj和obj的位置, 并把反向标签转为正向标签
        # 对于结果为对称标签, subj和obj的顺序无所谓, 因为我们在构造数据时, 为正序和倒序都分配了相同的标签
        visited = set([])
        for (subj_indices, obj_indices), result_info_dict in pair_dict.items():
            subj_obj_rel_logit = list(result_info_dict["rel_logits"])  # copy a new list
            subj_pred_ner_label = result_info_dict["subj_ner_label_id"]
            obj_pred_ner_label = result_info_dict["obj_ner_label_id"]

            if (subj_indices, obj_indices) in visited:
                continue
            if subj_indices == obj_indices:
                continue
            visited.add((subj_indices, obj_indices))
            visited.add((obj_indices, subj_indices))

            inverse_result_info_dict = pair_dict[(obj_indices, subj_indices)]
            obj_subj_rel_logits = inverse_result_info_dict["rel_logits"]

            # inverse_rel_labelid_list = [0,1,2,8-12,3-7]
            # 1) zip(a,b)把ab打包成由二元组组成的list,
            # 2) sorted把二元组按照第一个位置的元素进行升序排序,
            # 3) zip(*)把二元组unzip成两个list
            # 最后的结果就是把logits中非对称标签的位置对调, 即(8-12)和(3-7)的部分
            _, inverse_obj_subj_rel_logits = zip(*sorted(zip(eval_dataset.inverse_rel_labelid_list, obj_subj_rel_logits)))

            # 合并 rel logits
            rel_logits = [x + y for x, y in zip(subj_obj_rel_logit, inverse_obj_subj_rel_logits)]
            pred_rel_label_id = np.argmax(rel_logits)

            # 根据logits, 转化标签。如果预测的结果是反向标签, 那么就对调subj obj的位置
            gold_rel_label_id = result_info_dict["rel_gold_label_id"]
            if pred_rel_label_id > 0:
                pred_rel_label = eval_dataset.rel_label_infolist[pred_rel_label_id]
                if pred_rel_label["type"] == "inverse":
                    pred_rel_label_id = pred_rel_label["inverse_label_id"]
                    gold_rel_label_id = inverse_result_info_dict["rel_gold_label_id"]
                    pred_rel_label = eval_dataset.rel_label_infolist[pred_rel_label_id]
                    subj_indices, obj_indices = obj_indices, subj_indices
                    subj_pred_ner_label, obj_pred_ner_label = obj_pred_ner_label, subj_pred_ner_label

                pred_score = rel_logits[pred_rel_label_id]
                doc_candidate_results.append(
                    {
                        "pred_score": pred_score,
                        "subj_indices": subj_indices,
                        "obj_indices": obj_indices,
                        # "pred_rel_label": pred_rel_label,
                        "rel_label_pred_vs_gold": (pred_rel_label_id, gold_rel_label_id),
                        "subj_ner_label_pred_vs_gold": (subj_pred_ner_label["pred"], subj_pred_ner_label["gold"]),
                        "obj_ner_label_pred_vs_gold": (obj_pred_ner_label["pred"], obj_pred_ner_label["gold"]),
                    }
                )

        # 过滤预测结果:
        # 1. 按照得分排序 (越接近0, 越certain)
        # 2. 遍历sentence_results, 并过滤不符合条件的结果:
        # 3. 如果当前的pair与filtered_results中的pair在位置上存在部分重叠, 则丢弃当前的pair, 保留得分更高的 (使用pred ner时, 确实有可能重叠)
        # 比如:  (-9.523009538650513, (43, 46), (34, 34), 4, 'Task', 'Generic') 和 (-9.656012773513794, (43, 45), (34, 34), 4, 'Task', 'Generic')
        # 这一步相当于在预测时重叠的ner中, 选择rel得分最高的一个来作为pair的结果 TODO 可以去掉这部分, 看看会有什么后果
        def should_drop_candidate(candidate, valid_results):
            for valid_result in valid_results:
                if candidate["rel_label_pred_vs_gold"][0] == valid_result["rel_label_pred_vs_gold"][0] and is_indices_overlap(candidate["subj_indices"], valid_result["subj_indices"]) and is_indices_overlap(candidate["obj_indices"], valid_result["obj_indices"]):
                    yield True
                else:
                    yield False

        valid_results = []
        for candi_result in sorted(doc_candidate_results, key=lambda x: -x["pred_score"]):
            do_drop = any(should_drop_candidate(candi_result, valid_results))
            if not do_drop:
                valid_results.append(candi_result)

        # 真正的eval
        for result_item in valid_results:
            eval_results["binary_rel"]["num_pred_label"] += 1
            eval_results["rel"]["num_pred_label"] += 1
            eval_results["rel+"]["num_pred_label"] += 1
            rel_pred, rel_gold = result_item["rel_label_pred_vs_gold"]
            assert rel_pred > 0

            if rel_gold > 0:
                eval_results["binary_rel"]["num_correct_label"] += 1

                if rel_pred == rel_gold:
                    # the boundaries of two spans are correct and the predicted relation type is correct
                    eval_results["rel"]["num_correct_label"] += 1

                    subj_ner_pred, subj_ner_gold = result_item["subj_ner_label_pred_vs_gold"]
                    obj_ner_pred, obj_ner_gold = result_item["obj_ner_label_pred_vs_gold"]

                    if subj_ner_pred == subj_ner_gold and obj_ner_pred == obj_ner_gold:
                        # predicted entity types also must be correct
                        eval_results["rel+"]["num_correct_label"] += 1

    # Calculate the results
    for eval_field, result_dict in eval_results.items():
        num_corr = result_dict["num_correct_label"]
        num_pred = result_dict["num_pred_label"]
        num_gold = result_dict["num_gold_label"]
        p = num_corr / num_pred if num_corr > 0 else 0.0
        r = num_corr / num_gold if num_corr > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if num_corr > 0 else 0.0
        LOGGER.info("[%s]: P: %.5f, R: %.5f, 【F1: %.3f】", eval_field, p, r, f1 * 100)
        task_f1[eval_field] = f1

    return task_f1


##############################################
# functions
##############################################


def is_indices_overlap(ent_a, ent_b):
    if ent_b[0] <= ent_a[0] and ent_a[0] <= ent_b[1]:
        return True
    if ent_a[0] <= ent_b[0] and ent_b[0] <= ent_a[1]:
        return True
    return False


##############################################
# Script arguments
##############################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_name", type=str)

    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_extra_sent_for_candi_ent", type=int, default=0)
    parser.add_argument("--bert_path", type=str)
    parser.add_argument("--data_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        CONFIG.output_name = args.output_name
    if args.from_bash:
        CONFIG.output_name = args.output_name
        CONFIG.task = args.task
        CONFIG.seed = args.seed
        CONFIG.data_dir[CONFIG.task] = args.data_path
        CONFIG.model_name_or_path[CONFIG.task] = args.bert_path = args.bert_path
        # CONFIG.num_extra_sent_for_candi_ent = args.num_extra_sent_for_candi_ent

    # Reproducibility
    set_seed(CONFIG.seed)

    # Create output directory, and logging handler to file
    CONFIG.output_dir = os.path.join(CONFIG.output_dir, CONFIG.output_name)
    CONFIG.output_model_dir = os.path.join(CONFIG.output_model_dir, CONFIG.output_name)
    check_and_mkdir(CONFIG.output_dir)
    check_and_mkdir(CONFIG.output_model_dir)
    if CONFIG.do_train:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_dir, "train.log"), "w")
    elif CONFIG.do_eval:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_dir, "eval.log"), "w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.info([i for i in vars(CONFIG).items() if i[0][0] != "_"])

    ner_label_list = TASK_NER_LABEL_LIST[CONFIG.task]
    rel_label_infodict = TASK_REL_LABEL_INFO_DICT[CONFIG.task]
    num_ner_labels = len(ner_label_list)
    num_rel_labels = len(rel_label_infodict)

    start = time.time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CONFIG.do_train:
        # Initial model
        LOGGER.info("****************************** Initialize model ******************************")
        LOGGER.info("Encoder & Tokenizer from: %s", CONFIG.model_name_or_path[CONFIG.task])
        LOGGER.info("num_rel_labels = %s, num_ner_labels = %s", num_rel_labels, num_ner_labels)
        LOGGER.info("Device: %s", DEVICE)
        model_config = BertConfig.from_pretrained(CONFIG.model_name_or_path[CONFIG.task])
        model_config.num_rel_labels = num_rel_labels
        model_config.num_ner_labels = num_ner_labels
        tokenizer = BertTokenizer.from_pretrained(CONFIG.model_name_or_path[CONFIG.task])
        TOKEN_CONST.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "seq_start_token": tokenizer.cls_token,
                "seq_end_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        )
        model = REModel.from_pretrained(CONFIG.model_name_or_path[CONFIG.task], config=model_config)
        model.to(DEVICE)

        # weights sharing for subj and obj markers
        subj_start_marker_id = tokenizer.convert_tokens_to_ids(TOKEN_CONST["subj_start_marker"])
        subj_end_marker_id = tokenizer.convert_tokens_to_ids(TOKEN_CONST["subj_end_marker"])
        obj_start_marker_id = tokenizer.convert_tokens_to_ids(TOKEN_CONST["obj_start_marker"])
        obj_end_marker_id = tokenizer.convert_tokens_to_ids(TOKEN_CONST["obj_end_marker"])

        mask_id = tokenizer.convert_tokens_to_ids(TOKEN_CONST["mask_token"])
        subj_tok_id = tokenizer.convert_tokens_to_ids("subject")
        obj_tok_id = tokenizer.convert_tokens_to_ids("object")

        word_embeddings = model.bert.embeddings.word_embeddings.weight.data
        word_embeddings[subj_start_marker_id].copy_(word_embeddings[subj_tok_id])
        word_embeddings[subj_end_marker_id].copy_(word_embeddings[mask_id])
        word_embeddings[subj_start_marker_id].copy_(word_embeddings[obj_tok_id])
        word_embeddings[obj_end_marker_id].copy_(word_embeddings[mask_id])

        # 调用train方法, 开始模型训练
        train(model, tokenizer)

        # 保存tokenizer
        tokenizer.save_pretrained(CONFIG.output_model_dir)
        LOGGER.info("Tokenizer is saved to: %s", CONFIG.output_model_dir)

    if CONFIG.do_eval:
        if os.path.exists(CONFIG.fine_tuned_model_path):
            fine_tuned_model_path = CONFIG.fine_tuned_model_path
        else:
            fine_tuned_model_path = CONFIG.output_model_dir
        LOGGER.info("****************************** Load model from checkpoint ******************************")
        LOGGER.info("Encoder, Tokenizer & Classifiers from: %s", fine_tuned_model_path)
        LOGGER.info("max_seq_length: %d, eval_batch_size: %d", CONFIG.max_seq_length, CONFIG.eval_batch_size)
        LOGGER.info("Device: %s", DEVICE)

        model_config = BertConfig.from_pretrained(fine_tuned_model_path)
        tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
        TOKEN_CONST.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "seq_start_token": tokenizer.cls_token,
                "seq_end_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        )
        model = REModel.from_pretrained(fine_tuned_model_path, config=model_config)
        model.to(DEVICE)

        if CONFIG.task == "radgraph":
            eval_files = ["train", "dev", "test", "test1", "test_chexpert", "test1_chexpert", "test_mimic", "test1_mimic"]
        elif CONFIG.task == "cxrgraph":
            eval_files = ["train", "dev", "test", "test_chexpert", "test_mimic"]
        else:
            eval_files = ["train", "dev", "test"]

        for data_name in eval_files:
            eval_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], f"{data_name}.json")
            eval_dataset = TaskDataset(tokenizer=tokenizer, file_path=eval_data_path, do_eval=True)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)
            evaluate(model, eval_dataloader)

    end = time.time()
    LOGGER.info("Time: %d minutes", (end - start) / 60)
