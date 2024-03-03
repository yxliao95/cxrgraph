import json
import os
import re
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append("/root/workspace/cxr_structured_report")

import copy

import numpy as np
import torch
from config import TASK_NER_LABEL_LIST, TASK_REL_LABEL_INFO_DICT
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer
from utils import batched_index_select, check_and_mkdir, rindex, set_seed


class RelDataset(Dataset):
    """Load the dataset and return the tensor features of each data sample."""

    # 构造函数
    def __init__(self, tokenizer, docs, max_seq_length, num_extra_sent_for_candi_ent, ner_tag_for_eval="pred_ner"):
        """Args:
        span_source: ner|pred_ner, Indicates which field is used to build candidate span pairs.
        """
        self.tokenizer = tokenizer
        self.docs = docs

        self.max_seq_length = max_seq_length
        self.num_extra_sent_for_candi_ent = num_extra_sent_for_candi_ent
        self.ner_tag_for_eval = ner_tag_for_eval

        self.ner_label_list = TASK_NER_LABEL_LIST["cxrgraph"]
        self.ner_label2id_dict = {label: i for i, label in enumerate(self.ner_label_list)}  # ner_label_map

        self.rel_label_infodict = TASK_REL_LABEL_INFO_DICT["cxrgraph"]
        self.rel_label_infolist = sorted(self.rel_label_infodict.values(), key=lambda x: x["label_id"])
        self.inverse_rel_labelid_list = [i["inverse_label_id"] for i in self.rel_label_infolist]  # [0,1,2,8-12,3-7]

        # 用于eval的信息
        self.key2sample_dict = {}  # dict[sample_key_x3, subj_indices_x2] = sample

        # Each sentence in a doc is an sample
        self.samples = []

        for doc in self.docs:
            preprocess_info = self.preprocessing(doc)
            self._convert_doc_to_samples(doc, preprocess_info=preprocess_info)

    def preprocessing(self, doc):
        # 部分句子的长度超过 max_num_subtoks，需要将其拆分为多个句子，同时ner也要进行相应的拆分。
        max_num_subtoks = self.max_seq_length - 4  # CLS SEP <subj> </subj>

        doctoks, subtoks = [], []
        subtok_idx2doctok_idx, subtok2sentidx = [], []
        doctok_idx_indoc = 0
        exceeded_sent_ids = []  # 长度超过256的句子id
        for sent_idx, sent_doctoks in enumerate(doc["sentences"]):
            sent_subtoks = []
            for doctok in sent_doctoks:
                _subwords = self.tokenizer.tokenize(doctok)
                doctoks.append(doctok)
                subtoks.extend(_subwords)
                sent_subtoks.extend(_subwords)
                subtok_idx2doctok_idx.extend([doctok_idx_indoc] * len(_subwords))
                subtok2sentidx.extend([sent_idx] * len(_subwords))
                doctok_idx_indoc += 1

            if len(sent_subtoks) > max_num_subtoks:
                exceeded_sent_ids.append(sent_idx)

        if exceeded_sent_ids == []:
            return None
        else:
            subtok2sentidx_old = [i for i in subtok2sentidx]
            for sent_idx in exceeded_sent_ids:
                # 原本的句子长度
                sent_subtok_start_idx = subtok2sentidx.index(sent_idx)
                sent_subtok_end_idx = rindex(subtok2sentidx, sent_idx)
                sentence_length = sent_subtok_end_idx - sent_subtok_start_idx + 1  # 句子的长度
                assert sentence_length > max_num_subtoks

                sent_end_idx = sent_subtok_start_idx + 85
                while sent_end_idx <= sent_subtok_end_idx:  # 加了85个tok后，还是没有超过原本的末尾，说明可以继续拆分
                    doctok_idx = subtok_idx2doctok_idx[sent_end_idx]  # 新句子的最后一个token，对应的doctok idx（可能是中间token）
                    next_sent_start_subtok_idx = subtok_idx2doctok_idx.index(doctok_idx)  # 将这个doctok，对应的start subtok，视为下一个句子的开头
                    subtok2sentidx = [sent_idx + 1 if subtok_idx >= next_sent_start_subtok_idx else sent_idx for subtok_idx, sent_idx in enumerate(subtok2sentidx)]  # 这个subtok之后的所有sent idx + 1
                    sent_end_idx = next_sent_start_subtok_idx + 85

            pred_ners = [[] for i in range(subtok2sentidx[-1] + 1)]
            for sent_ners in doc["pred_ner"]:
                for ner in sent_ners:
                    sent_idx = subtok2sentidx[subtok_idx2doctok_idx.index(ner[0])]
                    pred_ners[sent_idx].append(ner)

            newsent2oldsent_idx_mapper = [-1 for i in range(subtok2sentidx[-1] + 1)]
            for i, j in zip(subtok2sentidx_old, subtok2sentidx):
                newsent2oldsent_idx_mapper[j] = i
            assert -1 not in newsent2oldsent_idx_mapper

            return (doctoks, subtoks, subtok_idx2doctok_idx, subtok2sentidx, pred_ners, newsent2oldsent_idx_mapper)

    def _convert_doc_to_samples(self, doc, preprocess_info=None):
        """
        1.There is only one subj per sample. That is, each ent in ner will form a sample. There will be an additional empty ent with the "X" label as subj, which is placed behind the SEP.
        2. There are multiple objs per sample. All ent in ner/pred_ner are candidate obj. But no additional empty ent with X label will be created.
        """
        
        if preprocess_info:
            doctoks, subtoks, subtok_idx2doctok_idx, subtok2sentidx, ners, newsent2oldsent_idx_mapper = preprocess_info
        else:
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
                    
            ners = doc[self.ner_tag_for_eval]
            newsent2oldsent_idx_mapper = None

        for sent_idx, sent_ners in enumerate(ners):
            ### A. 构造input ids

            # A1. 判断需要引入的上下文长度
            max_num_subtoks = self.max_seq_length - 4  # CLS SEP <subj> </subj>
            sent_subtok_start_idx = subtok2sentidx.index(sent_idx)
            sent_subtok_end_idx = rindex(subtok2sentidx, sent_idx)
            left_length = sent_subtok_start_idx  # 句子左边的长度, 正好是当前句子第一个tok的idx
            right_length = len(subtoks) - (sent_subtok_end_idx + 1)  # 句子右边的长度
            sentence_length = sent_subtok_end_idx - sent_subtok_start_idx + 1  # 句子的长度
            max_half_ctx_length = int((max_num_subtoks - sentence_length) / 2)  # 可用的单侧上下文长度

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
            if self.num_extra_sent_for_candi_ent > 0:
                expanded_sent_start_idx = sent_idx - self.num_extra_sent_for_candi_ent
                expanded_sent_start_idx = 0 if expanded_sent_start_idx < 0 else expanded_sent_start_idx
                expanded_sent_end_idx = sent_idx + self.num_extra_sent_for_candi_ent
                expanded_sent_end_idx = len(doc["sentences"]) - 1 if expanded_sent_end_idx >= len(doc["sentences"]) else expanded_sent_end_idx

            obj_entities = []
            for target_sent_idx in range(expanded_sent_start_idx, expanded_sent_end_idx + 1):
                obj_entities += [[ner[0], ner[1], ner[2], target_sent_idx] for ner in ners[target_sent_idx]]

            # 跳过无ner标签的句子
            # if not subj_entities:
            #     continue

            for subj_doctok_start_idx, subj_doctok_end_idx, subj_ner_label_str in subj_entities:

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

                rel_instances = []
                for obj_idx, (obj_doctok_start_idx, obj_doctok_end_idx, obj_ner_label_str, obj_sent_idx) in enumerate(obj_entities):

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
                            "pred_ner_label_str": obj_ner_label_str,  # train时是ner,  eval时是pred_ner
                            "ctx_indices": (ctx_left_idx, ctx_right_idx),
                        }
                    )

                # 当rel pair的数量大于CONFIG.max_pair_length时, 就增加一个样本（已取消）
                old_sent_idx = newsent2oldsent_idx_mapper[sent_idx] if newsent2oldsent_idx_mapper else sent_idx # 当句子被拆分时，使用原始的句子idx
                instance_idx = 0
                sample_key = (doc["doc_key"], old_sent_idx, instance_idx, subj_doctok_start_idx, subj_doctok_end_idx)
                input_token_len_without_objmarkers = len(final_input_tokens)
                final_input_tokens += [TOKEN_CONST["obj_start_marker"]] * len(rel_instances) + [TOKEN_CONST["obj_end_marker"]] * len(rel_instances)
                assert len(final_input_tokens) < 512

                sample = {
                    "doc_key": doc["doc_key"],
                    "sent_id": old_sent_idx,
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
                            "pred_ner_label_str": subj_ner_label_str,
                        },
                        "obj_list": rel_instances,
                    },
                    "sample_key": sample_key,
                }
                self.samples.append(sample)
                self.key2sample_dict[sample_key] = sample  # 由三元组组成的sample_key, 唯一定位样本

    # 返回数据集大小
    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def rel_collate_fn(batch):
    max_input_len = max([len(sample["input_ids"]) for sample in batch])
    max_num_objs = max([len(sample["info_dict"]["obj_list"]) for sample in batch])

    all_input_ids = []
    all_input_masks = []
    all_position_ids = []
    all_subj_indices = []
    all_rel_info = []
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
                ]
            )

        # rel_info添加padding: 1) label=-100时, 交叉熵会忽略该项；2) marker_indices理论上可以是任何值, 因为不会贡献loss; 3) ctx_indices设为 LOSS_IGNORE_INDEX,
        num_pad_objs = max_num_objs - len(sample["info_dict"]["obj_list"])
        rel_info += [[0, 0]] * num_pad_objs

        ### subj_indices
        all_subj_indices.append(sample["info_dict"]["subj"]["marker_indices"])

        all_input_ids.append(input_ids)
        all_input_masks.append(attention_mask)
        all_position_ids.append(position_ids)
        all_rel_info.append(rel_info)

    return {
        "input_ids": torch.tensor(all_input_ids).to(device),
        "input_masks": torch.tensor(np.array(all_input_masks), dtype=torch.int64).to(device),
        "position_ids": torch.tensor(all_position_ids).to(device),
        "subj_indices": torch.tensor(all_subj_indices, dtype=torch.int64).to(device),
        "rel_info": torch.tensor(all_rel_info, dtype=torch.int64).to(device),
        "sample_keys": [i["sample_key"] for i in batch],
    }


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

        return subj_embs, obj_embs

    def forward(self, input_dict):
        subj_embs, obj_embs = self._get_span_embeddings(input_dict)

        obj_ner_logits = self.ner_classifier(obj_embs)  # 对subj marker进行ent分类; (bsz, ent_len, hidd_dim*2)

        subj_rel_logits = self.re_classifier_subj(subj_embs)  # bsz, num_label
        obj_rel_logits = self.re_classifier_obj(obj_embs)  # bsz, ent_len, num_label
        rel_logits = subj_rel_logits.unsqueeze(1) + obj_rel_logits  # bsz, ent_len, num_label

        return {
            "rel_logits": rel_logits,
        }


def inference_rel(model, data_loader, output_file_path):

    eval_dataset = data_loader.dataset
    docs = data_loader.dataset.docs

    dockey2doc_mapper = dict()
    dockey_subj2sentidx_mapper = dict()
    pred_result_dict = defaultdict(dict)  # (doc_key): ((subj start,end),(obj start,end)): (rel pred logits)

    for doc in docs:
        doc["pred_rel"] = [[] for sent in doc["sentences"]]
        dockey2doc_mapper[doc["doc_key"]] = doc

    model.eval()
    with torch.no_grad():
        for input_tensors_dict in data_loader:
            # Model inference
            out = model(input_dict=input_tensors_dict)

            rel_logits = out["rel_logits"]
            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)  # torch.nn.functional.softmax(logits, dim=-1)
            rel_logits = rel_logits.cpu().numpy()

            # 把所有结果储存到 pred_result_dict 中
            # batch_inputs_dict中, 每个subj是一个样本；但在 pred_result_dict中, 需要合并为: 一个句子为一个item
            for i, sample_key in enumerate(input_tensors_dict["sample_keys"]):
                doc_key, sent_idx, _, _, _ = sample_key  # 后续按照文档进行预测
                src_sample = eval_dataset.key2sample_dict[sample_key]
                sample_info_dict = src_sample["info_dict"]
                for j, obj_info_dict in enumerate(sample_info_dict["obj_list"]):
                    subj_indices = sample_info_dict["subj"]["doctok_indices"]
                    obj_indices = obj_info_dict["doctok_indices"]

                    pred_result_dict[doc_key][(subj_indices, obj_indices)] = {
                        "rel_logits": rel_logits[i, j].tolist(),
                    }
                    dockey_subj2sentidx_mapper[(doc_key, subj_indices)] = sent_idx

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
            pred_rel_label = eval_dataset.rel_label_infolist[pred_rel_label_id]

            # 根据logits, 转化标签。如果预测的结果是反向标签, 那么就对调subj obj的位置
            if pred_rel_label_id > 0:
                if pred_rel_label["type"] == "inverse":
                    pred_rel_label_id = pred_rel_label["inverse_label_id"]
                    pred_rel_label = eval_dataset.rel_label_infolist[pred_rel_label_id]
                    subj_indices, obj_indices = obj_indices, subj_indices

                pred_score = rel_logits[pred_rel_label_id]
                doc_candidate_results.append(
                    {
                        "pred_score": pred_score,
                        "subj_indices": subj_indices,
                        "obj_indices": obj_indices,
                        "pred_rel_label": pred_rel_label,
                        "pred_rel_label_id": pred_rel_label_id,
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
                if candidate["pred_rel_label_id"] == valid_result["pred_rel_label_id"] and is_indices_overlap(candidate["subj_indices"], valid_result["subj_indices"]) and is_indices_overlap(candidate["obj_indices"], valid_result["obj_indices"]):
                    yield True
                else:
                    yield False

        valid_results = []
        for candi_result in sorted(doc_candidate_results, key=lambda x: -x["pred_score"]):
            do_drop = any(should_drop_candidate(candi_result, valid_results))
            if not do_drop:
                valid_results.append(candi_result)

        # final results
        for rel_pair in sorted(valid_results, key=lambda x: x["subj_indices"][0]):
            doc = dockey2doc_mapper[doc_key]
            sent_idx = dockey_subj2sentidx_mapper[(doc_key, rel_pair["obj_indices"])]
            doc["pred_rel"][sent_idx].append([rel_pair["subj_indices"][0], rel_pair["subj_indices"][1], rel_pair["obj_indices"][0], rel_pair["obj_indices"][1], rel_pair["pred_rel_label"]["label_str"]])

    # write
    with open(output_file_path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc))
            f.write("\n")


def is_indices_overlap(ent_a, ent_b):
    if ent_b[0] <= ent_a[0] and ent_a[0] <= ent_b[1]:
        return True
    if ent_a[0] <= ent_b[0] and ent_b[0] <= ent_a[1]:
        return True
    return False


if __name__ == "__main__":
    input_file_path = "/root/autodl-tmp/data/mimic/pred_ent.json"
    rel_model_path = "/root/autodl-tmp/offline_models/cxrgraph_pipe2_re_tokaux_sent0_seed_23"
    output_file_path = "/root/autodl-tmp/data/mimic/inference.json"

    check_and_mkdir(os.path.dirname(output_file_path))
    
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(input_file_path, "r", encoding="UTF-8") as f:
        docs = [json.loads(line) for line in f]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ner_label_list = TASK_NER_LABEL_LIST["cxrgraph"]
    rel_label_infodict = TASK_REL_LABEL_INFO_DICT["cxrgraph"]
    num_ner_labels = len(ner_label_list)
    num_rel_labels = len(rel_label_infodict)

    ner_tag_for_eval = "pred_ner"

    num_extra_sent_for_candi_ent = 0
    max_seq_length = 256
    dropout_prob = 0.1

    batch_size = 16

    rel_model_config = BertConfig.from_pretrained(rel_model_path)
    tokenizer = BertTokenizer.from_pretrained(rel_model_path)
    TOKEN_CONST = {
        "pad_token_id": tokenizer.pad_token_id,
        "seq_start_token": tokenizer.cls_token,
        "seq_end_token": tokenizer.sep_token,
        "mask_token": tokenizer.mask_token,
        "subj_start_marker": "[unused0]",
        "subj_end_marker": "[unused1]",
        "obj_start_marker": "[unused2]",
        "obj_end_marker": "[unused3]",
    }
    model = REModel.from_pretrained(rel_model_path, config=rel_model_config)
    model.to(device)

    doc_idx = 0
    chunk_size = 500
    progress_bar = tqdm(total=len(docs))
    
    while doc_idx < len(docs):
        sub_docs = docs[doc_idx : doc_idx + chunk_size]
        
        rel_dataset = RelDataset(tokenizer=tokenizer, docs=sub_docs, max_seq_length=max_seq_length, num_extra_sent_for_candi_ent=num_extra_sent_for_candi_ent, ner_tag_for_eval=ner_tag_for_eval)
        rel_sampler = SequentialSampler(rel_dataset)
        rel_dataloader = DataLoader(rel_dataset, sampler=rel_sampler, batch_size=batch_size, collate_fn=rel_collate_fn)

        inference_rel(model, rel_dataloader, output_file_path)

        doc_idx += chunk_size
        progress_bar.update(chunk_size)

    progress_bar.close()
