import json
import os
import re
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append("/root/workspace/cxr_structured_report")

import copy

import torch
from config import TASK_ATTR_LABEL_LIST, TASK_NER_LABEL_LIST
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import BertModel, BertTokenizer
from utils import batched_index_select, check_and_mkdir, rindex, set_seed


class EntDataset(Dataset):
    """Load the dataset and return the tensor features of each data sample."""

    # 构造函数
    def __init__(self, tokenizer, max_seq_length, max_span_length, docs):
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.tokenizer = tokenizer
        self.ner_label_list = TASK_NER_LABEL_LIST["cxrgraph"]
        self.ner_label2id_dict = {label: i for i, label in enumerate(self.ner_label_list)}

        self.attr_label_list = TASK_ATTR_LABEL_LIST["cxrgraph"]
        self.attr_label2id_dict = {label: i for i, label in enumerate(self.attr_label_list)}

        self.docs = docs

        # Each sentence in a doc is an sample
        self.samples = []
        for doc in self.docs:
            self._convert_doc_to_samples(doc)

    def _convert_doc_to_samples(self, doc):
        # 用于构造样本的一些元数据
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

        for sent_idx in range(len(doc["sentences"])):
            ### A. input_ids
            # A1. Determine the length of context that needs to be introduced: 2 + sent + left_ctx + right_ctx
            max_num_subtoks = self.max_seq_length - 2  # CLS SEP
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

            # A2. Under the constrain of max_seq_length, determine the position where seq needs to be truncated.
            seq_start_idx = sent_subtok_start_idx - left_context_length
            seq_end_idx = sent_subtok_end_idx + right_context_length
            input_tokens = subtoks[seq_start_idx : seq_end_idx + 1]
            input_tokens = [TOKEN_CONST["seq_start_token"]] + input_tokens + [TOKEN_CONST["seq_end_token"]]
            assert len(input_tokens) == self.max_seq_length or len(input_tokens) == (len(subtoks) + 2)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            ### B. span_info
            spans_info = []

            sent_doctok_start_idx = subtok_idx2doctok_idx[sent_subtok_start_idx]
            sent_doctok_end_idx = subtok_idx2doctok_idx[sent_subtok_end_idx]
            for span_start_doctok_idx_indoc in range(sent_doctok_start_idx, sent_doctok_end_idx + 1):
                span_start_subtok_idx_indoc = subtok_idx2doctok_idx.index(span_start_doctok_idx_indoc)
                span_start_subtok_idx_inseq = span_start_subtok_idx_indoc - seq_start_idx + 1  # +1 for CLS

                for i in range(self.max_span_length):
                    span_width = i + 1

                    span_end_doctok_idx_indoc = span_start_doctok_idx_indoc + i
                    if span_end_doctok_idx_indoc > sent_doctok_end_idx:
                        continue

                    span_end_subtok_idx_indoc = rindex(subtok_idx2doctok_idx, span_end_doctok_idx_indoc)
                    span_end_subtok_idx_inseq = span_end_subtok_idx_indoc - seq_start_idx + 1

                    spans_info.append(
                        {
                            "doctok_indices": (span_start_doctok_idx_indoc, span_end_doctok_idx_indoc),
                            "subtok_indices": (span_start_subtok_idx_inseq, span_end_subtok_idx_inseq),
                            "span_width": span_width,
                            "doctok_strs": doctoks[span_start_doctok_idx_indoc : span_end_doctok_idx_indoc + 1],
                            "subtok_strs": subtoks[span_start_subtok_idx_indoc : span_end_subtok_idx_indoc + 1],
                        }
                    )

            self.samples.append(
                {
                    "doc_key": doc["doc_key"],
                    "sent_idx": sent_idx,
                    "input_tokens": input_tokens,
                    "input_ids": input_ids,
                    "spans_info": spans_info,
                }
            )

    # 返回数据集大小
    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch_data):
    """Padding the batch data, and convert list to tensor"""

    # Pad the data for the current batch
    max_seq_length = max([len(i["input_ids"]) for i in batch_data])
    max_num_spans = max([len(i["spans_info"]) for i in batch_data])

    all_input_ids = []
    all_input_masks = []
    all_span_info = []
    all_span_masks = []
    for sample in batch_data:
        # padding the input sequence
        curr_seq_length = len(sample["input_ids"])
        num_pad_tokens = max_seq_length - curr_seq_length
        input_ids = copy.deepcopy(sample["input_ids"])
        input_ids += [TOKEN_CONST["pad_token_id"]] * num_pad_tokens
        all_input_ids.append(input_ids)
        all_input_masks.append([1] * curr_seq_length + [0] * num_pad_tokens)

        # padding the span
        curr_num_span = len(sample["spans_info"])
        num_pad_spans = max_num_spans - curr_num_span
        spans_info = []
        for span in sample["spans_info"]:
            spans_info.append(
                [
                    span["subtok_indices"][0],
                    span["subtok_indices"][1],
                    span["span_width"],
                ]
            )
        spans_info += [[0, 0, 0]] * num_pad_spans
        all_span_info.append(spans_info)
        all_span_masks.append([1] * curr_num_span + [0] * num_pad_spans)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long).to(device),
        "input_masks": torch.tensor(all_input_masks, dtype=torch.long).to(device),
        "spans_info": torch.tensor(all_span_info, dtype=torch.long).to(device),
        "spans_masks": torch.tensor(all_span_masks, dtype=torch.long).to(device),
        "spans_docinfo": [(i["doc_key"], i["sent_idx"], i["spans_info"]) for i in batch_data],
    }


class NERModel(nn.Module):
    def __init__(self, encoder, num_ner_labels, num_attr_labels, width_embedding_dim, max_span_length):
        super(NERModel, self).__init__()
        # if changing this name, make sure the params_group and state_dict are modified accordingly in the train() and save_model() functions
        self.encoder = encoder
        # share the dropout layer
        self.hidden_dropout = nn.Dropout(dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)

        self.ner_classifier = nn.Linear(self.encoder.config.hidden_size * 2 + 150, num_ner_labels)
        self.tok_classifier = nn.Linear(self.encoder.config.hidden_size, num_ner_labels)

        self.attr_classifier = nn.Linear(self.encoder.config.hidden_size * 2 + 150, num_attr_labels)

    def _get_span_and_tok_embeddings(self, input_dict):
        output = self.encoder(input_ids=input_dict["input_ids"], attention_mask=input_dict["input_masks"])
        last_hidden_state = output.last_hidden_state
        last_hidden_state = self.hidden_dropout(last_hidden_state)

        spans = input_dict["spans_info"]
        spans_tok_start = spans[:, :, 0]
        spans_tok_start_embedding = batched_index_select(last_hidden_state, spans_tok_start)

        spans_tok_end = spans[:, :, 1]
        spans_tok_end_embedding = batched_index_select(last_hidden_state, spans_tok_end)

        spans_width = spans[:, :, 2]
        spans_width_embedding = self.width_embedding(spans_width)

        span_ner_embs = torch.cat(
            (
                spans_tok_start_embedding,
                spans_tok_end_embedding,
                spans_width_embedding,
            ),
            dim=-1,
        )

        return span_ner_embs

    def forward(self, input_dict):
        span_ner_embs = self._get_span_and_tok_embeddings(input_dict)

        spans_mask = input_dict["spans_masks"]
        ner_logits = self.ner_classifier(span_ner_embs)
        attr_logits = self.attr_classifier(span_ner_embs)

        return {
            "ner_logits": ner_logits,
            "ner_masks": spans_mask,
            "attr_logits": attr_logits,
        }


def load_encoder(encoder_name_or_path):
    return BertModel.from_pretrained(encoder_name_or_path)


def load_model(model_path, width_embedding_dim, max_span_length, num_ner_labels, num_attr_labels):
    # load the fine-tuned encoder
    encoder = load_encoder(encoder_name_or_path=model_path)
    # Create a new model with the fine-tuned encoder;
    model = NERModel(encoder, width_embedding_dim=width_embedding_dim, max_span_length=max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
    # update the model state dict with the saved ner model state dict
    sd = model.state_dict()
    classifier_sd_path = os.path.join(model_path, "model_classifiers.pth")
    if not os.path.exists(classifier_sd_path):
        raise FileNotFoundError(f"The checkpoint `model_classifiers.pth` is not found in {classifier_sd_path}")
    sd.update(torch.load(classifier_sd_path))
    model.load_state_dict(sd)
    return model


def inference_ent(model, data_loader, output_file_path):
    ner_label_list = TASK_NER_LABEL_LIST["cxrgraph"]
    ner_id2label = {i: label_str for i, label_str in enumerate(ner_label_list)}

    attr_label_list = TASK_ATTR_LABEL_LIST["cxrgraph"]
    attr_id2label = {i: label_str for i, label_str in enumerate(attr_label_list)}

    model.eval()
    with torch.no_grad():
        docs = data_loader.dataset.docs

        dockey2doc_mapper = dict()
        for doc in docs:
            doc["pred_ner"] = [[] for sent in doc["sentences"]]
            doc["pred_attr"] = [[] for sent in doc["sentences"]]
            dockey2doc_mapper[doc["doc_key"]] = doc

        for input_tensors_dict in data_loader:
            # Model inference
            out = model(input_dict=input_tensors_dict)

            # ner
            pred_labels = out["ner_logits"].max(dim=-1).indices  # (bsz, num_spans)
            pred_ner_idx = torch.argwhere(pred_labels > 0)  # (n, 2)
            # attribute
            threshold = 0.5
            pred_attr_logits = out["attr_logits"].cpu().numpy()

            for batch_idx, span_idx in pred_ner_idx.tolist():
                # ner
                is_pad_span = True if out["ner_masks"][batch_idx][span_idx].item() == 0 else False
                if is_pad_span:
                    continue
                doc_key, sent_idx, spans_info = input_tensors_dict["spans_docinfo"][batch_idx]
                doctok_start_idx, doctok_end_idx = spans_info[span_idx]["doctok_indices"]

                label_id = pred_labels[batch_idx][span_idx].item()
                label_str = ner_id2label[label_id]

                doc = dockey2doc_mapper[doc_key]
                entity = [int(doctok_start_idx), int(doctok_end_idx), label_str]
                if entity:
                    doc["pred_ner"][sent_idx].append(entity)

                # attribute
                pred_span_attr_labels = pred_attr_logits[batch_idx][span_idx]
                normality, action, evolution = "NA", "NA", "NA"
                if pred_span_attr_labels[0] > threshold:
                    continue

                normality_logits = [pred_span_attr_labels[1], pred_span_attr_labels[2]]
                if any([i > threshold for i in normality_logits]):
                    label_id = normality_logits.index(max(normality_logits)) + 1
                    normality = attr_id2label[label_id]

                action_logits = [pred_span_attr_labels[3], pred_span_attr_labels[4]]
                if any([i > threshold for i in action_logits]):
                    label_id = action_logits.index(max(action_logits)) + 3
                    action = attr_id2label[label_id]

                evolution_logits = [pred_span_attr_labels[5], pred_span_attr_labels[6], pred_span_attr_labels[7]]
                if any([i > threshold for i in evolution_logits]):
                    label_id = evolution_logits.index(max(evolution_logits)) + 5
                    evolution = attr_id2label[label_id]

                entity_attribute = [int(doctok_start_idx), int(doctok_end_idx), normality, action, evolution]
                if entity:
                    doc["pred_attr"][sent_idx].append(entity_attribute)

    # write
    with open(output_file_path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc))
            f.write("\n")


if __name__ == "__main__":
    input_file_path = "/root/autodl-tmp/data/mimic/raw.json"
    ner_model_path = "/root/autodl-tmp/offline_models/cxrgraph_pipe1_ner_tokaux_attrcls_sent_seed_35"
    output_file_path = "/root/autodl-tmp/data/mimic/pred_ent.json"

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(input_file_path, "r", encoding="UTF-8") as f:
        docs = [json.loads(line) for line in f]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_ner_labels = len(TASK_NER_LABEL_LIST["cxrgraph"])
    num_attr_labels = len(TASK_ATTR_LABEL_LIST["cxrgraph"])
    width_embedding_dim = 150
    max_seq_length = 512
    max_span_length = 8
    dropout_prob = 0.1

    batch_size = 32

    tokenizer = BertTokenizer.from_pretrained(ner_model_path)
    TOKEN_CONST = {
        "pad_token_id": tokenizer.pad_token_id,
        "seq_start_token": tokenizer.cls_token,
        "seq_end_token": tokenizer.sep_token,
        "mask_token": tokenizer.mask_token,
    }

    model = load_model(ner_model_path, width_embedding_dim=width_embedding_dim, max_span_length=max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
    model.to(device)

    doc_idx = 0
    chunk_size = 500
    progress_bar = tqdm(total=len(docs))

    while doc_idx < len(docs):
        sub_docs = docs[doc_idx : doc_idx + chunk_size]

        ent_dataset = EntDataset(tokenizer=tokenizer, docs=sub_docs, max_seq_length=max_seq_length, max_span_length=max_span_length)
        ent_sampler = SequentialSampler(ent_dataset)
        ent_dataloader = DataLoader(ent_dataset, sampler=ent_sampler, batch_size=batch_size, collate_fn=collate_fn)

        inference_ent(model, ent_dataloader, output_file_path)

        doc_idx += chunk_size
        progress_bar.update(chunk_size)

    progress_bar.close()
