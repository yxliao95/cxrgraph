import argparse
import copy
import json
import logging
import os
import time

import torch
from config import TASK_ATTR_LABEL_LIST, TASK_NER_LABEL_LIST
from config import pipe1_ner_tokaux_attrcls_sent as CONFIG
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import batched_index_select, check_and_mkdir, rindex, set_seed

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.DEBUG)
LOGGER = logging.getLogger("ner_tokaux")


TOKEN_CONST = {
    "pad_token_id": None,
    "seq_start_token": None,
    "seq_end_token": None,
    "mask_token": None,
}

DEVICE = None

##############################################
# Model class
##############################################


class NERModel(nn.Module):
    def __init__(self, encoder, num_ner_labels, num_attr_labels, width_embedding_dim, max_span_length):
        super(NERModel, self).__init__()
        # if changing this name, make sure the params_group and state_dict are modified accordingly in the train() and save_model() functions
        self.encoder = encoder
        # share the dropout layer
        self.hidden_dropout = nn.Dropout(CONFIG.dropout_prob)
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

        tokens = input_dict["tokens_info"]
        token_indices = tokens[:, :, 0]
        token_embs = batched_index_select(last_hidden_state, token_indices)

        return span_ner_embs, token_embs

    def forward(self, input_dict, mode):
        span_ner_embs, token_embs = self._get_span_and_tok_embeddings(input_dict)

        ner_labels = input_dict["spans_info"][:, :, 3]
        spans_mask = input_dict["spans_masks"]

        ner_logits = self.ner_classifier(span_ner_embs)
        ner_active_logits = ner_logits.view(-1, ner_logits.shape[-1])[spans_mask.view(-1) > 0]
        ner_active_labels = ner_labels.view(-1)[spans_mask.view(-1) > 0]

        tok_labels = input_dict["tokens_info"][:, :, 1:]
        tok_masks = input_dict["tokens_masks"]

        tok_logits = self.tok_classifier(token_embs)
        tok_active_logits = tok_logits.view(-1, tok_logits.shape[-1])[tok_masks.view(-1) > 0]
        tok_active_labels = tok_labels.view(-1, tok_labels.shape[-1])[tok_masks.view(-1) > 0]
        
        attr_labels = input_dict["spans_info"][:, :, 4:]

        attr_logits = self.attr_classifier(span_ner_embs)
        attr_active_logits = attr_logits.view(-1, attr_logits.shape[-1])[spans_mask.view(-1) > 0]
        attr_active_labels = attr_labels.view(-1, attr_labels.shape[-1])[spans_mask.view(-1) > 0]

        if mode == "train":
            ce_loss_fct = nn.CrossEntropyLoss(reduction="mean")
            ce_loss = ce_loss_fct(ner_active_logits, ner_active_labels)

            bce_loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            tok_bce_loss = bce_loss_fct(tok_active_logits, tok_active_labels.float())
            
            attr_bce_loss = bce_loss_fct(attr_active_logits, attr_active_labels.float())

            return {
                "ner_loss": ce_loss.sum(),
                "tok_loss": tok_bce_loss.sum(),
                "attr_loss": attr_bce_loss.sum(),
            }

        elif mode == "eval":
            return {
                "ner_logits": ner_active_logits,
                "ner_labels": ner_active_labels,
                "tok_logits": tok_active_logits,
                "tok_labels": tok_active_labels,
                "attr_logits": attr_active_logits,
                "attr_labels": attr_active_labels,
                "_ner_logits": ner_logits,
                "_ner_masks": spans_mask,
                "_attr_logits": attr_logits,
                "_attr_labels": attr_labels,
            }

        elif mode == "pred":
            return {
                "ner_logits": ner_logits,
                "ner_labels": ner_labels,
                "ner_masks": spans_mask,
                "attr_logits": attr_logits,
                "attr_labels": attr_labels,
            }


##############################################
# Load dataset class
##############################################


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


class TaskDataset(Dataset):
    """Load the dataset and return the tensor features of each data sample."""

    # 构造函数
    def __init__(self, tokenizer, file_path):
        self.max_seq_length = CONFIG.max_seq_length
        self.max_span_length = CONFIG.max_span_length
        self.tokenizer = tokenizer
        self.src_path = file_path
        self.ner_label_list = TASK_NER_LABEL_LIST[CONFIG.task]
        self.ner_label2id_dict = {label: i for i, label in enumerate(self.ner_label_list)}

        self.attr_label_list = TASK_ATTR_LABEL_LIST[CONFIG.task]
        self.attr_label2id_dict = {label: i for i, label in enumerate(self.attr_label_list)}

        with open(file_path, "r", encoding="UTF-8") as f:
            self.docs = [json.loads(line) for line in f]

        # Each sentence in a doc is an sample
        self.samples = []
        for doc in tqdm(self.docs):
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
            span_indices2label_str = {(ner[0], ner[1]): ner[2] for ner in doc["ner"][sent_idx]}

            span_indices2attr_label_str = {(attr[0], attr[1]): (attr[2], attr[3], attr[4]) for attr in doc["entity_attributes"][sent_idx]}

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

                    label_str = span_indices2label_str.get((span_start_doctok_idx_indoc, span_end_doctok_idx_indoc), "X")
                    label_id = self.ner_label2id_dict[label_str]

                    attr_label_str_x3 = span_indices2attr_label_str.get((span_start_doctok_idx_indoc, span_end_doctok_idx_indoc), ("X", "X", "X"))
                    attr_label_onehot = [1] + [0] * (len(self.attr_label2id_dict) - 1)
                    for attr_label_str in attr_label_str_x3:
                        attr_label_str = "X" if attr_label_str == "NA" else attr_label_str
                        attr_label_id = self.attr_label2id_dict[attr_label_str]
                        if attr_label_id > 0:
                            attr_label_onehot[0] = 0
                            attr_label_onehot[attr_label_id] = 1

                    spans_info.append(
                        {
                            "ner_label": (label_str, label_id),
                            "attr_label": (attr_label_str_x3, attr_label_onehot),
                            "doctok_indices": (span_start_doctok_idx_indoc, span_end_doctok_idx_indoc),
                            "subtok_indices": (span_start_subtok_idx_inseq, span_end_subtok_idx_inseq),
                            "span_width": span_width,
                            "doctok_strs": doctoks[span_start_doctok_idx_indoc : span_end_doctok_idx_indoc + 1],
                            "subtok_strs": subtoks[span_start_subtok_idx_indoc : span_end_subtok_idx_indoc + 1],
                        }
                    )

            ### C. token_info
            sent_subtok_start_idx_inseq = sent_subtok_start_idx - seq_start_idx + 1
            sent_subtok_end_idx_inseq = sent_subtok_end_idx - seq_start_idx + 1
            tokens_idx2onehot_dict = {subtok_idx_inseq: [1] + [0] * (len(self.ner_label_list) - 1) for subtok_idx_inseq in range(sent_subtok_start_idx_inseq, sent_subtok_end_idx_inseq + 1)}

            for ner in doc["ner"][sent_idx]:
                # one entity has 1+ doctoks, one doctok has 1+ subtoks
                for doctok_idx_indoc in range(ner[0], ner[1] + 1):
                    doctok_start_subtok_idx = subtok_idx2doctok_idx.index(doctok_idx_indoc)
                    doctok_end_subtok_idx = rindex(subtok_idx2doctok_idx, doctok_idx_indoc)
                    for subtok_idx_indoc in range(doctok_start_subtok_idx, doctok_end_subtok_idx + 1):
                        subtok_idx_inseq = subtok_idx_indoc - seq_start_idx + 1
                        label_id = self.ner_label2id_dict[ner[2]]
                        tokens_idx2onehot_dict[subtok_idx_inseq][label_id] = 1
                        tokens_idx2onehot_dict[subtok_idx_inseq][0] = 0

            self.samples.append(
                {
                    "doc_key": doc["doc_key"],
                    "sent_idx": sent_idx,
                    "input_tokens": input_tokens,
                    "input_ids": input_ids,
                    "spans_info": spans_info,
                    "tokens_idx2onehot_dict": tokens_idx2onehot_dict,
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
    max_num_aux_toks = max([len(i["tokens_idx2onehot_dict"]) for i in batch_data])

    all_input_ids = []
    all_input_masks = []
    all_span_info = []
    all_span_masks = []
    all_tokens_info = []
    all_token_masks = []
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
                    span["ner_label"][1],
                ]
                + span["attr_label"][1]
            )
        onehot_label_dim = len(spans_info[0]) - 4
        spans_info += [[0, 0, 0, -1] + [-1] * onehot_label_dim] * num_pad_spans
        all_span_info.append(spans_info)
        all_span_masks.append([1] * curr_num_span + [0] * num_pad_spans)

        # padding the aux tokens
        curr_num_aux_toks = len(sample["tokens_idx2onehot_dict"])
        num_pad_aux_toks = max_num_aux_toks - curr_num_aux_toks
        aux_tokens_info = []
        for tok_idx, onehot_label in sample["tokens_idx2onehot_dict"].items():
            aux_tokens_info.append([tok_idx] + copy.deepcopy(onehot_label))
        onehot_label_dim = len(aux_tokens_info[0]) - 1
        aux_tokens_info += [[0] + [-1] * onehot_label_dim] * num_pad_aux_toks
        all_tokens_info.append(aux_tokens_info)
        all_token_masks.append([1] * curr_num_aux_toks + [0] * num_pad_aux_toks)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long).to(DEVICE),
        "input_masks": torch.tensor(all_input_masks, dtype=torch.long).to(DEVICE),
        "spans_info": torch.tensor(all_span_info, dtype=torch.long).to(DEVICE),
        "spans_masks": torch.tensor(all_span_masks, dtype=torch.long).to(DEVICE),
        "tokens_info": torch.tensor(all_tokens_info, dtype=torch.long).to(DEVICE),
        "tokens_masks": torch.tensor(all_token_masks, dtype=torch.long).to(DEVICE),
        "spans_docinfo": [(i["doc_key"], i["sent_idx"], i["spans_info"]) for i in batch_data],
    }


##############################################
# Train and eval
##############################################


def train(model, tokenizer):
    # 加载数据
    LOGGER.info("****************************** Loading data ******************************")
    train_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "train.json")
    train_dataset = TaskDataset(tokenizer=tokenizer, file_path=train_data_path)
    train_sampler = StableRandomSampler(train_dataset, CONFIG.num_epoch)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=CONFIG.train_batch_size, drop_last=True)

    dev_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "dev.json")
    dev_dataset = TaskDataset(tokenizer=tokenizer, file_path=dev_data_path)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)

    test_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], "test.json")
    test_dataset = TaskDataset(tokenizer=tokenizer, file_path=test_data_path)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)

    # Setting for training
    model_params = list(model.named_parameters())
    doc_encoder_params = [(n, p) for n, p in model_params if "encoder" in n]
    classifier_params = [(n, p) for n, p in model_params if "encoder" not in n]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in doc_encoder_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in doc_encoder_params if all(nd not in n for nd in no_decay)], "weight_decay": CONFIG.weight_decay},
        {"params": [p for n, p in classifier_params], "lr": CONFIG.ner_learning_rate, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG.encoder_learning_rate, correct_bias=True, eps=1e-8)  # eps=1e-6
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
        "batch_ner_loss": 0,
        "batch_tok_loss": 0,
        "batch_attr_loss": 0,
        "checkpoint_saved_at": 0,
        "ner": log_result_dict.copy(),
        "attr": log_result_dict.copy(),
        "avg": log_result_dict.copy(),
    }

    LOGGER.info("****************************** Training ******************************")
    LOGGER.info("Total samples = %d, batch size = %d", len(train_dataset), CONFIG.train_batch_size)
    LOGGER.info("Total epochs = %d, total iterations per epoch = %d", CONFIG.num_epoch, len(train_dataloader))
    LOGGER.info("Total optimization steps = %d", total_num_steps)
    LOGGER.info("Gradient accumulation steps = %d", CONFIG.grad_accum_steps)
    if CONFIG.grad_accum_steps > 1:
        LOGGER.info("Gradient accumulation at iter=%s per epoch", log_info["grad_accum_at_iter_n_per_epoch"])

    def eval_model_and_save_checkpoint_at(curr_epoch_or_step):
        dev_out = evaluate(model, dev_dataloader)
        test_out = evaluate(model, test_dataloader)

        LOGGER.info("****************************** Checkpoint ******************************")
        LOGGER.info("(curr_epoch_or_step=%d) Current [dev] ner f1: %.3f, attr f1: %.3f", curr_epoch_or_step, dev_out["ner"] * 100, dev_out["attr"] * 100)
        LOGGER.debug("(curr_epoch_or_step=%d) Current [test] ner f1: %.3f, attr f1: %.3f", curr_epoch_or_step, test_out["ner"] * 100, test_out["attr"] * 100)

        # print result
        for eval_type in ["ner", "attr"]:
            dev_f1 = dev_out[eval_type]
            test_f1 = test_out[eval_type]

            achieved_best_dev = True if dev_f1 > log_info[eval_type]["best_dev_f1"] else False
            achieved_best_test = True if test_f1 > log_info[eval_type]["best_test_f1"] else False
            
            if achieved_best_dev:
                log_info[eval_type]["best_dev_f1"] = dev_f1
                log_info[eval_type]["best_dev_test_f1"] = test_f1
                log_info[eval_type]["best_dev_at"] = curr_epoch_or_step
                LOGGER.info("(curr_epoch_or_step=%d) !!! Achieved best %s dev f1: %.3f", curr_epoch_or_step, eval_type, dev_f1 * 100)

            if achieved_best_test:
                log_info[eval_type]["best_test_dev_f1"] = dev_f1
                log_info[eval_type]["best_test_f1"] = test_f1
                log_info[eval_type]["best_test_at"] = curr_epoch_or_step
                LOGGER.debug("(curr_epoch_or_step=%d) !!! Achieved best %s test f1: %.3f", curr_epoch_or_step, eval_type, test_f1 * 100)
                
        # save checkpoint
        for eval_type in ["avg"]:
            dev_f1 = (dev_out["ner"] + dev_out["attr"]) / 2
            test_f1 = (test_out["ner"] + test_out["attr"]) / 2

            achieved_best_dev = True if dev_f1 > log_info[eval_type]["best_dev_f1"] else False
            achieved_best_test = True if test_f1 > log_info[eval_type]["best_test_f1"] else False
            
            if achieved_best_dev:
                log_info[eval_type]["best_dev_f1"] = dev_f1
                log_info[eval_type]["best_dev_test_f1"] = test_f1
                log_info[eval_type]["best_dev_at"] = curr_epoch_or_step
                LOGGER.info("(curr_epoch_or_step=%d) !!! Achieved best %s dev f1: %.3f", curr_epoch_or_step, eval_type, dev_f1 * 100)

                log_info["checkpoint_saved_at"] = curr_epoch_or_step
                LOGGER.info("Saving model checkpoint to %s", CONFIG.output_model_dir)
                save_model(model, CONFIG.output_model_dir)

            if achieved_best_test:
                log_info[eval_type]["best_test_dev_f1"] = dev_f1
                log_info[eval_type]["best_test_f1"] = test_f1
                log_info[eval_type]["best_test_at"] = curr_epoch_or_step
                LOGGER.debug("(curr_epoch_or_step=%d) !!! Achieved best %s test f1: %.3f", curr_epoch_or_step, eval_type, test_f1 * 100)

        # print the best result
        LOGGER.info("------------------------------------")
        for eval_type in ["ner", "attr", "avg"]:
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

    model.zero_grad()
    for curr_epoch in range(CONFIG.num_epoch):
        for curr_iter, batch_inputs_dict in enumerate(train_dataloader):
            model.train()
            out = model(input_dict=batch_inputs_dict, mode="train")

            ner_loss = out["ner_loss"]
            tok_loss = out["tok_loss"]
            attr_loss = out["attr_loss"]
            loss = ner_loss + tok_loss + attr_loss

            if CONFIG.grad_accum_steps > 1:
                loss = loss / CONFIG.grad_accum_steps
                ner_loss = ner_loss / CONFIG.grad_accum_steps
                tok_loss = tok_loss / CONFIG.grad_accum_steps
                attr_loss = attr_loss / CONFIG.grad_accum_steps

            loss.backward()

            curr_batch_size = batch_inputs_dict["input_ids"].shape[0]
            log_info["batch_trained_examples"] += curr_batch_size
            log_info["batch_loss"] += loss.item() * curr_batch_size
            log_info["batch_ner_loss"] += ner_loss.item() * curr_batch_size
            log_info["batch_tok_loss"] += tok_loss.item() * curr_batch_size
            log_info["batch_attr_loss"] += attr_loss.item() * curr_batch_size

            # Update model parameters
            if (curr_iter + 1) % CONFIG.grad_accum_steps == 0:
                if CONFIG.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_info["global_step"] += 1

                # print loss
                if log_info["global_step"] == 1 or log_info["global_step"] % CONFIG.print_loss_per_n_step == 0:
                    LOGGER.debug(
                        "Epoch=%d, iter=%d, steps=%d, loss=%.9f (ner=%.6f, tok=%.6f, attr=%.6f)",
                        curr_epoch,
                        curr_iter,
                        log_info["global_step"],
                        log_info["batch_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_ner_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_tok_loss"] / log_info["batch_trained_examples"],
                        log_info["batch_attr_loss"] / log_info["batch_trained_examples"],
                    )
                    log_info["batch_trained_examples"] = 0
                    log_info["batch_loss"] = 0
                    log_info["batch_ner_loss"] = 0
                    log_info["batch_tok_loss"] = 0
                    log_info["batch_attr_loss"] = 0

                # checkpoint, eval at specific steps:
                if log_info["global_step"] % CONFIG.eval_per_steps == 0:
                    eval_model_and_save_checkpoint_at(curr_epoch_or_step=log_info["global_step"])

        # checkpoint, eval at the end of each epoch:
        if curr_epoch >= (CONFIG.num_epoch / 2 - 1) and curr_epoch % CONFIG.eval_per_epoch == 0:
            eval_model_and_save_checkpoint_at(curr_epoch_or_step=curr_epoch)

    LOGGER.info("Best checkpoint saved at: %s", log_info["checkpoint_saved_at"])


def evaluate(model, eval_dataloader):
    eval_dataset = eval_dataloader.dataset

    sum_ner = 0
    for doc in eval_dataset.docs:
        for sent_ner in doc["ner"]:
            sum_ner += len(sent_ner)

    # Meta data for evaluation
    # num_gold_label is obtained based on the data from source json files
    task_f1 = {}
    eval_results = {
        "binary_ner": {
            "num_gold_label": sum_ner,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "ner": {
            "num_gold_label": sum_ner,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "aux_tok": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "attr": {
            "num_gold_label": 0,
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
        "ent_attr": {
            "num_gold_label": len([i for doc in eval_dataset.docs for sent_ele in doc["entity_attributes"] for ele in sent_ele for i in ele[2:] if i != "NA"]),
            "num_pred_label": 0,
            "num_correct_label": 0,
        },
    }

    LOGGER.info("****************************** Evaluation ******************************")
    LOGGER.info("Source = %s", eval_dataset.src_path)
    LOGGER.info("Batch size = %d", CONFIG.eval_batch_size)
    LOGGER.info("Num samples = %d", len(eval_dataset))

    # Run model to get the inference results
    model.eval()
    with torch.no_grad():
        for input_tensors_dict in eval_dataloader:
            # Model inference
            out = model(input_dict=input_tensors_dict, mode="eval")

            # Eval: Statistics on the NER result for later evaluation
            ner_logits = out["ner_logits"]
            _, ner_predicted_label = ner_logits.max(dim=-1)
            pred_ner = ner_predicted_label.cpu().numpy()
            gold_ner_label = out["ner_labels"].cpu().numpy()
            for gold, pred in zip(gold_ner_label, pred_ner):
                if pred > 0:
                    eval_results["binary_ner"]["num_pred_label"] += 1
                    eval_results["ner"]["num_pred_label"] += 1
                    if gold > 0:
                        eval_results["binary_ner"]["num_correct_label"] += 1
                        if gold == pred:
                            eval_results["ner"]["num_correct_label"] += 1

            # Token multi-cls classification
            threshold = 0.5
            tok_probs = torch.sigmoid(out["tok_logits"])
            pred_tok = torch.where(tok_probs > threshold, 1, 0).cpu().numpy()
            gold_tok_label = out["tok_labels"].cpu().numpy()
            for gold, pred in zip(gold_tok_label, pred_tok):
                for i, (y_gt, y_pred) in enumerate(zip(gold, pred)):
                    if i == 0:
                        continue
                    if y_gt == 1:
                        eval_results["aux_tok"]["num_gold_label"] += 1
                    if y_pred == 1:
                        eval_results["aux_tok"]["num_pred_label"] += 1
                    if y_gt == 1 and y_pred == 1:
                        eval_results["aux_tok"]["num_correct_label"] += 1
            
            # [attr-] span attribute multi-cls classification, may not be correct
            threshold = 0.5
            attr_probs = torch.sigmoid(out["attr_logits"])
            pred_attr = torch.where(attr_probs > threshold, 1, 0).cpu().numpy()
            gold_attr_label = out["attr_labels"].cpu().numpy()
            for gold, pred in zip(gold_attr_label, pred_attr):
                for i, (y_gt, y_pred) in enumerate(zip(gold, pred)):
                    if i == 0:
                        continue
                    if y_gt == 1:
                        eval_results["attr"]["num_gold_label"] += 1
                    if y_pred == 1:
                        eval_results["attr"]["num_pred_label"] += 1
                    if y_gt == 1 and y_pred == 1:
                        eval_results["attr"]["num_correct_label"] += 1

            # [attr] attribute with post processing; equals to the inference results
            threshold = 0.5
            pred_labels = out["_ner_logits"].max(dim=-1).indices  # (bsz, num_spans)
            pred_ner_idx = torch.argwhere(pred_labels > 0)  # (n, 2)
            pred_attr_logits = out["_attr_logits"].cpu().numpy()
            gold_attr_labels = out["_attr_labels"].cpu().numpy()
            for batch_idx, span_idx in pred_ner_idx.tolist():
                # ner
                is_pad_span = True if out["_ner_masks"][batch_idx][span_idx].item() == 0 else False
                if is_pad_span:
                    continue

                # attribute
                pred_span_attr_labels = pred_attr_logits[batch_idx][span_idx]
                gold_span_attr_labels = gold_attr_labels[batch_idx][span_idx]
                if pred_span_attr_labels[0] > threshold:
                    continue
                
                normality_logits = [pred_span_attr_labels[1], pred_span_attr_labels[2]]
                if any([i > threshold for i in normality_logits]):
                    label_id = normality_logits.index(max(normality_logits)) + 1
                    eval_results["ent_attr"]["num_pred_label"] += 1
                    if gold_span_attr_labels[label_id] == 1:
                        eval_results["ent_attr"]["num_correct_label"] += 1
                    
                action_logits = [pred_span_attr_labels[3], pred_span_attr_labels[4]]
                if any([i > threshold for i in action_logits]):
                    label_id = action_logits.index(max(action_logits)) + 3
                    eval_results["ent_attr"]["num_pred_label"] += 1
                    if gold_span_attr_labels[label_id] == 1:
                        eval_results["ent_attr"]["num_correct_label"] += 1
                    
                evolution_logits = [pred_span_attr_labels[5], pred_span_attr_labels[6], pred_span_attr_labels[7]]
                if any([i > threshold for i in evolution_logits]):
                    label_id = evolution_logits.index(max(evolution_logits)) + 5
                    eval_results["ent_attr"]["num_pred_label"] += 1
                    if gold_span_attr_labels[label_id] == 1:
                        eval_results["ent_attr"]["num_correct_label"] += 1
                
            
    # Evaluate the results
    for eval_field, result_dict in eval_results.items():
        num_corr = result_dict["num_correct_label"]
        num_pred = result_dict["num_pred_label"]
        num_gt = result_dict["num_gold_label"]
        p = num_corr / num_pred if num_corr > 0 else 0.0
        r = num_corr / num_gt if num_corr > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if num_corr > 0 else 0.0
        LOGGER.info("[%s]: P: %.5f, R: %.5f, 【F1: %.3f】", eval_field, p, r, f1 * 100)
        task_f1[eval_field] = f1

    return task_f1


def inference2json(model, data_loader, output_path):
    LOGGER.info("****************************** Inference ******************************")
    LOGGER.info("Source = %s", data_loader.dataset.src_path)
    LOGGER.info("Batch size = %d", CONFIG.eval_batch_size)
    LOGGER.info("Num docs = %d, Num samples = %d", len(data_loader.dataset.docs), len(data_loader.dataset))
    LOGGER.info("Json output = %s", output_path)

    # Run model to get the inference results
    ner_label_list = TASK_NER_LABEL_LIST[CONFIG.task]
    ner_id2label = {i: label_str for i, label_str in enumerate(ner_label_list)}
    
    attr_label_list = TASK_ATTR_LABEL_LIST[CONFIG.task]
    attr_id2label = {i: label_str for i, label_str in enumerate(attr_label_list)}

    model.eval()
    with torch.no_grad():
        docs = data_loader.dataset.docs

        dockey2doc_mapper = dict()
        for doc in docs:
            doc["pred_ner"] = [[] for sent in doc["ner"]]
            doc["pred_attributes"] = [[] for sent in doc["ner"]]
            dockey2doc_mapper[doc["doc_key"]] = doc

        for input_tensors_dict in tqdm(data_loader):
            # Model inference
            out = model(input_dict=input_tensors_dict, mode="pred")

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
                    doc["pred_attributes"][sent_idx].append(entity_attribute)
                        

    # write
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc))
            f.write("\n")


##############################################
# functions
##############################################


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label


def load_encoder(encoder_name_or_path):
    return BertModel.from_pretrained(encoder_name_or_path)


def init_model(encoder_name_or_path, width_embedding_dim, max_span_length, num_ner_labels, num_attr_labels):
    encoder = load_encoder(encoder_name_or_path)
    model = NERModel(encoder, width_embedding_dim=width_embedding_dim, max_span_length=max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
    return model


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


def save_model(model, output_dir):
    """
    Save the model to the output directory; the encoder and the classifiers are saved separately
    """

    # saving encoder to the same model as tokenizer
    model.encoder.save_pretrained(output_dir)

    # remove the encoder's state_dict and save the ner head only.
    classifiers_state_dict = {k: v for k, v in model.state_dict().items() if "encoder" not in k}
    torch.save(classifiers_state_dict, os.path.join(output_dir, "model_classifiers.pth"))


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
    parser.add_argument("--bert_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        CONFIG.output_name = args.output_name
    if args.from_bash:
        CONFIG.output_name = args.output_name
        # CONFIG.seed = args.seed
        CONFIG.model_name_or_path[CONFIG.task] = args.bert_path

    # 1. Reproducibility
    set_seed(CONFIG.seed)

    # 2. Create output directory, and logging handler to file
    CONFIG.output_dir = os.path.join(CONFIG.output_dir, CONFIG.output_name)
    CONFIG.output_model_dir = os.path.join(CONFIG.output_model_dir, CONFIG.output_name)
    check_and_mkdir(CONFIG.output_dir)
    check_and_mkdir(CONFIG.output_model_dir)
    if CONFIG.do_train:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_dir, "train.log"), "w")
    elif CONFIG.do_pred:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_dir, "pred.log"), "w")
    elif CONFIG.do_eval:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_dir, "eval.log"), "w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.info([i for i in vars(CONFIG).items() if i[0][0] != "_"])

    start = time.time()
    num_ner_labels = len(TASK_NER_LABEL_LIST[CONFIG.task])
    num_attr_labels = len(TASK_ATTR_LABEL_LIST[CONFIG.task])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CONFIG.do_train:
        # 4. Initial model
        LOGGER.info("****************************** Initialize model ******************************")
        LOGGER.info("Encoder & Tokenizer from: %s", CONFIG.model_name_or_path[CONFIG.task])
        LOGGER.info("width_emb_dim = %s, max_span_length = %s, num_ner_labels = %s, num_attr_labels = %s", CONFIG.width_embedding_dim, CONFIG.max_span_length, num_ner_labels, num_attr_labels)
        LOGGER.info("Device: %s", DEVICE)
        tokenizer = BertTokenizer.from_pretrained(CONFIG.model_name_or_path[CONFIG.task])
        TOKEN_CONST.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "seq_start_token": tokenizer.cls_token,
                "seq_end_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        )
        model = init_model(CONFIG.model_name_or_path[CONFIG.task], width_embedding_dim=CONFIG.width_embedding_dim, max_span_length=CONFIG.max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
        model.to(DEVICE)

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
        LOGGER.info("width_emb_dim = %s, max_span_length = %s, num_ner_labels = %s, num_attr_labels = %s", CONFIG.width_embedding_dim, CONFIG.max_span_length, num_ner_labels, num_attr_labels)
        LOGGER.info("Device: %s", DEVICE)
        tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
        TOKEN_CONST.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "seq_start_token": tokenizer.cls_token,
                "seq_end_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        )
        model = load_model(fine_tuned_model_path, width_embedding_dim=CONFIG.width_embedding_dim, max_span_length=CONFIG.max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
        model.to(DEVICE)

        eval_files = ["train", "dev", "test", "test_chexpert", "test_mimic"]

        for data_name in eval_files:
            eval_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], f"{data_name}.json")
            eval_dataset = TaskDataset(tokenizer=tokenizer, file_path=eval_data_path)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)
            evaluate(model, eval_dataloader)

    if CONFIG.do_pred:
        if os.path.exists(CONFIG.fine_tuned_model_path):
            fine_tuned_model_path = CONFIG.fine_tuned_model_path
        else:
            fine_tuned_model_path = CONFIG.output_model_dir
        LOGGER.info("****************************** Load model from checkpoint ******************************")
        LOGGER.info("Encoder, Tokenizer & Classifiers from: %s", fine_tuned_model_path)
        LOGGER.info("width_emb_dim = %s, max_span_length = %s, num_ner_labels = %s, num_attr_labels = %s", CONFIG.width_embedding_dim, CONFIG.max_span_length, num_ner_labels, num_attr_labels)
        LOGGER.info("Device: %s", DEVICE)
        tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
        TOKEN_CONST.update(
            {
                "pad_token_id": tokenizer.pad_token_id,
                "seq_start_token": tokenizer.cls_token,
                "seq_end_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        )
        model = load_model(fine_tuned_model_path, width_embedding_dim=CONFIG.width_embedding_dim, max_span_length=CONFIG.max_span_length, num_ner_labels=num_ner_labels, num_attr_labels=num_attr_labels)
        model.to(DEVICE)

        eval_files = ["train", "dev", "test", "test_chexpert", "test_mimic"]

        for data_name in eval_files:
            eval_data_path = os.path.join(CONFIG.data_dir[CONFIG.task], f"{data_name}.json")
            eval_dataset = TaskDataset(tokenizer=tokenizer, file_path=eval_data_path)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=CONFIG.eval_batch_size, collate_fn=collate_fn)

            json_output_dir = os.path.join(CONFIG.output_dir, "pred_ner")
            check_and_mkdir(json_output_dir)
            json_output_path = os.path.join(json_output_dir, f"{data_name}.json")

            inference2json(model, eval_dataloader, output_path=json_output_path)

    end = time.time()
    LOGGER.info("Time: %d minutes", (end - start) / 60)
