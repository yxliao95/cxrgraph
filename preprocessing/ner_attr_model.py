class TaskDataset(Dataset):
    """Load the dataset and return the tensor features of each data sample."""

    # 构造函数
    def __init__(self, tokenizer, docs, max_seq_length, num_extra_sent_for_candi_ent):
        """Args:
        span_source: ner|pred_ner, Indicates which field is used to build candidate span pairs.
        """
        self.tokenizer = tokenizer
        self.docs = docs
        
        self.max_seq_length = max_seq_length
        self.num_extra_sent_for_candi_ent = num_extra_sent_for_candi_ent

        self.ner_label_list = TASK_NER_LABEL_LIST["cxrgraph"]
        self.ner_label2id_dict = {label: i for i, label in enumerate(self.ner_label_list)}  # ner_label_map

        self.rel_label_infodict = TASK_REL_LABEL_INFO_DICT["cxrgraph"]
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
        ners = doc["pred_ner"]
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
            gold_ner_indices2label_checkdict = {}
            for target_sent_idx in range(expanded_sent_start_idx, expanded_sent_end_idx + 1):
                obj_entities += [[ner[0], ner[1], ner[2], target_sent_idx] for ner in ners[target_sent_idx]]
                gold_ner_indices2label_checkdict.update({(ner_info[0], ner_info[1]): ner_info[2] for ner_info in doc["ner"][target_sent_idx] if ner_info})

            # 跳过无ner标签的句子
            # if not subj_entities:
            #     continue

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
                            "ctx_indices": (ctx_left_idx, ctx_right_idx),
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