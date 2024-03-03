import os
import random
import operator

import numpy as np
import torch


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
) -> torch.Tensor:
    """Copy from Allennlp: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L1117-L1168

    Args:
        target: (bsz, seq_len, emb_size),
        indices: (bsz, d1, ..., dn), where each value in `dn` is in range [0, seq_len)
    Return:
        tensor (bsz, d1, ..., dn, emb_size)
    """
    sequence_length = target.size(1)
    batch_size = indices.size(0)
    emb_size = target.size(-1)

    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise IndexError(f"All elements in indices should be in range [0, {sequence_length - 1}]")

    # Returns a range vector with the desired size, starting at 0.
    # The CUDA implementation is meant to avoid copy data from CPU to GPU.
    range_vectors = torch.cuda.LongTensor(batch_size).fill_(1).cumsum(0) - 1 if indices.is_cuda else torch.arange(0, batch_size, dtype=torch.long)  # (bsz) = [0,1,2,...]

    offsets = range_vectors * sequence_length  # (bsz) = [0,288,576,...] when seq_len=288

    # offset_indices: (bsz, d1, ..., dn)
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)
    offset_indices = indices + offsets

    flattened_indices = offset_indices.view(-1)  # (bsz * d1 * ... * dn)
    flattened_target = target.view(-1, emb_size)  # (bsz * seq_len, emb_size)
    flattened_selected = torch.index_select(input=flattened_target, dim=0, index=flattened_indices)  # (bsz * d1 * ... * dn, emb_size)

    selected_shape = list(indices.size()) + [emb_size]
    selected_targets = flattened_selected.view(*selected_shape)  # (bsz, d1, ..., dn, emb_size)
    return selected_targets


def spans_pooling(target, spans_start, spans_end, excluded_indices=None, method=str):
    """
    1. spans_start 和 spans_end 是2D tensors。我们首先将其合成为3Dtensor。最后一维从原本表示单个位置的索引，变成表示从起始索引到结束索引的向量
    2. 我们会找到每个index对应的token_embeddings，此时3D tensor会变成4D tensor。然后，我们会对每个span的所有tok_embs进行池化。从而将4D tensor降到3D tensor。

    注意，spans_start, spans_end的值可能为-1或0，
    start = end = 0 表示padded span；
    start = end < 0 表示该span不存在，通常用于ctx_span。都需要设为全0向量
    但需要注意的是，对于Flan-T5这种模型，没有CLS token，此时需要进行相对应的调整，

    Args:
        target: (bsz, seq_len, emb_size),
        indices: (bsz, num_element), where each value of num_element is in range [0,sequence_len-1]
        excluded_indices: (bsz, n)
    Return:
        tensor (bsz, num_element, emb_size)
    """

    device = target.device
    
    # zero_embedding_idx = target.size(1)
    # zero_embedding = torch.zeros_like(target[:, 0, :]).unsqueeze(1) # (bsz, 1, emb_size)
    # token_embeddings = torch.cat((target, zero_embedding), dim=1) # (bsz, seq_len + 1, emb_size)

    # 相当于allennlp的batched_span_select
    span_lengths = spans_end - spans_start + 1  # (batch_size, num_spans)
    max_length = torch.max(span_lengths).item()

    # 1. 把2D的start end tensor，合并为3D
    # 首先按照最大的span长度，初始化一个3D tensor， 其最后一维全是 vectors: range(0,max_length)
    span_indices = torch.arange(max_length, device=device)
    extanded_shape = list(spans_start.size()) + [max_length]
    span_indices = span_indices.expand(extanded_shape)  # (batch_size, num_spans, max_length)

    # 把span_indices的每一个元素，根据spans_start进行偏移。在我们的数据中，ctx_start/end=-1时，表示上下文不存在，也即subj obj存在重叠
    span_indices = span_indices + torch.unsqueeze(spans_start, -1)

    # 这一步，我们需要把indices中有效的部分设为True，并将无效的部分替换为0来避免out of index error，后续会替换为特殊值并排除，因此不影响
    # 只有在 (0, end_index]的范围才算有效index。对于ctx_span不存在的情况，因为end_index=-1，所以全部视为无效index。
    # 注意，对于Flan-T5这种没有CLS tok的模型，有效范围可能会变为 [0,end_index]
    span_indices_bool = (span_indices <= torch.unsqueeze(spans_end, -1)) & (span_indices > 0)
    # 此时的上下文是从subj marker start到 obj subtok end。我们需要把subj marker排除
    if excluded_indices is not None:
        for batch_idx in range(span_indices.size(0)):
            batch_span_indices = span_indices[batch_idx, :, :]
            batch_excluded_indices = excluded_indices[batch_idx,:]
            batch_excluded_indices_bool = torch.isin(batch_span_indices, batch_excluded_indices)
            span_indices_bool[batch_idx,:,:] = span_indices_bool[batch_idx,:,:] & ~batch_excluded_indices_bool
    span_indices[~span_indices_bool] = 0

    # 2. 然后根据indices的值，找到对应的所有embs。然后进行池化，从而实现降维
    if method == "mean":
        span_tok_embs = batched_index_select(target, span_indices)  # (batch_size, num_spans, max_span_length, 768)
        span_tok_embs.masked_fill_(~span_indices_bool.unsqueeze(-1), torch.nan)
        aggregrated_span_embeddings = torch.nanmean(span_tok_embs, -2)
    elif method == "max":
        span_tok_embs = batched_index_select(target, span_indices)
        span_tok_embs.masked_fill_(~span_indices_bool.unsqueeze(-1), float("-inf"))
        aggregrated_span_embeddings, _ = torch.max(span_tok_embs, -2)
    elif method == "min":
        span_tok_embs = batched_index_select(target, span_indices)
        span_tok_embs.masked_fill_(~span_indices_bool.unsqueeze(-1), float("inf"))
        aggregrated_span_embeddings, _ = torch.min(span_tok_embs, -2)

    # start end = 0 表示padded span；start end < 0 表示该span不存在，通常用于ctx_span。都需要设为全0向量
    # 对于ctx_span，((spans_start == -1) & (spans_end == -1)) 等价于 (~span_indices_bool.any(-1))
    aggregrated_span_embeddings[(spans_start <= 0) & (spans_end <= 0)] = 0
    # zero_embedding = torch.zeros_like(aggregrated_span_embeddings[0, 0, :])
    # aggregrated_span_embeddings[(spans_start == -1) & (spans_end == -1)] = zero_embedding.expand_as(aggregrated_span_embeddings[(spans_start == -1) & (spans_end == -1)])
    # aggregrated_span_embeddings[(spans_start == 0) & (spans_end == 0)] = zero_embedding.expand_as(aggregrated_span_embeddings[(spans_start == 0) & (spans_end == 0)])

    return aggregrated_span_embeddings


def set_seed(seed):
    random.seed(seed)  # Python的随机性
    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    
def check_and_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def rindex(lst, value):
    return len(lst) - operator.indexOf(reversed(lst), value) - 1
