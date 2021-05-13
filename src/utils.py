import random
import string
import torch


def chunk_it(seq, num=1):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def batch_it(seq, num=1):
    assert num > 0
    batch = []
    for e in seq:
        if len(batch) == num:
            yield batch
            batch = [e]
        else:
            batch.append(e)
    yield batch


def shuffle_it(seq, seed=0):
    idx = list(range(len(seq)))
    random.Random(seed).shuffle(idx)
    for e in idx:
        yield seq[e]


def normalize(sent):
    return (
        sent.lower()
        .replace(" ", "")
        .translate(str.maketrans("", "", string.punctuation))
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
