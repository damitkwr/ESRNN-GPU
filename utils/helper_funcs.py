import torch


def colwise_batch_mask(target_shape_tuple, target_lens):
    # takes in (seq_len, B) shape and returns mask of same shape with ones up to the target lens
    mask = torch.zeros(target_shape_tuple)
    for i in range(target_shape_tuple[1]):
        mask[:target_lens[i], i] = 1
    return mask


def rowwise_batch_mask(target_shape_tuple, target_lens):
    # takes in (B, seq_len) shape and returns mask of same shape with ones up to the target lens
    mask = torch.zeros(target_shape_tuple)
    for i in range(target_shape_tuple[0]):
        mask[i, :target_lens[i]] = 1
    return mask


def unpad_sequence(padded_sequence, lens):
    seqs = []
    for i in range(padded_sequence.size(1)):
        seqs.append(padded_sequence[:lens[i], i])
    return seqs

