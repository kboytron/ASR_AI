import numpy as np


def sequence_mask(lengths, maxlen):
    """Computes sequence mask.

    Args:
        lengths: [batch]
        maxlen: int, output sequence dimension

    Returns:
        Output binary mask of the shape [batch, maxlen].
    """

    return np.arange(maxlen)[None, :] < lengths[:, None]


def pad_sequences(l):
    """Pad sequences into tensor.

    Args:
        l: the list of tensors, each of the shape [time, dim] with varying time.

    Returns:
        The padded tensor of the shape [batch, time, dim], where time equals the maximum input sequence length, and
            a tensor, of the shape [batch], storing the actual lengths of each utterance. Padded positions have
            the value 0.
    """
    lengths = [len(x) for x in l]
    T = max(lengths)

    output = []
    for x in l:
        x = np.array(x)
        pad_len = T - len(x)
        padding = np.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        output.append(np.concatenate([x, padding], axis=0))

    return np.stack(output, 0), np.array(lengths, np.int32)


def reverse_sequence_by_length(seq, lengths, flip_dim=1):
    """Reverse input sequences according to their lengths.

    Args:
        seq: [batch, t, ...]
        lengths: [batch], non-padded lengths along reverse dimension
        flip_dim: int, the dimension along which to flip.

    Returns:
        Tensor with the same dimension as input seq, but with sequences reversed in flip_dim.
    """

    ndims = seq.ndim
    # Assume dimension 0 is batch.
    dim_permute = [0] + [flip_dim] + list(range(1, flip_dim)) + list(range(flip_dim + 1, ndims))
    seq_permute = np.transpose(seq, dim_permute)

    output = []
    for i, l in enumerate(lengths):
        x = np.concatenate([np.flip(seq_permute[i, :l], [0]), seq_permute[i, l:]])
        output.append(x)
    output = np.stack(output)

    dim_permute = [0] + list(range(2, flip_dim + 1)) + [1] + list(range(flip_dim + 1, ndims))
    return np.transpose(output, dim_permute)


def collapse_ctc_alignment(seq, input_lens, max_output_len, blank_id=0):
    """Collapse CTC alignment sequence into label sequence.

    Args:
        seq: [batch, max_input_len]
        input_lens: [batch], lengths without padding
        max_output_len: int, output sequence dimension
        blank_id: int, token id of blank.

    Returns:
        Output labels of the shape [batch, max_output_len], and output lengths of the shape [batch]. Padded positions
        have value blank_id.
    """
    batch, max_input_len = seq.shape

    # Replace padded positions with -1, which should not appear in seq or blank.
    length_mask = sequence_mask(input_lens, max_input_len).astype(np.int32)
    seq = seq * length_mask - (1 - length_mask)
    # Pad input sequence with an extra -1.
    seq_comp = np.concatenate([seq[:, 1:], - np.ones([batch, 1], dtype=seq.dtype)], 1)
    # Remove repetitions and non-blanks.
    # For repeated tokens, pick the last appearance.
    label_indicator = np.logical_and(seq != seq_comp, seq != blank_id)
    # Output labels from all alignments, collected into a 1D tensor.
    labels = seq[label_indicator]

    batch_enumeration = np.tile(np.arange(batch)[:, None], [1, max_input_len])
    row_indices = batch_enumeration[label_indicator]
    # Note cumsum() is inclusive.
    label_lens = np.cumsum(label_indicator.astype(np.int32), 1)
    col_indices = label_lens[label_indicator] - 1
    output = np.ones_like(seq) * blank_id
    output[row_indices, col_indices] = labels

    output_lens = np.sum(label_indicator.astype(np.int32), 1)
    return output[:, :min(max_input_len, max_output_len)], np.minimum(output_lens, max_output_len * np.ones_like(output_lens))
