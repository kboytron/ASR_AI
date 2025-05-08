import numpy as np
from utils import sequence_mask, reverse_sequence_by_length
import tensorflow as tf

INF_COST = 1e34
NEG_INF = -1e34


def compute_softmax(x):
    """Compute softmax transformation in a numerically stable way.

    Args:
        x: logits, of the shape [batch, time, d]

    Returns:
        Softmax probability of the input logits, of the shape [batch, time, d].
    """
    out = x - np.max(x, 2, keepdims=True)
    out_exp = np.exp(out)
    exp_sum = np.sum(out_exp, 2, keepdims=True)
    probs = out_exp / exp_sum
    return probs


def compute_forced_alignment(costs, input_lens, labels, output_lens, blank_id=0):
    """Computes CTC forced alignment, which is the optimal token sequence for the given label sequenc
e.

    The algorithm is similar to that of computing alpha values, except that logaddexp is replaced wit
h maximum in
    computing the dynamic programming table.

    Args:
        costs: [batch, t_in, num_classes]
        input_lens: [batch]
        labels: [batch, t_out]
        output_lens: [batch]
        blank_id: int, token id of blank.

    Returns:
        CTC optimal path costs of shape [batch], optimal alignments of shape [batch, t_in].
    """

    B, Tin, C = costs.shape
    _, Tout = labels.shape
    L = 2 * Tout + 1
    alpha = INF_COST * np.ones([B, Tin, L], dtype=costs.dtype)
    # Save moves at each node: 0 for horizontal moves, 1 for diagonal moves, 2 for moves between distinct non-blanks.
    choices = np.zeros([B, Tin, L], dtype=np.int32)

    # t=0, l=0, blank
    # update has shape [batch]
    alpha[:, 0, 0] = costs[:, 0, blank_id]
    # t=0, l=1, first non-blank
    alpha[:, 0, 1] = np.take_along_axis(costs[:, 0, :], labels[:, 0, None], 1).squeeze(1)

    for t in range(1, Tin):

        # Horizontal moves between blanks.
        # shape [batch, 1]
        costs_blank = costs[:, t, blank_id, None]
        # Update has shape [batch, t_out + 1].
        alpha[:, t, 0::2] = alpha[:, t - 1, 0::2] + costs_blank

        # Horizontal moves between non-blanks.
        # shape [batch, t_out]
        costs_nonblank = np.take_along_axis(costs[:, t, :], labels, 1)
        # Update has shape [batch, t_out].
        alpha[:, t, 1::2] = alpha[:, t - 1, 1::2] + costs_nonblank

        # Diagonal moves, blank to non-blank.
        alpha_diagonal = alpha[:, t - 1, 0:-1:2] + costs_nonblank
        mask = np.greater(alpha[:, t, 1::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        # Update has shape [batch, t_out].
        alpha[:, t, 1::2] = alpha[:, t, 1::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice from 0 (default) to 1.
        int_mask = mask.astype(np.int32)
        choices[:, t, 1::2] = choices[:, t, 1::2] * (1 - int_mask) + 1 * int_mask

        # Diagonal moves, non-blank to blank.
        # Update has shape [batch, t_out].
        alpha_diagonal = alpha[:, t - 1, 1::2] + costs_blank
        mask = np.greater(alpha[:, t, 2::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        alpha[:, t, 2::2] = alpha[:, t, 2::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice from 0 (default) to 1.
        int_mask = mask.astype(np.int32)
        choices[:, t, 2::2] = choices[:, t, 2::2] * (1 - int_mask) + 1 * int_mask

        # Diagonal moves, non-blank to distinct non-blank.
        # Check if previous non-blank equals current non-blank.
        # shape [batch, t_out-1]
        distinct_mask = (labels[:, 1:] != labels[:, :-1]).astype(np.float32)
        costs_nonblank_distinct = costs_nonblank[:, 1:] * distinct_mask + INF_COST * (1.0 - distinct_mask)
        # Update has shape [batch, t_out - 1].
        alpha_diagonal = alpha[:, t - 1, 1:-3:2] + costs_nonblank_distinct
        mask = np.greater(alpha[:, t, 3::2], alpha_diagonal)
        float_mask = mask.astype(np.float32)
        alpha[:, t, 3::2] = alpha[:, t, 3::2] * (1.0 - float_mask) + alpha_diagonal * float_mask
        # Update choice to 2.
        int_mask = mask.astype(np.int32)
        choices[:, t, 3::2] = choices[:, t, 3::2] * (1 - int_mask) + 2 * int_mask

    # All labels including blanks.
    all_tokens = np.ones([B, L], dtype=np.int32) * blank_id
    all_tokens[:, 1::2] = labels
    input_seq_mask = sequence_mask(input_lens, Tin).astype(np.int32)

    # Reverse in time for back tracking.
    alpha_reversed = reverse_sequence_by_length(alpha, input_lens)
    # shape [batch, t_in, L], set out-of-bound choices to 0.
    choices_reversed = reverse_sequence_by_length(choices * input_seq_mask[:, :, None], input_lens)

    # shape [batch, 1]
    output_lens = output_lens[:, None]
    alpha_blank = np.take_along_axis(alpha_reversed[:, 0, :], 2 * output_lens, 1)
    alpha_nonblank = np.take_along_axis(alpha_reversed[:, 0, :], 2 * output_lens - 1, 1)

    # costs of best paths.
    blank_vs_non_mask = alpha_blank < alpha_nonblank
    # shape [batch, 1], indexing into all_tokens of length L.
    last_index = np.where(blank_vs_non_mask, 2 * output_lens, 2 * output_lens - 1)
    last_token = np.take_along_axis(all_tokens, last_index, 1)
    # Back pointer.
    last_choice = np.take_along_axis(choices_reversed[:, 0, :], last_index, 1)

    blank_vs_non_mask = blank_vs_non_mask.astype(np.float32)
    best_costs = alpha_blank * blank_vs_non_mask + alpha_nonblank * (1 - blank_vs_non_mask)
    best_costs = best_costs.squeeze(1)

    # Best path contains one token per input step.
    paths = [last_token]

    for t in range(1, Tin):
        # choice gives the change of index in all tokens.
        last_index = last_index - last_choice
        last_token = np.take_along_axis(all_tokens, last_index, 1)
        paths.append(last_token)
        last_choice = np.take_along_axis(choices_reversed[:, t, :], last_index, 1)

    paths = np.concatenate(paths, 1)
    paths = reverse_sequence_by_length(paths, input_lens)
    # Set out-of-bound tokens to blank.
    paths = paths * input_seq_mask + blank_id * (1 - input_seq_mask)

    return best_costs, paths


def beam_search(log_probs, input_lens):
    """Performs CTC beam search.

    Args:
        logits: [batch, T, num_classes]
        input_lens: [batch]

    Returns:
        A list of hypothesis (without blanks) for each input utterance.
    """
    # TF beam search assumes [T, B, D].
    log_probs = np.transpose(log_probs, [1, 0, 2])
    # TF assumes blank id to be num_classes-1 instead of 0.
    log_probs_tf = np.concatenate([log_probs[:, :, 1:], log_probs[:, :, 0, None]], axis=2)

    decoded, _ = tf.nn.ctc_beam_search_decoder(log_probs_tf, input_lens, beam_width=30, top_paths=1)
    # Padded positions in hyps will have value 0.
    hyps = tf.sparse.to_dense(decoded[0], default_value=-1).numpy() + 1
    return [h[h!=0] for h in hyps]


def compute_ctc_alpha(logits, input_lens, labels, output_lens, blank_id=0):
    """Computes CTC forward scores.

    Args:
        logits: [batch, t_in, num_classes]
        input_lens: [batch]
        labels: [batch, t_out]
        output_lens: [batch]
        blank_id: int, token id of blank.

    Returns:
        Log-likelihood of shape [batch], and log alpha probabilities of shape [batch, t_in, num_classes], where padded
        positions have the value -inf.
    """

    B, Tin, C = logits.shape
    _, Tout = labels.shape
    L = 2 * Tout + 1
    alpha = NEG_INF * np.ones([B, Tin, L], dtype=logits.dtype)

    # t=0, l=0, blank
    # Update has shape [batch]
    alpha[:, 0, 0] = logits[:, 0, blank_id]
    # t=0, l=1, first non-blank label
    alpha[:, 0, 1] = np.take_along_axis(logits[:, 0, :], labels[:, 0, None], axis=1).squeeze(1)

    for t in range(1, Tin):

        # Horizontal moves between blanks.
        # shape [batch, 1]
        logits_blank = logits[:, t, blank_id, None]
        # Update has shape [batch, t_out + 1]
        alpha[:, t, 0::2] = alpha[:, t - 1, 0::2] + logits_blank

        # Horizontal moves between non-blanks.
        # shape [batch, t_out]
        logits_nonblank = np.take_along_axis(logits[:, t, :], labels, axis=1)
        # update has shape [batch, t_out].
        alpha[:, t, 1::2] = alpha[:, t - 1, 1::2] + logits_nonblank

        # Diagonal moves, blank to non-blank.
        # Update has shape [batch, t_out].
        alpha[:, t, 1::2] = np.logaddexp(alpha[:, t, 1::2], alpha[:, t - 1, 0:-1:2] + logits_nonblank)

        # Diagonal moves, non-blank to blank.
        # Update has shape [batch, t_out].
        alpha[:, t, 2::2] = np.logaddexp(alpha[:, t, 2::2], alpha[:, t - 1, 1::2] + logits_blank)

        # Diagonal moves, non-blank to distinct non-blank.
        # Check if previous non-blank equals current non-blank.
        # shape [batch, t_out-1]
        distinct_mask = (labels[:, 1:] != labels[:, :-1]).astype(logits.dtype)
        logits_nonblank_distinct = logits_nonblank[:, 1:] * distinct_mask + NEG_INF * (1.0 - distinct_mask)
        # Update has shape [batch, t_out - 1]
        alpha[:, t, 3::2] = np.logaddexp(alpha[:, t, 3::2], alpha[:, t - 1, 1:-3:2] + logits_nonblank_distinct)

    # shape [batch]
    batch_enumeration = np.arange(B)
    alpha_blank = alpha[batch_enumeration, input_lens - 1, 2 * output_lens]
    alpha_nonblank = alpha[batch_enumeration, input_lens - 1, 2 * output_lens - 1]
    log_likelihood = np.logaddexp(alpha_blank, alpha_nonblank)

    # Set out-of-bound positions to NEG_INF.
    time_mask = sequence_mask(input_lens, Tin)[:, :, None].astype(logits.dtype)
    alpha = alpha * time_mask + NEG_INF * (1.0 - time_mask)
    label_mask = sequence_mask(2 * output_lens + 1, L)[:, None, :].astype(logits.dtype)
    alpha = alpha * label_mask + NEG_INF * (1.0 - label_mask)

    return log_likelihood, alpha


def compute_ctc_beta(logits, input_lens, labels, output_lens, blank_id=0):
    """Computes CTC backward scores.

    Args:
        logits: [batch, t_in, num_classes]
        input_lens: [batch]
        labels: [batch, t_out]
        output_lens: [batch]
        blank_id: int, token id of blank.

    Returns:
        Log-likelihood of shape [batch], and log beta probabilities of shape [batch, t_in, num_classes], where padded
        positions have the value -inf. Note the backward probabilities are computed in the same way as alpha
        probabilities, with reversed inputs and outputs.
    """

    # Reverse sequences in time and label.
    logits_reversed = reverse_sequence_by_length(logits, input_lens)
    labels_reversed = reverse_sequence_by_length(labels, output_lens)

    log_likelihood, beta_reversed = compute_ctc_alpha(logits_reversed, input_lens, labels_reversed, output_lens, blank_id)

    # Reverse back.
    beta_reversed = reverse_sequence_by_length(beta_reversed, input_lens, flip_dim=1)
    beta = reverse_sequence_by_length(beta_reversed, 2 * output_lens + 1, flip_dim=2)

    return log_likelihood, beta


def compute_ctc_loss_and_grad(logits, input_lens, labels, output_lens, blank_id=0):
    """Computes CTC loss and gradient.

    Args:
        logits: [batch, t_in, num_classes]
        input_lens: [batch]
        labels: [batch, t_out]
        output_lens: [batch]
        blank_id: int, token id of blank.

    Returns:
        CTC loss which is negative log-likelihood, of shape [batch], and the gradient of loss with respect to logits,
        of shape [batch, t_in, num_classes].
    """

    B, Tin, C = logits.shape
    _, Tout = labels.shape
    L = 2 * Tout + 1

    lle, alpha = compute_ctc_alpha(logits, input_lens, labels, output_lens)
    _, beta = compute_ctc_beta(logits, input_lens, labels, output_lens)

    # shape [B, Tin, L]
    logits_ab = np.zeros([B, Tin, L], dtype=logits.dtype)
    # shape [B, Tout]
    batch_enumeration = np.tile(np.arange(B)[:, None], [1, Tout])
    logits_ab[:, :, 0::2] = np.tile(logits[:, :, blank_id, None], [1, 1, Tout + 1])
    logits_ab[:, :, 1::2] = np.transpose(np.transpose(logits, [0, 2, 1])[batch_enumeration, labels], [0, 2, 1])

    # shape [B, L]
    labels_ab = np.zeros([B, L], dtype=np.int64)
    labels_ab[:, 0::2] = blank_id
    labels_ab[:, 1::2] = labels
    # shape [B, L, C], the correspondence between expanded labels and classes
    # labels_ab_onehot = torch.nn.functional.one_hot(labels_ab, C).float()

    labels_ab_onehot = np.zeros([B, L, C], dtype=logits_ab.dtype)
    row_idx = np.tile(np.arange(B, dtype=np.int64)[:, None], [1, L])
    col_idx = np.tile(np.arange(L, dtype=np.int64)[None, :], [B, 1])
    labels_ab_onehot[row_idx[:], col_idx[:], labels_ab[:]] = 1.0

    # shape [B, Tin, C]. Graves et al 2006, eqn 15 & 16.
    ab = alpha + beta - 2 * logits_ab - lle[:, None, None]
    grad_lle_over_probs = np.einsum('btl,blc->btc', np.exp(ab), labels_ab_onehot)
    probs = compute_softmax(logits)
    # Mask out gradient for out-of-bound positions.
    grad_lle_over_probs *= sequence_mask(input_lens, Tin)[:, :, None].astype(logits.dtype)
    grad_lle = grad_lle_over_probs * probs - np.einsum('btc,btc->bt', grad_lle_over_probs, probs)[:, :, None] * probs
    return - lle, - grad_lle
