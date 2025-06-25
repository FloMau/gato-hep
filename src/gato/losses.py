import tensorflow as tf


def low_bkg_penalty(bkg_yields, threshold=10.0):
    """
    bkg_yields: tf.Tensor of shape [ncat], the background yields in each category.
    The penalty is summed over all categories.
    """
    # penalty per category
    penalty_vals = (
        tf.nn.relu(threshold - bkg_yields)
    )**2
    # sum over categories
    total_penalty = tf.reduce_sum(penalty_vals)

    return total_penalty


def high_bkg_uncertainty_penalty(bkg_sumsq, bkg_yields, rel_threshold=0.2):
    """
    Penalize bins whose *relative* MC uncertainty exceeds rel_threshold.

    bkg_sumsq:   tf.Tensor [ncat], sum of w_i^2 in each bin
    bkg_yields:  tf.Tensor [ncat], sum of w_i in each bin
    rel_threshold: float, e.g. 0.2 for 20% relative error

    penalty_j = max(sigma_j / B_j - rel_threshold, 0)^2
    total_penalty = sum_j penalty_j
    """
    # avoid division by zero
    safe_B = tf.maximum(bkg_yields, 1e-8)
    # sigma_j = sqrt(sum_i w_i^2)
    sigma = tf.sqrt(tf.maximum(bkg_sumsq, 0.0))
    rel_unc = sigma / safe_B

    # only penalize above threshold
    over = tf.nn.relu(rel_unc - rel_threshold)
    penalty_per_bin = over**2

    return tf.reduce_sum(penalty_per_bin)
