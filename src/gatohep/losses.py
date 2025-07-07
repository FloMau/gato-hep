import tensorflow as tf


def low_bkg_penalty(bkg_yields, threshold=10.0):
    """
    Compute a penalty for background yields below a specified threshold.

    Parameters
    ----------
    bkg_yields : tf.Tensor
        A tensor of shape [ncat] representing the background yields in each category.
    threshold : float, optional
        The minimum background yield threshold. Default is 10.0.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the total penalty summed over all categories.
    """
    # penalty per category
    penalty_vals = (tf.nn.relu(threshold - bkg_yields)) ** 2

    return tf.reduce_sum(penalty_vals)


def high_bkg_uncertainty_penalty(bkg_sumsq, bkg_yields, rel_threshold=0.2):
    """
    Penalize bins whose relative Monte Carlo uncertainty exceeds a threshold.

    Parameters
    ----------
    bkg_sumsq : tf.Tensor
        A tensor of shape [ncat] representing the sum of squared weights
            (w_i^2) in each bin.
    bkg_yields : tf.Tensor
        A tensor of shape [ncat] representing the sum of weights (w_i) in each bin.
    rel_threshold : float, optional
        The relative uncertainty threshold. Default is 0.2 (20%).

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the total penalty summed over all bins.
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
