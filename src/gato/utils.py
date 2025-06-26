import hist
import numpy as np
import tensorflow as tf


def df_dict_to_tensors(data_dict):
    """
    Convert a dictionary of DataFrames to a dictionary of tensors.

    Parameters
    ----------
    data_dict : dict
        A dictionary where keys are process names and values are pandas.DataFrames
        with columns "NN_output" and "weight".

    Returns
    -------
    dict
        A dictionary where keys are process names and values are dictionaries
        containing tensors with keys "x" and "w".
    """
    tensor_data = {}
    for proc, df in data_dict.items():
        tensor_data[proc] = {
            col: tf.constant(df[col].values, dtype=tf.float32) for col in df.columns
        }
    return tensor_data


def create_hist(data, weights=None, bins=50, low=0.0, high=1.0, name="NN_output"):
    """
    Create a histogram from data and weights.

    Parameters
    ----------
    data : array_like
        Data to be binned.
    weights : array_like, optional
        Weights for the data. Default is None.
    bins : int or array_like, optional
        Number of bins or bin edges. Default is 50.
    low : float, optional
        Lower bound of the histogram range. Default is 0.0.
    high : float, optional
        Upper bound of the histogram range. Default is 1.0.
    name : str, optional
        Name of the histogram axis. Default is "NN_output".

    Returns
    -------
    hist.Hist
        A histogram object.
    """
    if isinstance(bins, int):
        h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    else:
        h = hist.Hist.new.Var(bins, name=name).Weight()
    if weights is not None:
        h.fill(data, weight=weights)
    else:
        h.fill(data)
    return h


def safe_sigmoid(z, steepness):
    """
    Compute a numerically stable sigmoid function.

    Parameters
    ----------
    z : tf.Tensor
        Input tensor.
    steepness : float
        Steepness of the sigmoid function.

    Returns
    -------
    tf.Tensor
        Output tensor after applying the sigmoid function.
    """
    z_clipped = tf.clip_by_value(-steepness * z, -75.0, 75.0)
    return 1.0 / (1.0 + tf.exp(z_clipped))


def asymptotic_significance(S, B, eps=1e-9):
    """
    Compute the asymptotic significance using the Asimov formula.

    Parameters
    ----------
    S : tf.Tensor
        Signal counts.
    B : tf.Tensor
        Background counts.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-9.

    Returns
    -------
    tf.Tensor
        Asymptotic significance values.
    """
    safe_B = tf.maximum(B, eps)
    ratio = S / safe_B
    Z_asimov = tf.sqrt(2.0 * ((S + safe_B) * tf.math.log(1.0 + ratio) - S))
    Z_approx = S / tf.sqrt(safe_B)
    return tf.where(ratio < 0.1, Z_approx, Z_asimov)


def compute_significance_from_hists(h_signal, h_bkg_list):
    """
    Compute the significance from signal and background histograms.

    Parameters
    ----------
    h_signal : hist.Hist
        Histogram of signal events.
    h_bkg_list : list of hist.Hist
        List of histograms for background events.

    Returns
    -------
    float
        Combined significance value.
    """
    B_vals = sum([h_bkg.values() for h_bkg in h_bkg_list])
    S_vals = h_signal.values()
    S_tensor = tf.constant(S_vals, dtype=tf.float32)
    B_tensor = tf.constant(B_vals, dtype=tf.float32)
    Z_bins = asymptotic_significance(S_tensor, B_tensor)
    return np.sqrt(np.sum(Z_bins.numpy()**2))


def align_boundary_tracks(history, dist_tol=0.02, gap_max=20):
    """
    Align boundary tracks across epochs.

    Parameters
    ----------
    history : list of lists
        Each inner list contains boundary values at a specific epoch.
    dist_tol : float, optional
        Maximum distance tolerance for matching boundaries. Default is 0.02.
    gap_max : int, optional
        Maximum gap in epochs for considering a track inactive. Default is 20.

    Returns
    -------
    ndarray
        A 2D array of shape (n_epochs, n_tracks) with NaNs where no boundary exists.
    """
    if not history:
        return np.empty((0, 0))

    n_epochs = len(history)
    n_tracks = len(history[0])
    tracks = np.full((n_epochs, n_tracks), np.nan)
    last_val = np.array(history[0] + [np.nan] * (n_tracks - len(history[0])))
    last_seen = np.zeros(n_tracks, dtype=int)

    tracks[0, :len(history[0])] = history[0]

    def add_track():
        nonlocal tracks, last_val, last_seen, n_tracks
        tracks = np.hstack([tracks, np.full((n_epochs, 1), np.nan)])
        last_val = np.append(last_val, np.nan)
        last_seen = np.append(last_seen, -gap_max * 2)
        n_tracks += 1
        return n_tracks - 1

    for ep in range(1, n_epochs):
        cuts = list(history[ep])

        for t in range(n_tracks):
            if np.isnan(last_val[t]) or not cuts:
                continue
            dist = np.abs(np.asarray(cuts) - last_val[t])
            j = np.argmin(dist)
            if dist[j] < dist_tol:
                last_val[t] = cuts.pop(j)
                last_seen[t] = ep
                tracks[ep, t] = last_val[t]

        for cut in list(cuts):
            cand = np.where(
                (np.isnan(tracks[ep, :])) &
                (ep - last_seen < gap_max) &
                (np.abs(last_val - cut) < dist_tol)
            )[0]
            if cand.size:
                t = cand[0]
                last_val[t] = cut
                last_seen[t] = ep
                tracks[ep, t] = cut
                cuts.remove(cut)

        for cut in cuts:
            t = add_track()
            last_val[t] = cut
            last_seen[t] = ep
            tracks[ep, t] = cut

    return tracks
