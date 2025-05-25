import tensorflow as tf
import hist
import numpy as np

def df_dict_to_tensors(data_dict):
    """
    Input:  data_dict: proc_name -> pandas.DataFrame with columns "NN_output","weight"
    Output: tensor_data: proc_name -> {"x": tf.Tensor, "w": tf.Tensor}
    """
    tensor_data = {}
    for proc, df in data_dict.items():
        tensor_data[proc] = {
            col: tf.constant(df[col].values, dtype=tf.float32) for col in df.columns
        }
    return tensor_data

def create_hist(data, weights=None, bins=50, low=0.0, high=1.0, name="NN_output"):
    # If bins is an integer, we do regular binning:
    if isinstance(bins, int):
        h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    # Otherwise, assume bins is an array of edges:
    else:
        h = hist.Hist.new.Var(bins, name=name).Weight()
    if weights is not None:
        h.fill(data, weight=weights)
    else:
        h.fill(data)
    return h

def align_boundary_tracks(history, dist_tol=0.02, gap_max=20):
    """
    history : list of lists, each inner list = boundaries at that epoch
    returns : (n_epochs, n_tracks) ndarray with NaNs where absent
    """
    if not history:
        return np.empty((0, 0))

    n_epochs    = len(history)
    n_tracks    = len(history[0])
    tracks      = np.full((n_epochs, n_tracks), np.nan)
    last_val    = np.array(history[0] + [np.nan]*(n_tracks - len(history[0])))
    last_seen   = np.zeros(n_tracks, dtype=int)          # epoch 0

    tracks[0, :len(history[0])] = history[0]

    def add_track():
        nonlocal tracks, last_val, last_seen, n_tracks
        tracks    = np.hstack([tracks, np.full((n_epochs, 1), np.nan)])
        last_val  = np.append(last_val,  np.nan)
        last_seen = np.append(last_seen, -gap_max*2)     # very old
        n_tracks += 1
        return n_tracks - 1

    for ep in range(1, n_epochs):
        cuts = list(history[ep])

        # --- try to match existing tracks first --------------------------
        for t in range(n_tracks):
            if np.isnan(last_val[t]) or not cuts:
                continue
            dist = np.abs(np.asarray(cuts) - last_val[t])
            j = np.argmin(dist)
            if dist[j] < dist_tol:
                last_val[t]    = cuts.pop(j)
                last_seen[t]   = ep
                tracks[ep, t]  = last_val[t]

        # --- any remaining cuts: first try inactive recent tracks --------
        for cut in list(cuts):              # iterate over a *copy*
            # candidates: inactive, recently seen, distance small
            cand = np.where(
                (np.isnan(tracks[ep, :])) &
                (ep - last_seen < gap_max) &
                (np.abs(last_val - cut) < dist_tol)
            )[0]
            if cand.size:
                t            = cand[0]
                last_val[t]  = cut
                last_seen[t] = ep
                tracks[ep, t] = cut
                cuts.remove(cut)

        # --- still left → brand-new columns ------------------------------
        for cut in cuts:
            t            = add_track()
            last_val[t]  = cut
            last_seen[t] = ep
            tracks[ep, t] = cut

    return tracks
