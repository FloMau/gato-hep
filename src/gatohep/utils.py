import hist
import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit


class TemperatureScheduler:
    r"""
    Anneal a GATO model's ``temperature`` variable during training.

    Parameters
    ----------
    model : gato_gmm_model
        The model whose ``temperature`` (tf.Variable) is updated in-place.
    t_initial : float
        Temperature at epoch 0.
    t_final : float
        Temperature at `total_epochs`.
    total_epochs : int
        Number of epochs that constitute one full annealing cycle.
    mode : {"exponential", "cosine"}, optional
        * **"exponential"** - geometric decay
          :math:`T_e = T_0 (T_f/T_0)^{e/E}`
        * **"cosine"** - half-cosine schedule
          :math:`T_e = T_f + 0.5\,(T_0 - T_f)\,[1+\cos(\pi e/E)]`
    verbose : bool, optional
        If *True*, prints the new temperature each epoch.

    Notes
    -----
    Call :py:meth:`update` **once per epoch** (or more often, if desired).
    """

    def __init__(
        self,
        model,
        t_initial=1.0,
        t_final=0.01,
        *,
        total_epochs=100,
        mode="exponential",
        verbose=False,
    ):
        self.model = model
        self.t0 = float(t_initial)
        self.tf = float(t_final)
        self.E = int(total_epochs)
        self.mode = mode.lower()
        self.verbose = verbose

        if self.mode not in ("exponential", "cosine"):
            raise ValueError("mode must be 'exponential' or 'cosine'")

    def _schedule(self, epoch: int) -> float:
        """Return temperature for epoch *epoch* based on the selected mode."""
        tau = epoch / max(1, self.E)  # normalised 0 -> 1
        if self.mode == "exponential":
            return self.t0 * (self.tf / self.t0) ** tau
        # cosine
        return self.tf + 0.5 * (self.t0 - self.tf) * (1 + np.cos(np.pi * tau))

    def update(self, epoch: int):
        """Update ``model.temperature`` for the given epoch index."""
        new_T = self._schedule(epoch)

        # Works for both tf.Variable *and* plain float
        if hasattr(self.model, "temperature"):
            if "Variable" in type(self.model.temperature).__name__:
                self.model.temperature.assign(new_T)  # tf.Variable case
            else:
                self.model.temperature = float(new_T)  # plain float attribute
        else:
            raise AttributeError("Model has no attribute 'temperature'.")

        if self.verbose:
            print(f"[TempScheduler-{self.mode}] epoch {epoch:3d} -> T = {new_T:.4f}")


class SteepnessScheduler(TemperatureScheduler):
    """
    Anneal every ``cfg["k"]`` in a ``gato_sigmoid_model``.

    Inherits all arguments from ``TemperatureScheduler`` but updates the
    *steepness* parameters stored in ``model.var_cfg[j]["k"]``.

    Notes
    -----
    * Call :py:meth:`update(epoch)` once per epoch, exactly like the
      TemperatureScheduler.
    * Works whether each ``k`` is a ``tf.Variable`` or a plain float.
    """

    def update(self, epoch: int):
        new_k = self._schedule(epoch)

        # loop over every discriminant in the model
        for cfg in self.model.var_cfg:
            k_var = cfg["k"]
            if "Variable" in type(k_var).__name__:  # tf.Variable
                k_var.assign(new_k)
            else:  # plain float
                cfg["k"] = float(new_k)

        if self.verbose:
            print(
                f"[SteepnessScheduler-{self.mode}] epoch {epoch:3d} -> k = {new_k:.4f}"
            )


class LearningRateScheduler(TemperatureScheduler):
    """Cosine or exponential annealing for an optimizer's learning rate."""

    def __init__(
        self,
        optimizer,
        lr_initial: float = 0.5,
        lr_final: float = 0.001,
        *,
        total_epochs: int = 100,
        mode: str = "cosine",
        verbose: bool = False,
    ):
        super().__init__(
            model=optimizer,
            t_initial=lr_initial,
            t_final=lr_final,
            total_epochs=total_epochs,
            mode=mode,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def update(self, epoch: int) -> float:
        """Update ``optimizer.learning_rate`` based on the current epoch."""
        new_lr = self._schedule(epoch)

        lr_attr = getattr(self.optimizer, "learning_rate", None)
        if hasattr(lr_attr, "assign"):
            lr_attr.assign(new_lr)
        elif lr_attr is not None:
            # Keras optimizers expose a property setter for plain floats.
            self.optimizer.learning_rate = new_lr
        else:
            lr_attr = getattr(self.optimizer, "lr", None)
            if hasattr(lr_attr, "assign"):
                lr_attr.assign(new_lr)
            elif lr_attr is not None:
                setattr(self.optimizer, "lr", new_lr)
            else:
                raise AttributeError(
                    "Optimizer has no assignable learning rate attribute."
                )

        if self.verbose:
            print(
                f"[LR scheduler-{self.mode}] epoch {epoch:3d} -> LR = {new_lr:.6f}"
            )
        return new_lr


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
    return np.sqrt(np.sum(Z_bins.numpy() ** 2))


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

    tracks[0, : len(history[0])] = history[0]

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
                (np.isnan(tracks[ep, :]))
                & (ep - last_seen < gap_max)
                & (np.abs(last_val - cut) < dist_tol)
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


def compute_mass_reweight_factors(
    model,
    data_dict,
    *,
    signal_labels=None,
    feature_key="NN_output",
    mass_column="mass",
    weight_column="weight",
    mass_sb_low=100.0,
    mass_sb_high=180.0,
    mass_sig_low=123.5,
    mass_sig_high=126.5,
    nbins=10,
):
    """
    Fit an exponential to each category's diphoton-mass spectrum and
    return per-bin factors that map the continuum yield in the full
    sideband (100-180 GeV by default) to the yield expected in the
    signal window (125 +/- 1 sigma).

    """

    def is_signal(name: str) -> bool:
        if signal_labels is None:
            return name.startswith("signal")
        return name in signal_labels

    n_cats = model.n_cats
    bin_width = (mass_sb_high - mass_sb_low) / nbins
    default_ratio = (mass_sig_high - mass_sig_low) / (mass_sb_high - mass_sb_low)

    hists = [
        hist.Hist.new.Reg(nbins, mass_sb_low, mass_sb_high, name="mass").Weight()
        for _ in range(n_cats)
    ]
    raw_sigwin = np.zeros(n_cats, dtype=np.float64)

    tf_inputs = {}
    for proc, df in data_dict.items():
        if df.empty:
            continue
        values = np.stack(df[feature_key].values)
        tf_inputs[proc] = {"NN_output": tf.constant(values, dtype=tf.float32)}

    assignments = model.get_bin_indices(tf_inputs)

    for proc, df in data_dict.items():
        if proc not in assignments or df.empty or is_signal(proc):
            continue

        cat_ids = assignments[proc].numpy()
        masses = df[mass_column].to_numpy()
        weights = df[weight_column].to_numpy()

        for k in range(n_cats):
            mask = cat_ids == k
            if not np.any(mask):
                continue

            hists[k].fill(masses[mask], weight=weights[mask])
            in_sig = mask & (masses >= mass_sig_low) & (masses < mass_sig_high)
            if np.any(in_sig):
                raw_sigwin[k] += weights[in_sig].sum()

    def exp_func(x, A, B):
        return A * np.exp(B * x)

    def integral_exp(A, B, x1, x2):
        if abs(B) < 1e-10:
            return A * (x2 - x1)
        return (A / B) * (np.exp(B * x2) - np.exp(B * x1))

    factors = np.full(n_cats, default_ratio, dtype=np.float64)

    for idx, hist_obj in enumerate(hists):
        vals = hist_obj.values()
        if not vals.size or float(np.sum(vals)) <= 0.0:
            continue

        edges = hist_obj.axes[0].edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        p0 = [max(vals[0], 1e-6), -0.03]
        try:
            (A, B), _ = curve_fit(
                exp_func,
                centers,
                vals,
                p0=p0,
                maxfev=2000,
            )
            pred_sig = integral_exp(A, B, mass_sig_low, mass_sig_high) / bin_width
            pred_tot = integral_exp(A, B, mass_sb_low, mass_sb_high) / bin_width
            if pred_tot > 0.0:
                factors[idx] = pred_sig / pred_tot
        except RuntimeError:
            pass

        if raw_sigwin[idx] <= 0.0:
            factors[idx] = default_ratio

    return factors.astype(np.float32)


def sample_truncated_exponential(rng, slope, size, *, low, high):
    """
    Draw samples from a truncated exponential distribution.

    Parameters
    ----------
    rng : np.random.Generator
        Random-number generator used for sampling.
    slope : float
        Positive exponential slope ``λ`` in ``exp(-λ·x)``.
    size : int
        Number of samples to draw.
    low : float
        Lower bound of the truncation interval.
    high : float
        Upper bound of the truncation interval (must exceed *low*).

    Returns
    -------
    np.ndarray
        Array of shape ``(size,)`` with samples in ``[low, high]``.
    """
    if high <= low:
        raise ValueError("upper bound must exceed lower bound for sampling.")
    span = high - low
    if slope <= 0.0:
        return low + rng.random(size) * span
    norm = 1.0 - np.exp(-slope * span)
    u = rng.random(size)
    # Guard against numerical issues when u * norm -> 1.
    epsilon = np.finfo(np.float64).tiny
    inner = np.clip(1.0 - u * norm, epsilon, None)
    return low - np.log(inner) / slope


def generate_resonance_toy_data(
    n_signal1=60_000,
    n_signal2=60_000,
    n_bkg=400_000,
    *,
    noise_scale=0.2,
    mass_sigma=1.5,
    seed=7,
    background_slopes=None,
):
    """
    Extend the 3-class toy dataset with Higgs-like diphoton masses.

    Parameters
    ----------
    n_signal1, n_signal2, n_bkg : int
        Event counts passed to :func:`generate_toy_data_3class_3D`.
    noise_scale : float, optional
        Multiplicative feature noise forwarded to the base generator.
    mass_sigma : float, optional
        Gaussian width of the resonant signal peak.
    seed : int, optional
        Seed for deterministic feature and mass sampling.
    background_slopes : sequence of float, optional
        Exponential slopes for the continuum components.  If omitted,
        a default tuple is used and cycled over all background processes.

    Returns
    -------
    dict[str, pandas.DataFrame]
        The original dataframes augmented with a ``"mass"`` column.
    """
    from gatohep.data_generation import generate_toy_data_3class_3D

    data = generate_toy_data_3class_3D(
        n_signal1=n_signal1,
        n_signal2=n_signal2,
        n_bkg=n_bkg,
        noise_scale=noise_scale,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    mass_range = (100.0, 180.0)
    slopes = background_slopes or (0.02,)
    bkg_names = [p for p in data if p.startswith("bkg")]
    slope_map = {
        name: slopes[idx % len(slopes)] for idx, name in enumerate(bkg_names)
    }

    low, high = mass_range
    for proc, df in data.items():
        n_events = len(df)
        if n_events == 0:
            df["mass"] = np.array([], dtype=np.float32)
            continue
        if proc.startswith("signal"):
            masses = np.clip(rng.normal(125.0, mass_sigma, size=n_events), low, high)
        else:
            slope = slope_map.get(proc, slopes[0])
            masses = sample_truncated_exponential(
                rng, slope, n_events, low=low, high=high
            )
        df["mass"] = masses.astype(np.float32)
    return data


def slice_to_2d_features(data_dict):
    """
    Drop the background node of the pseudo-softmax feature vector.

    Parameters
    ----------
    data_dict : dict[str, pandas.DataFrame]
        Input dictionary produced by the toy generator.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Shallow copies where ``"NN_output"`` only retains the first two
        components per event.
    """
    sliced = {}
    for proc, df in data_dict.items():
        df_copy = df.copy()
        df_copy["NN_output"] = [vals[:2] for vals in df_copy["NN_output"].values]
        sliced[proc] = df_copy
    return sliced


def convert_mass_data_to_tensors(data_dict):
    """
    Convert the dataframe-based storage into TensorFlow tensors.

    Parameters
    ----------
    data_dict : dict[str, pandas.DataFrame]
        Mapping whose dataframes contain ``NN_output``, ``weight`` and ``mass``.

    Returns
    -------
    dict[str, dict[str, tf.Tensor]]
        Dictionary mirroring the input keys with tensor-valued payload.
    """
    tensors = {}
    for proc, df in data_dict.items():
        nn = np.stack(df["NN_output"].values)
        w = df["weight"].values
        mass = df["mass"].values
        tensors[proc] = {
            "NN_output": tf.constant(nn, dtype=tf.float32),
            "weight": tf.constant(w, dtype=tf.float32),
            "mass": tf.constant(mass, dtype=tf.float32),
        }
    return tensors


def build_mass_histograms(
    data_dict,
    *,
    bins=60,
    mass_range=(100.0, 180.0),
    axis_name="mass",
):
    """
    Create diphoton-mass histograms for every process dataframe.

    Parameters
    ----------
    data_dict : dict[str, pandas.DataFrame]
        Mapping with ``"mass"`` and ``"weight"`` columns.
    bins : int, optional
        Number of uniform bins in the specified mass range.
    mass_range : tuple[float, float], optional
        Inclusive histogram range in GeV.
    axis_name : str, optional
        Name assigned to the histogram axis (for plotting labels).

    Returns
    -------
    dict[str, hist.Hist]
        One histogram per process.
    """
    hists = {}
    for proc, df in data_dict.items():
        hists[proc] = create_hist(
            df["mass"].values,
            weights=df["weight"].values,
            bins=bins,
            low=mass_range[0],
            high=mass_range[1],
            name=axis_name,
        )
    return hists


def build_category_mass_maps(
    assignments,
    data_dict,
    n_cats,
    *,
    bins=40,
    mass_range=(100.0, 180.0),
    axis_name="mass",
):
    """
    Build per-category diphoton-mass histograms for each process.

    Parameters
    ----------
    assignments : dict[str, np.ndarray]
        Hard bin assignments produced by :meth:`gato_gmm_model.get_bin_indices`.
    data_dict : dict[str, pandas.DataFrame]
        Input frames containing ``"mass"`` and ``"weight"``.
    n_cats : int
        Total number of GMM categories.
    bins, mass_range, axis_name :
        Passed through to :func:`create_hist`.

    Returns
    -------
    list[dict[str, hist.Hist]]
        One entry per category with per-process histograms.
    """
    per_cat = []
    for k in range(n_cats):
        proc_hists = {}
        for proc, df in data_dict.items():
            cat_ids = assignments.get(proc)
            if cat_ids is None:
                continue
            mask = cat_ids == k
            masses = df["mass"].values[mask]
            weights = df["weight"].values[mask]
            h = hist.Hist.new.Reg(
                bins,
                mass_range[0],
                mass_range[1],
                name=axis_name
            ).Weight()
            if masses.size:
                h.fill(masses, weight=weights)
            proc_hists[proc] = h
        per_cat.append(proc_hists)
    return per_cat
