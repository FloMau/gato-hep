from gatohep.utils import asymptotic_significance, safe_sigmoid
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence


tfd = tfp.distributions


class gato_gmm_model(tf.Module):
    """
    A differentiable category model based on a Gaussian mixture.

    The model learns, for each of `n_cats`:
      - Mixture logits (which give the mixing weights),
      - Mean vector (of dimension `dim`),
      - An unconstrained lower-triangular matrix that is transformed into a
        positive-definite Cholesky factor for the covariance.

    The per-event soft membership is computed by evaluating the log pdf of each
    Gaussian at the event's feature vector and adding the log mixture weight.
    A temperatured softmax is then applied.

    Attributes
    ----------
    n_cats : int
        Number of categories (Gaussian components).
    dim : int
        Dimensionality of the feature space.
    temperature : float
        Temperature parameter for the softmax function.
    mean_norm : {"softmax", "sigmoid"}, optional
        Strategy that constrains the component means **at initialisation
        time**:

        * ``"softmax"`` - Raw means are passed through a softmax over
          ``dim + 1`` logits, so every mean lies on the *dim*-simplex
          Recommended for softmax-classifier outputs.
        * ``"sigmoid"`` - Raw means are transformed with a component-wise
          sigmoid and then *linearly scaled* into ``mean_range``.
          Recommended for a feature space e.g. spanned by mutliple 1D discriminants.
          The range of each component can be customized with
          the `mean_range` parameter.

        Default is ``"softmax"``.
    mean_range : tuple(float, float) or sequence of tuples, optional
        Lower and upper bounds that define the allowed interval(s) when
        ``mean_norm="sigmoid"``.  Accepts
        * a **single** ``(lo, hi)`` tuple, applied to *every* dimension, or
        * a **list**/tuple of ``dim`` separate ``(lo, hi)`` pairs for\
        per-dimension ranges.

    mixture_logits : tf.Variable
        Trainable logits for the mixture weights.
    means : tf.Variable
        Trainable mean vectors for each Gaussian component.
    unconstrained_L : tf.Variable
        Trainable unconstrained lower-triangular matrices for covariance factors.
    """

    def __init__(
        self,
        n_cats,
        dim,
        temperature=1.0,
        mean_norm: str = "softmax",
        mean_range: tuple | list = (0.0, 1.0),
        cov_offdiag_damping: float = 0.1,
        name="gato_gmm_model",
    ):
        """
        Initialize the gato Gaussian mixture model.

        Parameters
        ----------
        n_cats : int
            Number of categories (Gaussian components).
        dim : int
            Dimensionality of the feature space.
        temperature : float, optional
            Temperature parameter for the softmax function. Default is 1.0.
        cov_offdiag_damping : float, optional
            Multiplicative damping applied to the off-diagonal entries of the
            Cholesky factors to stabilise learned covariances. Default is 0.1.
        name : str, optional
            Name of the model. Default is "gato_gmm_model".
        """
        super().__init__(name=name)
        self.n_cats = n_cats
        self.dim = dim
        self.temperature = temperature
        self.cov_offdiag_damping = float(cov_offdiag_damping)

        self.mixture_logits = tf.Variable(
            tf.random.normal([n_cats], stddev=0.1),
            trainable=True,
            name="mixture_logits",
        )

        if mean_norm not in {"softmax", "sigmoid"}:
            raise ValueError(
                "mean_norm must be 'softmax' or 'sigmoid'."
                "For sigmoid, you can set a custom range for"
                "each variable with `mean_range`."
            )
        self.mean_norm = mean_norm

        self.mean_range = tuple(mean_range)
        # --- normalise `mean_range` into two 1-D tensors -----------------
        if isinstance(mean_range[0], (list, tuple, np.ndarray)):
            # list[(lo,hi), (lo,hi), ...] per-dimension
            lows, highs = zip(*mean_range)  # length == dim
        else:
            # single (lo,hi) -> broadcast
            lo, hi = mean_range
            lows = [lo] * dim
            highs = [hi] * dim
        self._mean_lo = tf.constant(lows, dtype=tf.float32)  # shape (dim,)
        self._mean_hi = tf.constant(highs, dtype=tf.float32)

        self.means = tf.Variable(
            tf.random.normal([n_cats, dim], stddev=2.0), trainable=True, name="means"
        )

        m = max(dim - 1, 1)
        V_simp = math.sqrt(dim) / math.factorial(dim - 1)
        V_ball = math.pi ** (m / 2) / math.gamma(m / 2 + 1)
        sigma_base = (V_simp / (n_cats * V_ball)) ** (1.0 / m)
        self._sigma_base = tf.constant(sigma_base, dtype=tf.float32)

        init = np.zeros((n_cats, dim, dim), dtype=np.float32)
        self.unconstrained_L = tf.Variable(init, trainable=True, name="unconstrained_L")

    def get_scale_tril(self):
        """
        Compute the lower-triangular scale factors for the covariance matrices.

        Returns
        -------
        tf.Tensor
            A tensor of shape (n_cats, dim, dim) representing the lower-triangular
            scale factors for each Gaussian component.
        """
        L_raw = tf.linalg.band_part(self.unconstrained_L, -1, 0)
        off = L_raw - tf.linalg.diag(tf.linalg.diag_part(L_raw))
        off = self.cov_offdiag_damping * off
        raw_diag = tf.linalg.diag_part(L_raw)
        sigma = self._sigma_base * tf.exp(raw_diag)
        return tf.linalg.set_diag(off, sigma)

    def get_effective_means(self):
        """
        Return the means already mapped into the user's requested space.

        Shape: (n_cats, dim)
        """
        if self.mean_norm == "softmax":
            # softmax normalization: each mean is a point in the simplex
            # we add a zero to the end of each mean vector to make it a simplex
            # and then apply softmax to get the probabilities
            zeros = tf.zeros((self.n_cats, 1), dtype=self.means.dtype)
            full = tf.concat([self.means, zeros], axis=1)  # (k, dim+1)
            probs = tf.nn.softmax(full, axis=1)
            return probs[:, : self.dim]  # simplex coords
        # we apply sigmoid to each mean and then scale it to the user's range
        span = self._mean_hi - self._mean_lo  # (dim,)
        return self._mean_lo + tf.sigmoid(self.means) * span

    def get_mixture_weight(self) -> tf.Tensor:
        """
        Log-space mixture weights   log pi_k  obtained via log-softmax.

        Returns
        -------
        tf.Tensor
            Shape ``(n_cats,)``; ``tf.exp(result)`` sums to 1.
        """
        return tf.nn.log_softmax(self.mixture_logits)

    def get_mixture_pdf(self) -> tfd.Distribution:
        """
        Full Gaussian-mixture distribution for the current parameters.

        Returns
        -------
        tfd.MixtureSameFamily
            Ready to call ``log_prob`` or ``sample``.
        """
        # Component Gaussians
        loc = self.get_effective_means()  # (k, dim)
        scale_tril = self.get_scale_tril()  # (k, dim, dim)
        components = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

        # Mixing distribution using softmax-normalised logits internally
        cat = tfd.Categorical(logits=self.mixture_logits)

        return tfd.MixtureSameFamily(
            mixture_distribution=cat,
            components_distribution=components,
            name="GMM",
        )

    def get_probs(self, data, temperature=None):
        """
        Return the soft assignment matrix ``gamma_ik`` for any input form.

        The method evaluates the log-pdf of every GMM component via the
        model's :py:meth:`get_mixture_pdf` helper—so it automatically uses
        the **current** means, covariances, and softmax-normalised mixture
        weights stored in ``self.mixture_logits``.  The per-component
        log-probabilities ``log p_k(x)`` are combined with the log-weights
        and converted to probabilities through a *temperature-scaled*
        soft-max:

        ``gamma_ik = softmax((log p_k(x_i) + log pi_k) / T)``

        Parameters
        ----------
        data :
            * **tf.Tensor** of shape ``(N, dim)``
            * **np.ndarray** with the same shape
            * **dict** mapping process names to either tensors/arrays **or**\
            nested dicts that contain a key ``"NN_output"``, exactly like\
            the training loop uses.
        temperature : float, optional
            Soft-max temperature *T*.  If *None*, the instance attribute
            ``self.temperature`` is used.  Smaller values make the weights
            approach a hard arg-max; larger values smooth them out.

        Returns
        -------
        Union[tf.Tensor, dict]
            * If *data* is a tensor/array: a tensor of shape ``(N, n_cats)``\
            containing the soft weights for each event.
            * If *data* is a mapping: a dict with the same keys and weight\
            tensors as values.
        """

        T = temperature or self.temperature
        gmm = self.get_mixture_pdf()  # MixtureSameFamily

        # Log-softmax of the logits held inside the mixture distribution
        mix_log_w = tf.nn.log_softmax(
            gmm.mixture_distribution.logits_parameter()
        )  # (k,)

        def _single(x):
            # convert to rank-2 tensor (N, dim)
            x = tf.convert_to_tensor(x, tf.float32)
            if x.shape.rank == 1:
                x = x[..., tf.newaxis]

            # Per-component log-pdfs: (N,1,dim) vs (k,dim) → (N,k)
            lp = gmm.components_distribution.log_prob(x[:, tf.newaxis, :])

            gamma_log = (lp + mix_log_w) / T
            return tf.nn.softmax(gamma_log, axis=-1)  # (N,k)

        if isinstance(data, dict):
            return {
                key: _single(val["NN_output"] if isinstance(val, dict) else val)
                for key, val in data.items()
            }
        return _single(data)

    def get_bin_indices(self, data, temperature: float | None = None):
        """
        Convert input events into *hard* bin indices.

        This is a convenience wrapper that first calls :py:meth:`get_probs` to obtain
        the soft-assignment matrix :math:`\\gamma_{ik}` and then selects the bin
        with the largest probability for each event.

        Parameters
        ----------
        data : Union[tf.Tensor, np.ndarray, Mapping[str, Any]]
            Input data describing one or more event collections.
            * **Tensor / array** - shape ``(N, dim)`` where *N* is the number of
            events and *dim* is the feature dimension.
            * **Mapping** - a dictionary whose values are tensors/arrays **or**
            nested dicts that contain a key ``"NN_output"`` holding the data
            tensor (mimicking the structure used in gato-hep training examples).
        temperature : float, optional
            Temperature factor for the softmax used inside
            :py:meth:`get_probs`.  If *None* (default), the instance attribute
            ``self.temperature`` is used.

        Returns
        -------
        Union[tf.Tensor, Mapping[str, tf.Tensor]]
            Hard bin indices (dtype ``tf.int32``).  The shape is ``(N,)`` when the
            input is a single tensor/array.  If *data* is a mapping, the function
            returns a dictionary with the same keys and ``(N,)`` vectors as values.
        """
        probs = self.get_probs(data, temperature)
        if isinstance(probs, dict):
            return {k: tf.argmax(v, axis=1) for k, v in probs.items()}
        return tf.argmax(probs, axis=1)

    def get_bias(self, data_dict, temperature=None, eps=1e-8):
        """
        Quantify the per-bin bias introduced when the discrete arg-max assignment
        is approximated by a softmax with finite temperature.

        The bias for bin *k* is defined as

        .. math::
            \\text{bias}_k \\;=\\; \\frac{B^{\\text{hard}}_k \\, - \\,
                                    B^{\\text{soft}}_k}
                                    {B^{\\text{hard}}_k}

        where

        * :math:`B^{\\text{hard}}_k` is the sum of event weights that fall into\
        the bin when events are assigned by *argmax*,
        * :math:`B^{\\text{soft}}_k` is the sum of the same weights multiplied by\
        their soft-assignment probability :math:`\\gamma_{ik}`.

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Dictionary of event collections.  Each inner dict must contain\
            the keys ``"NN_output"`` and ``"weight"`` exactly as in the training loop.

        temperature : float or None, optional
            Softmax temperature.  If *None*, the instance attribute
            ``self.temperature`` is used.

        eps : float, optional
            Tiny constant to protect against division by zero. Default is ``1e-8``.

        Returns
        -------
        np.ndarray
            One-dimensional array of length ``n_cats`` with the bias for every bin.
        """

        temp = temperature or self.temperature
        n_cats = self.n_cats

        # ------------------------------------------------------------------
        # 1) Soft assignments and hard argmax assignments
        # ------------------------------------------------------------------
        probs_dict = self.get_probs(data_dict, temperature=temp)  # {proc: (N,k)}
        bins_dict = self.get_bin_indices(data_dict, temperature=temp)  # {proc: (N,)}

        hard_y = tf.zeros(n_cats, dtype=tf.float32)  # Σ w (hard)
        soft_y = tf.zeros(n_cats, dtype=tf.float32)  # Σ γ w (soft)

        for proc, gamma in probs_dict.items():
            w = tf.convert_to_tensor(data_dict[proc]["weight"], tf.float32)
            bins = bins_dict[proc]

            # Hard assignment: sum weights per bin
            hard_y += tf.math.unsorted_segment_sum(w, bins, n_cats)

            # Soft assignment: sum γ·w per bin
            soft_y += tf.reduce_sum(gamma * w[:, None], axis=0)

        bias = (hard_y - soft_y) / tf.maximum(hard_y, eps)
        return bias.numpy()

    def get_differentiable_significance(
        self,
        data_dict,
        *,
        signal_labels: Sequence[str],
        background_reweight: Optional[tf.Tensor | np.ndarray | Sequence[float]] = None,
        reweight_processes: Optional[Sequence[str]] = None,
        return_details: bool = False,
    ):
        """
        Compute differentiable Asimov significances for arbitrary signal sets.

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Input tensors with at least ``"NN_output"`` and ``"weight"`` fields.
        signal_labels : Sequence[str]
            Names of the processes to treat as signal.  The order is preserved
            in the returned mapping.
        background_reweight : array_like, optional
            Per-category scale factors (length = ``n_cats``) applied to the
            accumulated background yield.  Defaults to ``None`` (no scaling).
        reweight_processes : Sequence[str], optional
            Subset of background process names that should receive the
            reweighting factors.  If provided, only these processes are scaled;
            otherwise all background processes are.
        return_details : bool, optional
            If *True*, also return a dictionary with the per-bin yields used
            to compute the significances.

        Returns
        -------
        OrderedDict
            Mapping ``signal_label -> tf.Tensor`` with the differentiable
            significance for each signal.
        tf.Tensor, optional
            Per-bin background yields (only if ``return_details`` is True).
        tf.Tensor, optional
            Per-bin background sum of squared weights (if ``return_details`` is True).
        """

        significances, bkg_yield, bkg_sum_w2 = _compute_significance_common(
            self,
            data_dict,
            signal_labels,
            background_reweight,
            reweight_processes,
        )
        if return_details:
            return significances, bkg_yield, bkg_sum_w2
        return significances

    def call(self, data_dict):
        """
        Placeholder method for computing yields and loss.

        Parameters
        ----------
        data_dict : dict
            A dictionary of input data tensors.

        Raises
        ------
        NotImplementedError
            This method must be overridden in subclasses.
        """
        raise NotImplementedError(
            "Base class: user must override the `call` method to define how yields are "
            "computed, to match the analysis specific needs! Examples are in /examples."
        )

    def get_effective_parameters(self):
        """
        Retrieve the learned mixture weights, means, and covariance factors.

        Returns
        -------
        dict
            A dictionary containing the mixture weights, means, and scale factors.
        """
        return {
            "log mixture weights": tf.math.log_sigmoid(self.mixture_logits)
            .numpy()
            .tolist(),
            "means": self.get_effective_means().numpy().tolist(),
            "scale_tril": self.get_scale_tril().numpy().tolist(),
        }

    def save(self, path: str):
        """
        Save the model's trainable variables to a checkpoint.

        Parameters
        ----------
        path : str
            Directory path to save the checkpoint.
        """
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=3)
        checkpoint_path = manager.save()
        print(f"INFO: model saved to {checkpoint_path}")

    def restore(self, path: str):
        """
        Restore the model's trainable variables from a checkpoint.

        Parameters
        ----------
        path : str
            Directory path to load the checkpoint from.
        """
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"INFO: model restored from {manager.latest_checkpoint}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found under '{path}'. "
                "Make sure the path points to a directory with saved checkpoints."
            )

    def get_effective_boundaries_1d(
        self,
        *,
        n_points: int = 100_000,
        return_mapping: bool = False,
    ):
        """
        Find the 1-D decision boundaries implied by the current GMM.

        The method probes the physical data range, converts those probe
        points to *hard* bin indices via :py:meth:`get_bin_indices`, and records
        where the index changes.

        Parameters
        ----------
        n_points : int, optional
            Number of evenly spaced probe points.  Default is 5 000.
        return_mapping : bool, optional
            If True, also return a permutation that orders categories
            from left to right.

        Returns
        -------
        boundaries : tf.Tensor, shape (n_cats - 1,)
            Boundary locations in the same scale as the input data.
        order : tf.Tensor, shape (n_cats,), optional
            Category permutation, only if *return_mapping* is True.
        """
        # determine the physical range of the discriminant
        lo, hi = self.mean_range if hasattr(self, "mean_range") else (0.0, 1.0)

        # probe the range densely and assign bins
        grid = tf.linspace(lo, hi, n_points)  # (N,)
        grid = tf.reshape(grid, (-1, 1))  # (N,1) expected by get_bin_indices
        hard = self.get_bin_indices(grid)  # (N,)
        flips = tf.where(hard[1:] != hard[:-1])[:, 0]

        # take mid-points between flips as boundaries
        x_left = tf.gather(grid[:, 0], flips)
        x_right = tf.gather(grid[:, 0], flips + 1)
        boundaries = 0.5 * (x_left + x_right)  # (n_cats - 1,)

        if not return_mapping:
            return tf.cast(boundaries, tf.float32)

        # build permutation that sorts categories left-to-right
        first_occ = tf.math.unsorted_segment_min(
            tf.range(n_points), hard, num_segments=self.n_cats
        )
        order = tf.argsort(first_occ)  # (n_cats,)
        return (
            tf.cast(boundaries, tf.float32),
            tf.cast(order, tf.int32),
        )

    def compute_hard_bkg_stats(self, data_dict, signal_labels=None, eps=1e-8):
        """
        Compute per-bin background yields and their relative statistical
        uncertainties, then sort the bins by combined signal
        significance (or in 1D by position).

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Dictionary of event collections.  Each value must contain

            * ``"NN_output"`` - tensor/array with shape ``(N, dim)``.
            * ``"weight"`` - tensor/array with shape ``(N,)``.

        signal_labels : Sequence[str] or None, optional
            Names of the processes that should be treated as *signal*.
            If *None* (default), every key that **starts with** ``"signal"`` is
            considered a signal process.

        eps : float, optional
            Small constant to avoid division by zero when computing
            relative uncertainties.  Default is ``1e-8``.

        Returns
        -------
        B_sorted : np.ndarray
            Background yields per bin, sorted in descending signal significance
            (shape ``(n_cats,)``).

        rel_unc_sorted : np.ndarray
            Relative statistical uncertainties for the same bins
            ``sqrt(sum w^2) / sum w`` (shape ``(n_cats,)``).

        order : np.ndarray
            Indices that map the sorted arrays back to the original bin order
            (dtype ``np.int32``, shape ``(n_cats,)``).
        """

        if signal_labels is None:
            is_signal = lambda name: name.startswith("signal")
        else:
            signal_set = set(signal_labels)
            is_signal = lambda name: name in signal_set

        bins_dict = self.get_bin_indices(data_dict)  # {proc: (N,)}

        n_cats = self.n_cats
        B = tf.zeros(n_cats, dtype=tf.float32)  # Σ w   (background)
        B2 = tf.zeros(n_cats, dtype=tf.float32)  # Σ w²  (background)
        S = tf.zeros(n_cats, dtype=tf.float32)  # Σ w   (combined signal)

        for proc, bins in bins_dict.items():
            w = tf.convert_to_tensor(data_dict[proc]["weight"], tf.float32)

            # accumulate sums per bin on the device
            w_sum = tf.math.unsorted_segment_sum(w, bins, n_cats)
            if is_signal(proc):
                S += w_sum
            else:
                B += w_sum
                B2 += tf.math.unsorted_segment_sum(tf.square(w), bins, n_cats)

        sigma = tf.sqrt(B2)
        rel_unc = sigma / tf.maximum(B, eps)

        # Determine bin order
        if self.dim == 1:
            # 1-D: keep the natural left-to-right ordering
            _, order = self.get_effective_boundaries_1d(return_mapping=True)
            order = np.asarray(order, dtype=np.int32)
        else:
            # multi-D: sort by Z = S / √B
            Zbins = asymptotic_significance(S, B)
            order = tf.argsort(Zbins, direction="DESCENDING").numpy().astype(np.int32)

        B_sorted = tf.gather(B, order).numpy()
        rel_unc_sorted = tf.gather(rel_unc, order).numpy()

        return B_sorted, rel_unc_sorted, order


class gato_sigmoid_model(tf.Module):
    """
    Gato model for optimisation of cuts based on sigmoid-approximated
    boundaries. Can be applied to multiple discriminants, each with its own
    number of bins and steepness.

    Each discriminant j is split into n_j bins by (n_j - 1) *trainable*
    cut points b_{j,i}.  The full event bin is the Cartesian product of
    the one-dimensional bins.

    Parameters
    ----------
    variables_config : list of dict
        One entry per discriminant, for example::

            [
                {"name": "disc", "bins": 3, "range": (0.0, 1.0)},
                {"name": "mjj",  "bins": 2, "range": (200.0, 3000.0)},
            ]

        Keys
        ----
        bins : int
            Number of bins (> 1) for this variable.
        range : tuple(float, float)
            Inclusive lower and upper bound of the variable.
        name : str, optional
            Plain-text label used only in logs.
        steepness : float, optional
            Individual initial slope k_j.  If omitted the global
            ``global_steepness`` is used.
    global_steepness : float, optional
        Default initial steepness k for variables that do not override it.
        Can be annealed during training.
    name : str, optional
        TensorFlow name scope.

    Public helpers
    --------------
    get_probs(x)
        Soft assignment gamma_ik (shape N x n_cats).
    get_bin_indices(x)
        Hard bin index per event (shape N,).
    get_bias(data)
        Per-bin bias (hard minus soft) divided by hard yield.
    """

    def __init__(
        self,
        variables_config: List[Dict],
        *,
        global_steepness: float = 5.0,
        name: str = "gato_sigmoid_model",
    ):
        super().__init__(name=name)

        self.var_cfg = []
        self.n_cats = 1

        for idx, cfg in enumerate(variables_config):
            bins = int(cfg["bins"])
            if bins < 2:
                raise ValueError("bins must be at least 2")

            lo, hi = cfg.get("range", (0.0, 1.0))
            k_val = cfg.get("steepness", global_steepness)

            entry = dict(
                name=cfg.get("name", f"var{idx}"),
                bins=bins,
                lo=float(lo),
                hi=float(hi),
                span=float(hi) - float(lo),
                raw=tf.Variable(
                    tf.random.uniform((bins - 1,), -2.0, 2.0),
                    name=f"raw_{idx}",
                ),
                k=tf.Variable(float(k_val), dtype=tf.float32, name=f"k_{idx}"),
            )
            self.var_cfg.append(entry)
            self.n_cats *= bins

    def _calculate_boundaries(self, raw_boundaries: tf.Tensor) -> tf.Tensor:
        """
        Transform unconstrained values into an ordered set in (0, 1).

        Parameters
        ----------
        raw_boundaries : tf.Tensor, shape (m,)
            Trainable raw logits (m = n_bins - 1).

        Returns
        -------
        tf.Tensor, shape (m,)
            Sorted boundaries strictly between 0 and 1.
        """
        # Pad one extra zero so softmax has length m + 1 and sums to 1.
        pad = tf.zeros_like(raw_boundaries[:1])
        logits = tf.concat([raw_boundaries, pad], axis=0)  # (m + 1,)
        increments = tf.nn.softmax(logits)  # (m + 1,)
        boundaries = tf.cumsum(increments)[:-1]  # keep m values
        return boundaries

    def calculate_boundaries(self, j: int = 0) -> tf.Tensor:
        """
        Return the ordered physical boundaries for variable *j*.

        Parameters
        ----------
        j : int, optional
            Index of the discriminant whose boundaries are requested.

        Returns
        -------
        tf.Tensor
            Tensor of shape ``(n_bins - 1,)`` with boundary locations expressed
            in the variable's original scale.
        """
        cfg = self.var_cfg[j]
        in_unit = self._calculate_boundaries(cfg["raw"])
        return cfg["lo"] + in_unit * cfg["span"]

    def get_probs(self, data, *, steepness_scale: float | None = None):
        """
        Soft weights gamma_ik for arbitrary input structure.

        Accepts the same tensor / dict shapes used in the GMM example.
        """

        def _single(x):
            if not tf.is_tensor(x):
                x = tf.convert_to_tensor(x, tf.float32)
            if x.shape.rank == 1:
                x = tf.expand_dims(x, -1)  # (N, 1)

            weights = []
            for j, cfg in enumerate(self.var_cfg):
                boundaries = tf.expand_dims(self.calculate_boundaries(j), 0)  # (1, m_j)
                k = cfg["k"] * (steepness_scale or 1.0)
                xj = tf.expand_dims(x[:, j], 1)  # (N, 1)

                # sig contains "probabilities that the event is right of the boundary"
                sig = safe_sigmoid(xj - boundaries, k)  # (N, m_j)
                # everything left of the first cut is in the first bin
                left = 1.0 - sig[:, :1]
                middle = sig[:, :-1] - sig[:, 1:]  # weights in all interior bins
                right = sig[:, -1:]  # weight in last (rightmost) bin
                wj = tf.concat([left, middle, right], axis=1)  # (N, n_j)
                weights.append(wj)

            w_full = weights[0]
            for wj in weights[1:]:
                w_full = tf.einsum("ni,nj->nij", w_full, wj)
                w_full = tf.reshape(w_full, (tf.shape(x)[0], -1))
            return w_full  # (N, n_cats)

        if isinstance(data, dict):
            return {
                k: _single(v["NN_output"] if isinstance(v, dict) else v)
                for k, v in data.items()
            }
        return _single(data)

    def get_bin_indices(self, data, *, steepness_scale: float | None = None):
        """
        Convert input data into hard bin indices.

        Parameters
        ----------
        data : Union[tf.Tensor, np.ndarray, Mapping[str, Any]]
            Input events as a tensor/array of shape ``(N, n_disc)`` or a mapping
            that mirrors the training loop structure with ``"NN_output"`` keys.
        steepness_scale : float, optional
            Multiplicative factor applied to every sigmoid steepness.  ``None``
            (default) keeps the model's stored values.

        Returns
        -------
        Union[tf.Tensor, dict]
            Hard bin indices with dtype ``tf.int32``.  When *data* is a mapping,
            the result is a dict with the same keys and ``(N,)`` tensors.  Otherwise
            a single ``(N,)`` tensor is returned.
        """
        probs = self.get_probs(data, steepness_scale=steepness_scale)
        if isinstance(probs, dict):
            return {
                k: tf.argmax(v, axis=1, output_type=tf.int32) for k, v in probs.items()
            }
        return tf.argmax(probs, axis=1, output_type=tf.int32)

    def get_bias(self, data_dict, *, steepness_scale: float | None = None, eps=1e-8):
        """
        Estimate the per-bin bias from using soft assignments.

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Event collections with ``"NN_output"`` and ``"weight"`` entries,
            identical to what :meth:`get_probs` expects.
        steepness_scale : float, optional
            Additional factor applied to every steepness value before computing
            probabilities.  Defaults to ``None`` for no rescaling.
        eps : float, optional
            Small positive constant that guards against division by zero.

        Returns
        -------
        np.ndarray
            One-dimensional array of length ``n_cats`` containing
            ``(hard - soft) / hard`` for each bin.
        """
        soft = self.get_probs(data_dict, steepness_scale=steepness_scale)
        hard = self.get_bin_indices(data_dict, steepness_scale=steepness_scale)

        hard_y = tf.zeros(self.n_cats, tf.float32)
        soft_y = tf.zeros(self.n_cats, tf.float32)

        for proc, w_soft in soft.items():
            w = tf.convert_to_tensor(data_dict[proc]["weight"], tf.float32)
            hard_y += tf.math.unsorted_segment_sum(w, hard[proc], self.n_cats)
            soft_y += tf.reduce_sum(w_soft * w[:, None], axis=0)

        return ((hard_y - soft_y) / tf.maximum(hard_y, eps)).numpy()

    def get_differentiable_significance(
        self,
        data_dict,
        *,
        signal_labels: Sequence[str],
        background_reweight: Optional[tf.Tensor | np.ndarray | Sequence[float]] = None,
        reweight_processes: Optional[Sequence[str]] = None,
        return_details: bool = False,
    ):
        """
        Compute differentiable Asimov significances for sigmoid-based models.

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Input tensors with at least ``"NN_output"`` and ``"weight"`` fields.
        signal_labels : Sequence[str]
            Names of the processes treated as signal.
        background_reweight : array_like, optional
            Per-category scale factors applied to the accumulated background
            yield (length = ``n_cats``).  ``None`` disables reweighting.
        reweight_processes : Sequence[str], optional
            Background process names that should be scaled by the provided
            factors.  If omitted, all background processes receive the
            reweighting.
        return_details : bool, optional
            If *True*, also return the per-bin yield tensors used internally.

        Returns
        -------
        OrderedDict
            Map from signal label to differentiable significance.
        tf.Tensor, optional
            Background yield per bin (if ``return_details`` is True).
        tf.Tensor, optional
            Background sum of squared weights per bin (if ``return_details`` is True).
        """

        significances, bkg_yield, bkg_sum_w2 = _compute_significance_common(
            self,
            data_dict,
            signal_labels,
            background_reweight,
            reweight_processes,
        )
        if return_details:
            return significances, bkg_yield, bkg_sum_w2
        return significances

    def save(self, path: str):
        """
        Save the model's trainable variables to a checkpoint.

        Parameters
        ----------
        path : str
            Directory path to save the checkpoint.
        """
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=3)
        checkpoint_path = manager.save()
        print(f"INFO: model saved to {checkpoint_path}")

    def restore(self, path: str):
        """
        Restore the model's trainable variables from a checkpoint.

        Parameters
        ----------
        path : str
            Directory path to load the checkpoint from.
        """
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"INFO: model restored from {manager.latest_checkpoint}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found under '{path}'. "
                "Make sure the path points to a directory with saved checkpoints."
            )

    def compute_hard_bkg_stats(self, data_dict, signal_labels=None, eps=1e-8):
        """
        Compute per-bin background yields and their relative statistical
        uncertainties.

        Parameters
        ----------
        data_dict : Mapping[str, dict]
            Dictionary of event collections.  Each value must contain

            * ``"NN_output"`` - tensor/array with shape ``(N, dim)``.
            * ``"weight"`` - tensor/array with shape ``(N,)``.

        signal_labels : Sequence[str] or None, optional
            Names of the processes that should be treated as *signal*.
            If *None* (default), every key that **starts with** ``"signal"`` is
            considered a signal process.

        eps : float, optional
            Small constant to avoid division by zero when computing
            relative uncertainties.  Default is ``1e-8``.

        Returns
        -------
        B_sorted : np.ndarray
            Background yields per bin
            (shape ``(n_cats,)``).

        rel_unc_sorted : np.ndarray
            Relative statistical uncertainties for the same bins
            ``sqrt(sum w^2) / sum w`` (shape ``(n_cats,)``).
        """

        if signal_labels is None:
            is_signal = lambda name: name.startswith("signal")
        else:
            signal_set = set(signal_labels)
            is_signal = lambda name: name in signal_set

        bins_dict = self.get_bin_indices(data_dict)  # {proc: (N,)}

        n_cats = self.n_cats
        B = tf.zeros(n_cats, dtype=tf.float32)  # Σ w   (background)
        B2 = tf.zeros(n_cats, dtype=tf.float32)  # Σ w²  (background)
        S = tf.zeros(n_cats, dtype=tf.float32)  # Σ w   (combined signal)

        for proc, bins in bins_dict.items():
            w = tf.convert_to_tensor(data_dict[proc]["weight"], tf.float32)

            # accumulate sums per bin on the device
            w_sum = tf.math.unsorted_segment_sum(w, bins, n_cats)
            if is_signal(proc):
                S += w_sum
            else:
                B += w_sum
                B2 += tf.math.unsorted_segment_sum(tf.square(w), bins, n_cats)

        sigma = tf.sqrt(B2)
        rel_unc = sigma / tf.maximum(B, eps)

        return B, rel_unc


def _compute_significance_common(
    model,
    data_dict,
    signal_labels,
    background_reweight,
    reweight_processes,
):
    if not signal_labels:
        raise ValueError("signal_labels must contain at least one entry.")

    probs_inputs = {
        proc: {"NN_output": entry["NN_output"]}
        if isinstance(entry, dict) and "NN_output" in entry
        else entry
        for proc, entry in data_dict.items()
    }
    probs = model.get_probs(probs_inputs)

    n_cats = model.n_cats
    dtype = tf.float32
    signal_labels = list(signal_labels)
    signal_yields = {
        label: tf.zeros(n_cats, dtype) for label in signal_labels
    }

    apply_reweight = background_reweight is not None
    if apply_reweight:
        factors = tf.reshape(
            tf.convert_to_tensor(background_reweight, dtype), (n_cats,)
        )
    else:
        factors = tf.ones(n_cats, dtype)

    if reweight_processes is not None:
        reweight_set = set(reweight_processes)
        if not reweight_set and apply_reweight:
            apply_reweight = False
    else:
        reweight_set = None

    bkg_static = tf.zeros(n_cats, dtype)
    bkg_static_w2 = tf.zeros(n_cats, dtype)
    bkg_rew = tf.zeros(n_cats, dtype)
    bkg_rew_w2 = tf.zeros(n_cats, dtype)

    signal_set = set(signal_labels)

    for proc, gamma in probs.items():
        weights = tf.convert_to_tensor(
            data_dict[proc]["weight"], dtype
        )
        yields = tf.reduce_sum(gamma * weights[:, None], axis=0)
        if proc in signal_set:
            signal_yields[proc] += yields
            continue

        sumw2 = tf.reduce_sum(gamma * (weights**2)[:, None], axis=0)
        if apply_reweight and (reweight_set is None or proc in reweight_set):
            bkg_rew += yields
            bkg_rew_w2 += sumw2
        else:
            bkg_static += yields
            bkg_static_w2 += sumw2

    background_yield = (
        bkg_static + (bkg_rew * factors if apply_reweight else bkg_rew)
    )
    background_sumw2 = (
        bkg_static_w2 + (bkg_rew_w2 * factors if apply_reweight else bkg_rew_w2)
    )

    significances = OrderedDict()
    for label in signal_labels:
        others = tf.zeros(n_cats, dtype)
        for other in signal_labels:
            if other == label:
                continue
            others += signal_yields[other]
        total_bkg = background_yield + others
        z_bins = asymptotic_significance(signal_yields[label], total_bkg)
        significances[label] = tf.sqrt(tf.reduce_sum(z_bins**2))

    return significances, background_yield, background_sumw2
