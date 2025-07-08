import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm

from gatohep.utils import asymptotic_significance, safe_sigmoid

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
        name : str, optional
            Name of the model. Default is "gato_gmm_model".
        """
        super().__init__(name=name)
        self.n_cats = n_cats
        self.dim = dim
        self.temperature = temperature

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
        off = 0.1 * off
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

    def get_probs(self, data, temperature: float | None = None):
        """
        Soft assignment gamma_ik  for arbitrary inputs.

        Parameters
        ----------
        data :  • tf.Tensor  shape (N, dim)
                • np.ndarray
                • dict  {name: {"NN_output": tensor|array, ...}}  (as in examples)
        Returns
        -------
        tf.Tensor or dict with shape (N, n_cats)
        """
        T = temperature or self.temperature
        loc = self.get_effective_means()
        scale_tril = self.get_scale_tril()
        log_mix = tf.nn.log_softmax(self.mixture_logits)

        def _single(x):
            """
            Return probs_ik for a single tensor/array ``x`` of shape (N, dim).
            * Ensures rank-2 input even for one-dimensional features.
            * Vectorises the log-pdf evaluation across all Gaussian components.
            """
            # Convert to tensor and make sure it is (N, dim)
            if not tf.is_tensor(x):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            if tf.rank(x) == 1:  # (N,)  →  (N, 1)
                x = tf.expand_dims(x, -1)

            # Broadcast against batch of Gaussians: (N, 1, dim) ↔ (k, dim)
            x = tf.expand_dims(x, 1)  # (N, 1, dim)

            mvn = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
            lp = mvn.log_prob(x)  # (N, k) – dynamic shape

            logits = (lp + log_mix) / T
            logits = tf.reshape(
                logits, [-1, self.n_cats]
            )  # <<< NEW – rank now known (N, k)

            return tf.nn.softmax(logits, axis=-1)  # axis=-1 is safely 1

        # flexible input handling
        if isinstance(data, dict):
            return {
                k: _single(v["NN_output"] if isinstance(v, dict) else v)
                for k, v in data.items()
            }
        return _single(data)

    def get_bin(self, data, temperature: float | None = None):
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
        gamma = self.get_probs(data, temperature)
        if isinstance(gamma, dict):
            return {k: tf.argmax(v, axis=1) for k, v in gamma.items()}
        return tf.argmax(gamma, axis=1)

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
        bins_dict = self.get_bin(data_dict, temperature=temp)  # {proc: (N,)}

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
            "mixture_weights": tf.nn.softmax(self.mixture_logits).numpy().tolist(),
            "means": self.means.numpy().tolist(),
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
            print(f"INFO: no checkpoint found in {path}, starting fresh")

    def get_effective_boundaries_1d(self, n_steps=100_000, return_mapping=False):
        """
        Numerically find the crossing points in [0,1] where the most likely
        Gaussian component switches.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps for the numerical grid. Default is 100000.
        return_mapping : bool, optional
            Whether to return the component ordering. Default is False.

        Returns
        -------
        list
            A list of crossing points in [0,1].
        list, optional
            If `return_mapping` is True, also returns the component ordering.
        """
        # 1) fetch & activate parameters
        weight = tf.nn.softmax(self.mixture_logits).numpy()  # (n_cats,)
        mu_raw = np.array(self.means)[:, 0]  # raw
        mu = tf.math.sigmoid(mu_raw).numpy()  # in [0,1]
        sig = self.get_scale_tril().numpy()[:, 0, 0]  # (n_cats,)

        # 2) build grid & evaluate weighted pdfs
        xgrid = np.linspace(0.0, 1.0, n_steps)
        pdfs = np.vstack(
            [w * norm.pdf(xgrid, loc=m, scale=s) for w, m, s in zip(weight, mu, sig)]
        ).T  # shape = (n_steps, n_cats)

        # 3) who wins where?
        winners = np.argmax(pdfs, axis=1)  # for each x_i

        # 4) compute avg x-position for each component
        avg_x = np.full(self.n_cats, 1, dtype=float)
        for k in range(self.n_cats):
            mask = winners == k
            if np.any(mask):
                avg_x[k] = xgrid[mask].mean()

        # 5) order components by avg_x
        order = np.argsort(avg_x)  # gives original -> sorted mapping

        # 6) now re-index winners into sorted order (“bin 0” is leftmost, bin 1, ...)
        inv_order = np.argsort(order)  # maps old index -> new index
        winners_sorted = inv_order[winners]

        # 7) find the first n_cats-1 transitions
        jumps = np.where(winners_sorted[:-1] != winners_sorted[1:])[0]
        cuts = []
        for j in jumps:
            cuts.append(0.5 * (xgrid[j] + xgrid[j + 1]))
            if len(cuts) >= self.n_cats - 1:
                break

        cuts.sort()
        if return_mapping:
            return cuts, order.tolist()
        return cuts

    def get_boundaries_for_history(self, n_steps=100_000):
        """
        Compute the boundaries between Gaussian components for each epoch.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps for the numerical grid. Default is 100000.

        Returns
        -------
        list
            A list of boundary positions in [0,1].
        """
        # activated parameters -------------------------------------------------
        w = tf.nn.softmax(self.mixture_logits).numpy()  # (n_cats,)
        mu = tf.math.sigmoid(self.means[:, 0]).numpy()  # (n_cats,)
        sig = self.get_scale_tril().numpy()[:, 0, 0]  # (n_cats,)

        # sort by mu to get a *stable* left-to-right ordering
        order = np.argsort(mu)
        w, mu, sig = w[order], mu[order], sig[order]

        # dense grid + winning component at each x -----------------------------
        x = np.linspace(0.0, 1.0, n_steps)
        pdf = np.vstack(
            [w_i * norm.pdf(x, mu_i, s_i) for w_i, mu_i, s_i in zip(w, mu, sig)]
        ).T
        winner = np.argmax(pdf, axis=1)  # (n_steps,)

        # where does the winner change?  → boundaries
        jumps = np.where(winner[:-1] != winner[1:])[0]
        cuts = 0.5 * (x[jumps] + x[jumps + 1])  # (<= n_cats-1,)

        # put them into fixed slots (NaN where missing)
        slots = np.full(self.n_cats - 1, np.nan, dtype=float)
        for idx, cut in zip(winner[jumps], cuts):
            # idx = the left component of the pair (after sorting by µ)
            if idx < self.n_cats - 1:
                slots[idx] = cut
        return slots.tolist()  # length = n_cats-1

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

        bins_dict = self.get_bin(data_dict)  # {proc: (N,)}

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
    A generic model that uses sigmoid functions with trainable boundaries.

    The model splits events by "soft membership" in each bin, accumulates yields,
    and computes a figure of merit (e.g., significance).

    Attributes
    ----------
    variables_config : list of dict
        Configuration for each variable, including name, number of categories,
        and optional steepness for soft transitions.
    raw_boundaries_list : list of tf.Variable
        Trainable raw boundaries for each variable.
    """

    def __init__(self, variables_config, name="DifferentiableCutModel", **kwargs):
        """
        Initialize the sigmoid-based model.

        Parameters
        ----------
        variables_config : list of dict
            Configuration for each variable, including name, number of categories,
            and optional steepness for soft transitions.
        name : str, optional
            Name of the model. Default is "DifferentiableCutModel".
        """
        super().__init__(name=name)
        self.variables_config = variables_config
        self.raw_boundaries_list = []
        for var_cfg in variables_config:
            n_cats = var_cfg["n_cats"]
            shape = (n_cats,) if n_cats > 1 else (0,)
            init_tensor = tf.random.normal(shape=shape, stddev=0.3)
            raw_var = tf.Variable(
                init_tensor, trainable=True, name=f"raw_bndry_{var_cfg['name']}"
            )
            self.raw_boundaries_list.append(raw_var)

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
            "Base class: user must override the `call` method to define how yields are"
            " computed to match the analysis specific needs! Examples are in /examples."
        )

    def get_effective_boundaries(self):
        """
        Retrieve the effective boundaries for each variable.

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are lists of
            sorted boundaries in [0,1].
        """
        out = {}
        for var_cfg, raw_var in zip(self.variables_config, self.raw_boundaries_list):
            if raw_var.shape[0] == 0:
                out[var_cfg["name"]] = []
            else:
                out[var_cfg["name"]] = calculate_boundaries(raw_var)
        return out


def soft_bin_weights(x, raw_boundaries, steepness=50.0):
    """
    Compute soft membership weights for each bin.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape [n_samples].
    raw_boundaries : tf.Tensor
        Trainable raw boundaries for the bins.
    steepness : float, optional
        Steepness of the sigmoid transitions. Default is 50.0.

    Returns
    -------
    list of tf.Tensor
        A list of tensors representing membership weights for each bin.
    """
    if raw_boundaries.shape[0] == 0:
        return [tf.ones_like(x, dtype=tf.float32)]

    boundaries = calculate_boundaries(raw_boundaries)
    n_cats = len(boundaries) + 1
    w_list = []
    for i in range(n_cats):
        if i == 0:
            w0 = 1.0 - safe_sigmoid(x - boundaries[0], steepness)
            w_list.append(w0)
        elif i == (n_cats - 1):
            wN = safe_sigmoid(x - boundaries[-1], steepness)
            w_list.append(wN)
        else:
            w_i = safe_sigmoid(x - boundaries[i - 1], steepness) - safe_sigmoid(
                x - boundaries[i], steepness
            )
            w_list.append(w_i)
    return w_list


def calculate_boundaries(raw_boundaries):
    """
    Transform raw boundaries into an ordered set in (0,1).

    Parameters
    ----------
    raw_boundaries : tf.Tensor
        Trainable raw boundaries.

    Returns
    -------
    tf.Tensor
        A tensor of sorted boundaries in (0,1).
    """
    increments = tf.nn.softmax(raw_boundaries)
    boundaries = tf.cumsum(increments)
    return boundaries[:-1]
