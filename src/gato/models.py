from gato.utils import safe_sigmoid, asymptotic_significance
import math
import numpy as np
from scipy.special import expit
from scipy.stats import norm
import tensorflow as tf
import tensorflow_probability as tfp
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
    mixture_logits : tf.Variable
        Trainable logits for the mixture weights.
    means : tf.Variable
        Trainable mean vectors for each Gaussian component.
    unconstrained_L : tf.Variable
        Trainable unconstrained lower-triangular matrices for covariance factors.
    """

    def __init__(self, n_cats, dim, temperature=1.0, name="gato_gmm_model"):
        """
        Initialize the Gaussian mixture model.

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
            trainable=True, name="mixture_logits"
        )
        self.means = tf.Variable(
            tf.random.normal([n_cats, dim], stddev=2.0),
            trainable=True, name="means"
        )

        m = max(dim - 1, 1)
        V_simp = math.sqrt(dim) / math.factorial(dim - 1)
        V_ball = math.pi ** (m / 2) / math.gamma(m / 2 + 1)
        sigma_base = (V_simp / (n_cats * V_ball)) ** (1.0 / m)
        self._sigma_base = tf.constant(sigma_base, dtype=tf.float32)

        init = np.zeros((n_cats, dim, dim), dtype=np.float32)
        self.unconstrained_L = tf.Variable(
            init, trainable=True, name="unconstrained_L"
        )

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
            "scale_tril": self.get_scale_tril().numpy().tolist()
        }

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
        weight = tf.nn.softmax(self.mixture_logits).numpy()      # (n_cats,)
        mu_raw = np.array(self.means)[:, 0]                      # raw
        mu = tf.math.sigmoid(mu_raw).numpy()                # in [0,1]
        sig = self.get_scale_tril().numpy()[:,0,0]           # (n_cats,)

        # 2) build grid & evaluate weighted pdfs
        xgrid = np.linspace(0.0, 1.0, n_steps)
        pdfs = np.vstack([
            w * norm.pdf(xgrid, loc=m, scale=s)
            for w,m,s in zip(weight, mu, sig)
        ]).T  # shape = (n_steps, n_cats)

        # 3) who wins where?
        winners = np.argmax(pdfs, axis=1)    # for each x_i

        # 4) compute avg x-position for each component
        avg_x = np.full(self.n_cats, 1, dtype=float)
        for k in range(self.n_cats):
            mask = (winners == k)
            if np.any(mask):
                avg_x[k] = xgrid[mask].mean()

        # 5) order components by avg_x
        order = np.argsort(avg_x)   # gives original -> sorted mapping

        # 6) now re-index winners into sorted order (“bin 0” is leftmost, bin 1, ...)
        inv_order = np.argsort(order)  # maps old index -> new index
        winners_sorted = inv_order[winners]

        # 7) find the first n_cats-1 transitions
        jumps = np.where(winners_sorted[:-1] != winners_sorted[1:])[0]
        cuts = []
        for j in jumps:
            cuts.append(0.5*(xgrid[j] + xgrid[j+1]))
            if len(cuts) >= self.n_cats - 1:
                break

        cuts.sort()
        if return_mapping:
            return cuts, order.tolist()
        return cuts

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
        w = tf.nn.softmax(self.mixture_logits).numpy()           # (n_cats,)
        mu = tf.math.sigmoid(self.means[:, 0]).numpy()            # (n_cats,)
        sig = self.get_scale_tril().numpy()[:, 0, 0]               # (n_cats,)

        # sort by mu to get a *stable* left-to-right ordering
        order = np.argsort(mu)
        w, mu, sig = w[order], mu[order], sig[order]

        # dense grid + winning component at each x -----------------------------
        x = np.linspace(0.0, 1.0, n_steps)
        pdf = np.vstack([w_i * norm.pdf(x, mu_i, s_i)
                        for w_i, mu_i, s_i in zip(w, mu, sig)]).T
        winner = np.argmax(pdf, axis=1)                     # (n_steps,)

        # where does the winner change?  → boundaries
        jumps = np.where(winner[:-1] != winner[1:])[0]
        cuts = 0.5 * (x[jumps] + x[jumps + 1])            # (<= n_cats-1,)

        # put them into fixed slots (NaN where missing)
        slots = np.full(self.n_cats - 1, np.nan, dtype=float)
        for idx, cut in zip(winner[jumps], cuts):
            # idx = the left component of the pair (after sorting by µ)
            if idx < self.n_cats - 1:
                slots[idx] = cut
        return slots.tolist()          # length = n_cats-1

    def compute_hard_bkg_stats(self, data_dict, low=0.0, high=1.0, eps=1e-8):
        """
        Compute per-bin background yields and relative uncertainties.

        Parameters
        ----------
        data_dict : dict
            A dictionary of input data tensors.
        low : float, optional
            Lower bound for the feature space. Default is 0.0.
        high : float, optional
            Upper bound for the feature space. Default is 1.0.
        eps : float, optional
            Small value to avoid division by zero. Default is 1e-8.

        Returns
        -------
        tuple
            A tuple containing sorted background yields, relative uncertainties,
            and the component ordering.
        """
        raw_logits = self.mixture_logits.numpy()                     # (n_cats,)
        log_mix = tf.nn.log_softmax(raw_logits).numpy()           # (n_cats,)
        raw_means = self.means.numpy()                              # (n_cats,)
        if self.dim == 1:
            # map raw means into [0,1] with sigmoid
            mu_act = expit(raw_means.flatten())
        else:
            # map raw means into valid vectors via softmax across dims for each bin
            mu_act = tf.nn.softmax(raw_means, axis=1).numpy()
        # covariance
        scales = self.get_scale_tril().numpy()

        n_cats = self.n_cats

        # collect all background events
        X_parts, W_parts = [], []
        for proc,t in data_dict.items():
            if proc.startswith("signal"):
                continue
            arr = np.asarray(t["NN_output"]).reshape(-1, self.dim)
            X_parts.append(arr)
            W_parts.append(np.asarray(t["weight"]))
        X = np.concatenate(X_parts, axis=0)   # (N,1)
        W = np.concatenate(W_parts)          # (N,)

        # compute the exact same log-joint as in training
        logj = np.zeros((len(W), n_cats))
        for i in range(n_cats):
            dist = tfd.MultivariateNormalTriL(loc=mu_act[i], scale_tril=scales[i])
            logj[:,i] = dist.log_prob(X).numpy() + log_mix[i]

        # hard-assign to the winning component
        comp = np.argmax(logj, axis=1)

        # accumulate B and B2
        B = np.zeros(n_cats)
        B2 = np.zeros(n_cats)
        for i in range(n_cats):
            sel = (comp == i)
            if not sel.any():
                continue
            w_i = W[sel]
            B[i] = w_i.sum()
            B2[i] = (w_i**2).sum()

        # compute rel-uncertainty
        sigma = np.sqrt(np.maximum(B2,0.0))
        rel_unc = sigma/np.maximum(B,eps)

        # determine order
        if self.dim == 1:
            _, order = self.get_effective_boundaries_1d(return_mapping=True)
        else:
            # multi-D: sort by per-bin Z
            S_dummy = tf.zeros_like(tf.constant(B, dtype=tf.float32))
            Zbins = asymptotic_significance(
                S_dummy,
                tf.constant(B, dtype=tf.float32)
            ).numpy()
            order = np.argsort(-Zbins)

        return B[order], rel_unc[order], order


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
            shape = (n_cats, ) if n_cats > 1 else (0, )
            init_tensor = tf.random.normal(shape=shape, stddev=0.3)
            raw_var = tf.Variable(
                init_tensor,
                trainable=True,
                name=f"raw_bndry_{var_cfg['name']}"
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
            w_i = safe_sigmoid(x - boundaries[i - 1], steepness) - \
                safe_sigmoid(x - boundaries[i], steepness)
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
