import tensorflow as tf
import numpy as np
from scipy.stats import norm
from scipy.special import expit
import tensorflow_probability as tfp
tfd = tfp.distributions
import math


class gato_gmm_model(tf.Module):
    """
    A differentiable category model based on a Gaussian mixture.

    The model learns, for each of n_cats:
      - mixture_logits (which give the mixing weights),
      - mean vector (of dimension 'dim'),
      - an unconstrained lower-triangular matrix that is transformed into a 
        positive-definite Cholesky factor for the covariance.

    The per-event soft membership is computed by evaluating the log pdf of each
    Gaussian at the event's feature vector and adding the log mixture weight.
    A temperatured softmax is then applied.

    The call() method loops over processes in data_tensor (a dict of tf tensors
    with columns "NN_output" (an array-like of shape [dim]) and "weight"), computes 
    soft memberships, accumulates yields, and returns the negative overall significance
    as loss (plus background yields).
    """

    def __init__(self, n_cats, dim, temperature=1.0, name="gato_gmm_model"):
        super().__init__(
            name=name
        )
        self.n_cats      = n_cats
        self.dim         = dim
        self.temperature = temperature

        # -- mixture logits and means as before --
        self.mixture_logits = tf.Variable(
            tf.random.normal([n_cats], stddev=0.1),
            trainable=True, name="mixture_logits"
        )
        self.means = tf.Variable(
            tf.random.normal([n_cats, dim], stddev=2.0),
            trainable=True, name="means"
        )

        # --------------------------------------------------------------------
        # Intrinsic manifold dimension (softmax on dim coordinates -> simplex of dim-1)
        m = max(dim - 1, 1)

        # Volume of the (m-simplex) { x_i>=0, sum x_i=1 } under the induced Euclid metric: V_simplex = sqrt(dim) / ( (dim-1)! )
        V_simp = math.sqrt(dim) / math.factorial(dim - 1)

        # Volume of the unit m-ball: V_ball = π^(m/2) / Γ(m/2 + 1)
        V_ball = math.pi ** (m / 2) / math.gamma(m / 2 + 1)

        # Choose sigma_base so that K · (V_ball · sigma^m) = V_simp
        sigma_base = (V_simp / (n_cats * V_ball)) ** (1.0 / m)
        self._sigma_base = tf.constant(sigma_base, dtype=tf.float32)

        # Initialize unconstrained_L = zeros (raw_diag=0, offdiag=0)
        init = np.zeros((n_cats, dim, dim), dtype=np.float32)
        self.unconstrained_L = tf.Variable(
            init, trainable=True, name="unconstrained_L"
        )

    def get_scale_tril(self):
        """
        Return a (n_cats, dim, dim) tensor of lower-triangular scale-factors L_k
        so that Sigma_k = L_k @ L_k^T:

        -> diag(L_k) = sigma_base * exp(raw_diag)
        -> off-diag(L_k) = 0.1 * raw_offdiag
        """
        # isolate lower triangle
        L_raw = tf.linalg.band_part(self.unconstrained_L, -1, 0)

        off = L_raw - tf.linalg.diag(tf.linalg.diag_part(L_raw))
        # damp off-diagonals
        off = 0.1 * off

        # build positive diag via exp
        raw_diag = tf.linalg.diag_part(L_raw)           # shape = (n_cats, dim)
        sigma    = self._sigma_base * tf.exp(raw_diag)  # same shape

        # reassemble
        return tf.linalg.set_diag(off, sigma)

    def call(self, data_dict):
        raise NotImplementedError(
            "Base class: user must override the `call` method to define how yields are computed, to match the analysis specific needs! Examples are in /examples."
        )

    def get_effective_parameters(self):
        """
        Returns the learned mixture weights, means, and covariance factors.
        """
        return {
            "mixture_weights": tf.nn.softmax(self.mixture_logits).numpy().tolist(),
            "means": self.means.numpy().tolist(),
            "scale_tril": self.get_scale_tril().numpy().tolist()
        }

    def get_effective_boundaries_1d(self,
            n_steps: int = 100_000,
            return_mapping: bool = False):
        """
        Numerically find the (n_cats-1) crossing points in [0,1] where the most likely Gaussian component switches, 
        and compute the ordering of components by where they dominate.

        If return_mapping=True, also returns `order` such that order[j] = original component index that occupies bin j.
        """
        # 1) fetch & activate parameters
        weight   = tf.nn.softmax(self.mixture_logits).numpy()      # (n_cats,)
        mu_raw  = np.array(self.means)[:, 0]                      # raw
        mu      = tf.math.sigmoid(mu_raw).numpy()                # in [0,1]
        sig     = self.get_scale_tril().numpy()[:,0,0]           # (n_cats,)

        # 2) build grid & evaluate weighted pdfs
        xgrid = np.linspace(0.0, 1.0, n_steps)
        pdfs  = np.vstack([
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

        # 6) now re-index winners into this sorted order (so that “bin 0” is the leftmost, bin 1 next, etc.)
        inv_order = np.argsort(order)  # maps old index -> new index
        winners_sorted = inv_order[winners]

        # 7) find the first n_cats-1 transitions
        jumps = np.where(winners_sorted[:-1] != winners_sorted[1:])[0]
        cuts  = []
        for j in jumps:
            cuts.append(0.5*(xgrid[j] + xgrid[j+1]))
            if len(cuts) >= self.n_cats - 1:
                break

        cuts.sort()
        if return_mapping:
            return cuts, order.tolist()
        return cuts

    def get_boundaries_for_history(self, n_steps: int = 100_000):
        """
        Return a list of length (n_cats-1).
        • position i  = boundary between component i and i+1
        • value       = crossing point in [0,1]   (float)
        • NaN         = this boundary does not exist in this epoch
        """
        from scipy.stats import norm
        import numpy as np

        # activated parameters -------------------------------------------------
        w   = tf.nn.softmax(self.mixture_logits).numpy()           # (n_cats,)
        mu  = tf.math.sigmoid(self.means[:, 0]).numpy()            # (n_cats,)
        sig = self.get_scale_tril().numpy()[:, 0, 0]               # (n_cats,)

        # sort by mu to get a *stable* left-to-right ordering
        order = np.argsort(mu)
        w, mu, sig = w[order], mu[order], sig[order]

        # dense grid + winning component at each x -----------------------------
        x    = np.linspace(0.0, 1.0, n_steps)
        pdf  = np.vstack([w_i * norm.pdf(x, mu_i, s_i)
                        for w_i, mu_i, s_i in zip(w, mu, sig)]).T
        winner = np.argmax(pdf, axis=1)                     # (n_steps,)

        # where does the winner change?  → boundaries
        jumps = np.where(winner[:-1] != winner[1:])[0]
        cuts  = 0.5 * (x[jumps] + x[jumps + 1])            # (<= n_cats-1,)

        # put them into fixed slots (NaN where missing)
        slots = np.full(self.n_cats - 1, np.nan, dtype=float)
        for idx, cut in zip(winner[jumps], cuts):
            # idx  = the left component of the pair (after sorting by µ)
            if idx < self.n_cats - 1:
                slots[idx] = cut
        return slots.tolist()          # length = n_cats-1

    def compute_hard_bkg_stats(self, data_dict, low=0.0, high=1.0, eps=1e-8):
        """
        Hard-assign each background event to its most-likely Gaussian component,
        then compute per-bin background yield B_j and relative uncertainty sigma_j/B_j,
        and return them sorted by bin position in 1D or significance in multi-D.

        Returns:
          B_sorted       np.array (n_cats,)
          rel_unc_sorted np.array (n_cats,)
          order          np.array mapping original→sorted component indices
        """

        raw_logits = self.mixture_logits.numpy()                     # (n_cats,)
        log_mix    = tf.nn.log_softmax(raw_logits).numpy()           # (n_cats,)
        raw_means  = self.means.numpy()                              # (n_cats,)
        if self.dim == 1:
            # map raw means into [0,1] with sigmoid
            mu_act = expit(raw_means.flatten())
        else:
            # map raw means into valid vectors via softmax across dims for each component
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
        B2= np.zeros(n_cats)
        for i in range(n_cats):
            sel = (comp==i)
            if not sel.any(): continue
            w_i = W[sel]
            B[i]  = w_i.sum()
            B2[i] = (w_i**2).sum()

        # compute rel-uncertainty
        sigma  = np.sqrt(np.maximum(B2,0.0))
        rel_unc= sigma/np.maximum(B,eps)

        # determine order
        if self.dim == 1:
            _, order = self.get_effective_boundaries_1d(return_mapping=True)
        else:
            # multi-D: sort by per-bin Z
            S_dummy = tf.zeros_like(tf.constant(B, dtype=tf.float32))
            Zbins = asymptotic_significance(S_dummy, tf.constant(B, dtype=tf.float32)).numpy()
            order = np.argsort(-Zbins)

        return B[order], rel_unc[order], order


def asymptotic_significance(S, B, eps=1e-9):
    """
    Default asymptotic significance function with S/sqrt(B) approximation at very low S/B.
    """
    safe_B = tf.maximum(B, eps)
    ratio = S / safe_B
    # Full Asimov formula:
    Z_asimov = tf.sqrt(2.0 * ((S + safe_B)*tf.math.log(1.0 + ratio) - S))
    # S/sqrt(B) approximation for small S/B:
    Z_approx = S / tf.sqrt(safe_B)
    # Switch where ratio < 0.1
    return tf.where(ratio < 0.1, Z_approx, Z_asimov)

def compute_significance_from_hists(h_signal, h_bkg_list):
    # Sum background counts bin-by-bin.
    B_vals = sum([h_bkg.values() for h_bkg in h_bkg_list])
    S_vals = h_signal.values()
    S_tensor = tf.constant(S_vals, dtype=tf.float32)
    B_tensor = tf.constant(B_vals, dtype=tf.float32)
    Z_bins = asymptotic_significance(S_tensor, B_tensor)
    Z_overall = np.sqrt(np.sum(Z_bins.numpy()**2))
    return Z_overall

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





# keeping some old stuff below
class _old_DifferentiableCutModel(tf.Module):
    """
    A generic model that:
      - Has one or more variables to cut on,
      - Each variable has (n_bins - 1) trainable boundaries,
      - Splits events by "soft membership" in each bin,
      - Accumulates yields for each process in each bin,
      - Then computes a figure of merit (like significance).

    Users can override or extend:
      - how yields are computed,
      - how significance is combined across categories,
      - how multiple signals or backgrounds are handled, etc.
    """

    def __init__(
        self,
        variables_config,
        significance_func=asymptotic_significance,
        name="DifferentiableCutModel",
        **kwargs
    ):
        """
        variables_config: list of dicts. Each dict must have:
          - "name": str, the name of the column in the data
          - "n_cats": int, number of categories for this variable
          - optionally "steepness": float for the soft cut transitions (defaults to 50)

        significance_func: function S, B -> significance
        """
        super().__init__(name=name)
        self.variables_config = variables_config
        self.significance_func = significance_func

        # Create a trainable tf.Variable for each variable's boundaries.
        self.raw_boundaries_list = []
        for var_cfg in variables_config:
            n_cats = var_cfg["n_cats"]
            shape = (n_cats,) if n_cats > 1 else (0,) # in current implementation, the last value is 1, i.e., will be ignored

            init_tensor = tf.random.normal(shape=shape, stddev=0.3)
            raw_var = tf.Variable(init_tensor, trainable=True, name=f"raw_bndry_{var_cfg['name']}")
            self.raw_boundaries_list.append(raw_var)

    def call(self, data_dict):
        """
        For each process in data_dict, we retrieve the relevant columns
        and apply the soft cut membership for each variable in self.variables_config.

        Then we combine membership from all variables. 
        (For simpler usage, we might do "product" of memberships or "logical" approach.)

        Returns a figure of merit (scalar) that we want to maximize.
        """
        raise NotImplementedError(
            "Base class: user must override the `call` method to define how yields are computed, to match the analysis specific needs! Examples are in /examples."
        )

    def get_effective_boundaries(self):
        """Return a dict { var_name: sorted list of boundaries in [0,1]. }"""
        out = {}
        for var_cfg, raw_var in zip(self.variables_config, self.raw_boundaries_list):
            if raw_var.shape[0] == 0:
                out[var_cfg["name"]] = []
            else:
                # out[var_cfg["name"]] = tf.sort(tf.sigmoid(raw_var)).numpy().tolist()
                out[var_cfg["name"]] = calculate_boundaries(raw_var)
        return out

def safe_sigmoid(z, steepness):
    z_clipped = tf.clip_by_value(-steepness * z, -75.0, 75.0)
    return 1.0 / (1.0 + tf.exp(z_clipped))

def soft_bin_weights(x, raw_boundaries, steepness=50.0):
    """
    Given a 1D array x and 'n_cats -1' raw boundaries (trainable in [-∞, +∞]),
    apply sigmoid to each boundary so they lie in [0,1], then do piecewise
    membership weighting for each category. 
    Returns a list of length n_cats, each entry is a TF tensor of membership weights in [0..1].
    """
    if raw_boundaries.shape[0] == 0:
        # means only 1 category => everything in that category
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
            w_i = safe_sigmoid(x - boundaries[i - 1], steepness) - safe_sigmoid(x - boundaries[i], steepness)
            w_list.append(w_i)
    return w_list

def calculate_boundaries(raw_boundaries):
    """
    Transforms raw boundaries (trainable) into an ordered set in (0,1)
    using a softmax transformation followed by a cumulative sum.

    The softmax ensures the increments are positive and sum to one,
    and the cumulative sum produces strictly increasing boundaries in [0,1].
    """
    increments = tf.nn.softmax(raw_boundaries)

    boundaries = tf.cumsum(increments)

    return boundaries[:-1] # ignore last value which is always 1.0