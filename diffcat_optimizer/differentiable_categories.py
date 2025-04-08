import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

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
    # Switch where ratio < 0.001
    return tf.where(ratio < 0.1, Z_approx, Z_asimov)

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


def low_bkg_penalty(b_nonres_cat, threshold=10.0, steepness=10):
    """
    b_nonres_cat: tf.Tensor of shape [ncat], the nonres yields in each category 
                  (over the entire mass range).
    We define penalty_j = alpha * sigmoid( steepness * (threshold - b_nonres_cat[j]) ).
    Then we sum over all categories.
    """
    # clamp it to avoid negative yields
    safe_b = tf.maximum(b_nonres_cat, 0.0)
    # define x = threshold - safe_b
    x = threshold - safe_b
    # define s = sigmoid( x * steepness )
    step = safe_sigmoid(x, steepness)
    # penalty per category
    penalty_vals = step * (b_nonres_cat - threshold)**2
    # sum over categories
    total_penalty = tf.reduce_sum(penalty_vals)
    return total_penalty


class DifferentiableCutModel(tf.Module):
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


class DiffCatModelMultiDimensional(DifferentiableCutModel):
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

    The call() method loops over processes in data_dict (a dict of pandas DataFrames
    with columns "NN_output" (an array-like of shape [dim]) and "weight"), computes 
    soft memberships, accumulates yields, and returns the negative overall significance
    as loss (plus background yields).
    """
    def __init__(self, n_cats, dim, temperature=1.0, name="DiffCatModelMultiDimensional"):
        # Here, we don't use the 1D boundaries. We use our own parameters.
        super().__init__(variables_config=[{"name": "NN_output", "n_cats": n_cats}], name=name)
        self.n_cats = n_cats
        self.dim = dim
        self.temperature = temperature

        # Mixture logits (shape: [n_cats])
        self.mixture_logits = tf.Variable(tf.random.normal([n_cats], stddev=0.1), trainable=True, name="mixture_logits")

        # Means (shape: [n_cats, dim])
        # self.raw_means = tf.Variable(tf.random.normal([n_cats, dim], stddev=0.3), trainable=True, name="raw_means")
        # self.means = tf.Variable(tf.nn.sigmoid(self.raw_means), trainable=True)
        self.means = tf.Variable(tf.random.normal([n_cats, dim], mean=0., stddev=2), trainable=True, name="means")

        # Unconstrained lower triangular matrices (shape: [n_cats, dim, dim])
        initial_L = np.array(
            [1/(2*self.n_cats ** (1/self.dim)) * np.eye(dim) for _ in range(n_cats)],
            dtype=np.float32
        )
        self.unconstrained_L = tf.Variable(
            tf.random.normal([n_cats, dim, dim], mean=0.0, stddev=1/(10*self.n_cats ** (1/self.dim))) + initial_L,
            trainable=True,
            name="unconstrained_L"
        )

    def get_scale_tril(self):
        """
        Returns the Cholesky factors (lower-triangular with positive diagonal) for each category.
        """
        L = tf.linalg.band_part(self.unconstrained_L, -1, 0)
        # print("L before:", L)
        # L = tf.nn.sigmoid(L) # constrain the covariances of the Gaussians
        # print("L after:", L)
        diag = tf.linalg.diag_part(L)
        # print("Diag", diag)
        # diag_positive = tf.nn.relu(diag)
        # diag_positive = 1/np.sqrt(self.n_cats) * tf.nn.softplus(diag)
        diag_positive = 1/(2*self.n_cats ** (1/self.dim)) * tf.nn.softplus(diag - 1)
        # print("Diag pos", diag_positive)
        # print("return:", tf.linalg.set_diag(L, diag_positive))
        return tf.linalg.set_diag(0*L, diag_positive)

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
