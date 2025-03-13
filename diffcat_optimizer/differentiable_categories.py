import numpy as np
import pandas as pd
import tensorflow as tf

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
    return tf.where(ratio < 0.001, Z_approx, Z_asimov)

def safe_sigmoid(z, steepness):
    z_clipped = tf.clip_by_value(-steepness * z, -50.0, 50.0)
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

    boundaries = tf.sort(tf.sigmoid(raw_boundaries))  # ensure ascending order in [0,1]
    #### to be checked!!!

    # define a safe sigmoid that won't overflow for large inputs

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

        # Create a trainable tf.Variable for each variable's boundaries
        self.raw_boundaries_list = []
        for var_cfg in variables_config:
            n_cats = var_cfg["n_cats"]
            shape = (n_cats - 1,) if n_cats > 1 else (0,)
            raw_var = tf.Variable(
                tf.random.normal(shape=shape, stddev=0.3),
                trainable=True,
                name=f"raw_bndry_{var_cfg['name']}"
            )
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
                out[var_cfg["name"]] = tf.sort(tf.sigmoid(raw_var)).numpy().tolist()
        return out
