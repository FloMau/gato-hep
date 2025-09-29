import numpy as np
import tensorflow as tf

from gatohep.models import gato_sigmoid_model
from gatohep.data_generation import generate_toy_data_1D
from gatohep.utils import asymptotic_significance


def _to_tensor_1d(df):
    """Convert the 1D toy dataframe into the tensor dict the model expects."""
    x = df["NN_output"].values.astype("float32").reshape(-1, 1)  # (N,1)
    w = df["weight"].values.astype("float32")                    # (N,)
    return {
        "NN_output": tf.convert_to_tensor(x, tf.float32),
        "weight": tf.convert_to_tensor(w, tf.float32),
    }


def _loss_sig_vs_bkg(model, tdata):
    """
    Single-signal significance (as in the 1D example):
    Z = sqrt(sum_k Z_k^2) with Z_k from the Asimov formula.
    """
    probs = model.get_probs({k: {"NN_output": v["NN_output"]}
                             for k, v in tdata.items()})
    K = model.n_cats
    S = tf.zeros(K, tf.float32)
    B = tf.zeros(K, tf.float32)

    for proc, g in probs.items():
        w = tdata[proc]["weight"]                           # (N,)
        y = tf.reduce_sum(g * w[:, None], axis=0)           # (K,)
        if proc == "signal":
            S += y
        else:
            B += y

    Z_bins = asymptotic_significance(S, B)
    Z = tf.sqrt(tf.reduce_sum(Z_bins ** 2))
    return -Z  # minimise


def test_sigmoid_end_to_end(tmp_path):
    # 1) Generate a small dataset via the real 1D pipeline
    df = generate_toy_data_1D(
        n_signal=600, n_bkg=1800, xs_signal=0.5, xs_bkg1=100, xs_bkg2=80, xs_bkg3=50,
        lumi=100, seed=123
    )
    tdata = {k: _to_tensor_1d(v) for k, v in df.items()}

    # 2) Build a 1D sigmoid model with 4 bins on [0,1]
    tf.random.set_seed(321)
    model = gato_sigmoid_model(
        variables_config=[{"name": "NN_output", "bins": 4, "range": (0.0, 1.0)}],
        global_steepness=50.0,
    )

    # 3) One quick gradient step on the significance loss
    opt = tf.keras.optimizers.Adam(1e-2)
    raw_before = [cfg["raw"].numpy().copy() for cfg in model.var_cfg]

    with tf.GradientTape() as tape:
        loss = _loss_sig_vs_bkg(model, tdata)
    grads = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None and tf.reduce_any(tf.math.is_finite(g)) for g in grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    raw_after = [cfg["raw"].numpy() for cfg in model.var_cfg]
    # Some boundary parameter should change
    assert any(np.linalg.norm(a - b) > 0 for a, b in zip(raw_after, raw_before))

    # 4) API checks: get_probs shape/normalization and get_bin_indices bounds
    probs = model.get_probs(
        {
            k: {"NN_output": v["NN_output"]}
            for k, v in tdata.items()
        }
    )
    for p in probs.values():
        assert p.shape[1] == model.n_cats
        rowsum = tf.reduce_sum(p, axis=1).numpy()
        np.testing.assert_allclose(
            rowsum,
            np.ones_like(rowsum),
            rtol=1e-5,
            atol=1e-5,
        )
        pmin, pmax = float(tf.reduce_min(p)), float(tf.reduce_max(p))
        assert -1e-6 <= pmin <= 1.0 + 1e-6 and 0.0 <= pmax <= 1.0 + 1e-6

    hard = model.get_bin_indices(
        {
            k: {"NN_output": v["NN_output"]}
            for k, v in tdata.items()
        }
    )
    for h in hard.values():
        hnp = h.numpy()
        assert hnp.ndim == 1 and hnp.size > 0
        assert hnp.min() >= 0 and hnp.max() < model.n_cats

    # 5) Exercise compute_hard_bkg_stats (per-bin B and rel. unc.)
    B, rel_unc = model.compute_hard_bkg_stats(
        {
            k: {"NN_output": v["NN_output"], "weight": v["weight"]}
            for k, v in tdata.items()
        },
        signal_labels=["signal"],
    )
    assert int(B.shape[0]) == model.n_cats
    assert int(rel_unc.shape[0]) == model.n_cats
    assert np.all(np.isfinite(B.numpy()))
    assert np.all(np.isfinite(rel_unc.numpy()))

    # 6) Save and restore using the class methods
    ckpt_dir = tmp_path.as_posix()
    model.save(ckpt_dir)

    model2 = gato_sigmoid_model(
        variables_config=[{"name": "NN_output", "bins": 4, "range": (0.0, 1.0)}],
        global_steepness=50.0,
    )
    model2.restore(ckpt_dir)

    # Boundaries and steepness must match after reload
    b1 = model.calculate_boundaries().numpy()
    b2 = model2.calculate_boundaries().numpy()
    np.testing.assert_allclose(b1, b2, rtol=1e-6, atol=1e-6)

    # Also compare raw params for completeness
    for cfg1, cfg2 in zip(model.var_cfg, model2.var_cfg):
        np.testing.assert_allclose(
            cfg1["raw"].numpy(),
            cfg2["raw"].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cfg1["k"].numpy(),
            cfg2["k"].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
