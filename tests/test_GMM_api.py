import numpy as np
import tensorflow as tf

from gatohep.models import gato_gmm_model
from gatohep.data_generation import generate_toy_data_3class_3D
from gatohep.utils import asymptotic_significance


def _to_tensor_2d(df):
    """Use first two softmax nodes -> (N,2) like the example."""
    x = np.stack(df["NN_output"].values)[:, :2].astype("float32")
    w = df["weight"].values.astype("float32")
    return {
        "NN_output": tf.convert_to_tensor(x, tf.float32),
        "weight": tf.convert_to_tensor(w, tf.float32),
    }


def _geom_mean_loss(model, tdata):
    """
    Geometric mean of the two signal Asimov significances (as in the example).
    Uses project methods: get_probs + asymptotic_significance.
    """
    probs = model.get_probs({k: {"NN_output": v["NN_output"]}
                             for k, v in tdata.items()})
    K = model.n_cats
    sig1_y = tf.zeros(K, tf.float32)
    sig2_y = tf.zeros(K, tf.float32)
    bkg_y = tf.zeros(K, tf.float32)

    for proc, g in probs.items():
        w = tdata[proc]["weight"]                          # (N,)
        y = tf.reduce_sum(g * w[:, None], axis=0)          # (K,)
        if proc == "signal1":
            sig1_y += y
        elif proc == "signal2":
            sig2_y += y
        else:
            bkg_y += y

    Z1_bins = asymptotic_significance(sig1_y, bkg_y + sig2_y)
    Z2_bins = asymptotic_significance(sig2_y, bkg_y + sig1_y)
    Z1 = tf.sqrt(tf.reduce_sum(Z1_bins**2))
    Z2 = tf.sqrt(tf.reduce_sum(Z2_bins**2))
    return -tf.sqrt(Z1 * Z2)  # scalar loss


def test_threeclass_end_to_end(tmp_path):
    # 1) small dataset via your generator
    data_df = generate_toy_data_3class_3D(
        n_signal1=600, n_signal2=600, n_bkg=1800, noise_scale=0.2, seed=7
    )
    tensor_data = {k: _to_tensor_2d(df) for k, df in data_df.items()}

    # 2) model like the example (2-D, softmax means)
    tf.random.set_seed(123)
    model = gato_gmm_model(n_cats=4, dim=2, mean_norm="softmax", temperature=0.7)

    # 3) one quick gradient step on the geometric-mean Z loss
    opt = tf.keras.optimizers.Adam(1e-2)
    w_before = model.mixture_logits.numpy().copy()
    with tf.GradientTape() as tape:
        loss = _geom_mean_loss(model, tensor_data)
    grads = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None and tf.reduce_any(tf.math.is_finite(g)) for g in grads)

    opt.apply_gradients(zip(grads, model.trainable_variables))
    w_after = model.mixture_logits.numpy()
    assert np.linalg.norm(w_after - w_before) > 0.0  # parameters updated

    # 4) API checks: get_probs shape/normalization; get_bin_indices range
    probs = model.get_probs(
        {
            k: {"NN_output": v["NN_output"]}
            for k, v in tensor_data.items()
        }
    )
    for p in probs.values():
        assert p.shape[1] == model.n_cats
        row_sums = tf.reduce_sum(p, axis=1).numpy()
        np.testing.assert_allclose(
            row_sums,
            np.ones_like(row_sums),
            rtol=1e-5,
            atol=1e-5,
        )
        pmin, pmax = float(tf.reduce_min(p)), float(tf.reduce_max(p))
        assert -1e-6 <= pmin <= 1.0 + 1e-6 and 0.0 <= pmax <= 1.0 + 1e-6

    hard = model.get_bin_indices(
        {
            k: {"NN_output": v["NN_output"]}
            for k, v in tensor_data.items()
        }
    )
    for h in hard.values():
        arr = h.numpy()
        assert arr.ndim == 1 and arr.size > 0
        assert arr.min() >= 0 and arr.max() < model.n_cats

    # 5) also exercise compute_hard_bkg_stats
    B_sorted, rel_unc_sorted, order = model.compute_hard_bkg_stats(
        {
            k: {"NN_output": v["NN_output"], "weight": v["weight"]}
            for k, v in tensor_data.items()
        },
        signal_labels=["signal1", "signal2"],
    )
    assert B_sorted.shape == rel_unc_sorted.shape == order.shape
    assert B_sorted.shape[0] == model.n_cats
    assert np.all(np.isfinite(B_sorted)) and np.all(np.isfinite(rel_unc_sorted))

    # 6) save via class method, then restore into a fresh instance and compare
    ckpt_dir = tmp_path.as_posix()
    model.save(ckpt_dir)   # uses gato_gmm_model.save()

    model2 = gato_gmm_model(
        n_cats=model.n_cats, dim=model.dim,
        mean_norm=model.mean_norm,
        mean_range=getattr(model, "mean_range", (0.0, 1.0)),
        temperature=model.temperature,
    )
    model2.restore(ckpt_dir)  # uses gato_gmm_model.restore()

    np.testing.assert_allclose(
        model.mixture_logits.numpy(),
        model2.mixture_logits.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.means.numpy(),
        model2.means.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.get_scale_tril().numpy(),
        model2.get_scale_tril().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
