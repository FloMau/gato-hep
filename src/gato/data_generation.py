import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from scipy.stats import multivariate_normal

tfd = tfp.distributions


def sample_gaussian(n_events, mean, cov, seed=None):
    """
    Draw n_events from N(mean, cov) in ℝ³.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_events)


def generate_toy_data_1D(
    n_signal=100000, n_bkg=300000,
    xs_signal=0.5, xs_bkg1=50, xs_bkg2=15, xs_bkg3=10,
    lumi=100, seed=None
):
    """
    3D Gaussian toy: sample points for signal & 3 bkgs, then compute
    the likelihood-ratio discriminant (sig vs sum of bkgs), mapped to [0,1].
    Returns dict of DataFrames with columns “NN_output” and “weight”.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) define means & covariances
    means = {
        "signal": np.array([0.5,  0.5,  1.0]),
        "bkg1":   np.array([-0.5,  1.,  -1]),
        "bkg2":   np.array([0.25,  -0.25,  3.0]),
        "bkg3":   np.array([0.75,  0.25,  -0.5]),
    }

    covs = {
        "signal": np.array([[1, 0.2, 0.1],
                            [0.2, 1, 0.2],
                            [0.1, 0.2, 1]]),
        "bkg1":   np.array([[0.5, 0.2, 0.0],
                            [0.2, 0.5, 0.2],
                            [0.0, 0.2, 0.5]]),
        "bkg2":   np.array([[0.5, 0.1, 0.3],
                            [0.1, 0.5, 0.1],
                            [0.3, 0.1, 0.5]]),
        "bkg3":   np.array([[0.5, 0.2, 0.4],
                            [0.2, 0.5, 0.2],
                            [0.4, 0.2, 0.5]]),
    }

    # 2) split the single n_bkg *proportionally* to each cross-section so that
    # w_i = xs_i * lumi / N_i is constant across bkg processes:
    total_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3
    n_bkg1 = int(n_bkg * xs_bkg1 / total_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / total_xs_bkg)
    n_bkg3 = n_bkg - (n_bkg1 + n_bkg2)
    counts = {
        "signal": n_signal,
        "bkg1":   n_bkg1,
        "bkg2":   n_bkg2,
        "bkg3":   n_bkg3,
    }

    xs = {
        "signal": xs_signal,
        "bkg1":   xs_bkg1,
        "bkg2":   xs_bkg2,
        "bkg3":   xs_bkg3,
    }

    # 3) sample
    raw = {}
    for proc in means.items():
        X = sample_gaussian(counts[proc], means[proc], covs[proc], seed=seed)
        w = xs[proc] * lumi / counts[proc]
        raw[proc] = {"X": X, "w": w}

    # 4) build scipy MVN pdfs
    pdfs = {proc: multivariate_normal(means[proc], covs[proc])
            for proc in means}

    # total cross section of background
    total_bkg_xs = sum(xs[p] for p in pdfs if p != "signal")

    # 5) compute optimal discriminant for each proc's points:
    dfs = {}
    for proc, info in raw.items():
        X = info["X"]
        w = info["w"]
        p_sig = pdfs["signal"].pdf(X)
        # sum of background pdfs at X
        p_bkg = sum(
            (xs[p] / total_bkg_xs) * pdfs[p].pdf(X) for p in pdfs if p != "signal"
        )

        # noise
        p_sig *= np.abs(1 + np.random.normal(scale=0.3, size=p_sig.shape))
        p_bkg *= np.abs(1 + np.random.normal(scale=0.3, size=p_bkg.shape))
        # likelihood-ratio and map to [0,1] via sigmoid-like:
        lr = p_sig / (p_bkg + 1e-12)

        disc = lr / (1.0 + lr)
        # disc = 2 * tf.nn.sigmoid(lr) - 1
        dfs[proc] = pd.DataFrame({
            "NN_output": disc,
            "weight":    w
        })

    return dfs


# 3D Toy Data Generator for 3-class classifier
# Background consists of 5 individual Gaussian processes
def generate_toy_data_3class_3D(
    n_signal1=100000, n_signal2=100000,
    n_bkg=500000,
    xs_signal1=0.5, xs_signal2=0.1,
    xs_bkg1=100, xs_bkg2=80, xs_bkg3=50, xs_bkg4=20, xs_bkg5=10,
    lumi=100.0, noise_scale=0.2, seed=None
):
    """
    Generate 3D Gaussian data for 2 signal and 5 background classes.
    For each point, compute likelihood-ratio-based 3-class scores:
        [score_signal1, score_signal2, score_background]
    Returns dict of DataFrames with columns: 'NN_output' (3-vector) and 'weight'.

    n_bkg is the total number of background events; it will be
    split across bkg1..bkg5 in proportion to their cross-sections
    so that each background class ends up with the same per-event weight.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) split the single n_bkg into the five background counts
    total_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5
    n_bkg1 = int(n_bkg * xs_bkg1 / total_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / total_xs_bkg)
    n_bkg3 = int(n_bkg * xs_bkg3 / total_xs_bkg)
    n_bkg4 = int(n_bkg * xs_bkg4 / total_xs_bkg)
    n_bkg5 = n_bkg - (n_bkg1 + n_bkg2 + n_bkg3 + n_bkg4)

    processes = ["signal1", "signal2", "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"]

    means = {
        "signal1": np.array([1.5, -1.0, -1.0]),
        "signal2": np.array([-1.0, 1.5, -1.0]),
        "bkg1":    np.array([-0.5, -0.5, 1.0]),
        "bkg2":    np.array([0.5, -0.5, 0.8]),
        "bkg3":    np.array([0.5,  0.5, -0.6]),
        "bkg4":    np.array([-0.5,  1.0, -0.4]),
        "bkg5":    np.array([-0.5,  0.5, -0.2])
    }

    # Slightly correlated 3D Gaussian
    cov = np.eye(3)*1.0 + 0.2*(np.ones((3,3)) - np.eye(3))

    counts = {
        "signal1": n_signal1,
        "signal2": n_signal2,
        "bkg1":    n_bkg1,
        "bkg2":    n_bkg2,
        "bkg3":    n_bkg3,
        "bkg4":    n_bkg4,
        "bkg5":    n_bkg5
    }

    xs = {
        "signal1": xs_signal1,
        "signal2": xs_signal2,
        "bkg1":    xs_bkg1,
        "bkg2":    xs_bkg2,
        "bkg3":    xs_bkg3,
        "bkg4":    xs_bkg4,
        "bkg5":    xs_bkg5
    }

    # 1. Sample raw 3D data
    raw = {
        p: np.random.multivariate_normal(mean=means[p], cov=cov, size=counts[p])
        for p in processes
    }

    # 2. Add multiplicative noise
    for p in processes:
        noise = np.random.normal(loc=1.0, scale=noise_scale, size=raw[p].shape)
        raw[p] *= noise

    # 3. Build PDFs
    pdfs = {
        p: multivariate_normal(mean=means[p], cov=cov)
        for p in processes
    }

    # 4. Combined background PDF with proper cross-section weighting
    bkg_processes = [p for p in processes if p.startswith("bkg")]
    total_bkg_xs = sum(xs[p] for p in bkg_processes)

    def combined_bkg_pdf(X):
        return sum(
            (xs[p] / total_bkg_xs) * pdfs[p].pdf(X)
            for p in bkg_processes
        )

    # 5. Compute likelihood-ratio-based scores
    data = {}
    for proc in processes:
        X = raw[proc]
        weight = xs[proc] * lumi / counts[proc]

        p1 = pdfs["signal1"].pdf(X)
        p2 = pdfs["signal2"].pdf(X)
        pb = combined_bkg_pdf(X)

        total = p1 + p2 + pb + 1e-12  # avoid divide-by-zero

        score1 = p1 / total
        score2 = p2 / total
        score_bkg = pb / total

        nn_output = np.stack([score1, score2, score_bkg], axis=1)
        nn_output = list(nn_output)

        data[proc] = pd.DataFrame({
            "NN_output": nn_output,
            "weight":    weight
        })

    return data
