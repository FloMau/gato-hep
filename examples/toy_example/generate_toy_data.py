import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import tensorflow as tf

def sample_gaussian(n_events, mean, cov, seed=None):
    """
    Draw n_events from N(mean, cov) in ℝ³.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_events)

def generate_toy_data_gauss(
    n_signal=100000, n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
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
        "signal": np.array([ 0.5,  0.5,  1.0]),
        "bkg1":   np.array([ -0.5,  1.,  -1]),
        "bkg2":   np.array([ 0.25,  -0.25,  3.0]),
        "bkg3":   np.array([ 0.75,  0.25,  -0.5]),
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

    # 2) how many to draw and what weights
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
    for proc in means:
        X = sample_gaussian(counts[proc], means[proc], covs[proc], seed=seed)
        w = xs[proc] * lumi / counts[proc]
        raw[proc] = {"X": X, "w": w}

    # 4) build scipy MVN pdfs
    pdfs = {proc: multivariate_normal(means[proc], covs[proc])
            for proc in means}
    
    # total cross section of background
    total_bkg_xs = sum(xs[p] for p in pdfs if p != "signal")

    # 5) compute optimal discriminant for each proc’s points:
    dfs = {}
    for proc, info in raw.items():
        X = info["X"]
        w = info["w"]
        p_sig = pdfs["signal"].pdf(X)
        # sum of background pdfs at X
        p_bkg = sum((xs[p] / total_bkg_xs) * pdfs[p].pdf(X) for p in pdfs if p != "signal")

        # noise 
        p_sig *= np.abs((1 + np.random.normal(scale=0.2, size=p_sig.shape)))
        p_bkg *= np.abs((1 + np.random.normal(scale=0.2, size=p_bkg.shape)))
        # likelihood‐ratio and map to [0,1] via sigmoid‐like:
        lr = p_sig / (p_bkg + 1e-12)

        disc = lr / (1.0 + lr)
        # disc = 2 * tf.nn.sigmoid(lr) - 1
        dfs[proc] = pd.DataFrame({
            "NN_output": disc,
            "weight":    w
        })

    return dfs

# quick test
if __name__ == "__main__":
    toy = generate_toy_data_gauss(seed=42)
    for p, df in toy.items():
        print(f"{p}: {len(df)} events → NN_output in [{df.NN_output.min():.3f}, {df.NN_output.max():.3f}]")
