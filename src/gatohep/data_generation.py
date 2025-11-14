import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# 0.  Global Gaussian parameters  (2 signals + 5 backgrounds), used in both generators
MEANS = {
    "signal1": np.array([1.5, -1.0, -1.0]),
    "signal2": np.array([-1.0, 1.5, -1.0]),
    "bkg1":    np.array([-0.5, -0.5, 1.0]),
    "bkg2":    np.array([0.5, -0.5, 0.8]),
    "bkg3":    np.array([0.5, 0.5, -0.6]),
    "bkg4":    np.array([-0.5, 1.0, -0.4]),
    "bkg5":    np.array([-0.5, 0.5, -0.2]),
}

# Slightly correlated 3-D covariance shared by *all* components
COV = np.eye(3) * 1.0 + 0.2 * (np.ones((3, 3)) - np.eye(3))


# Helper to sample a named component
def _sample(name: str, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(MEANS[name], COV, size=n)


# 1.  Unchanged 3-D generator (now just uses global MEANS / COV)
def generate_toy_data_3class_3D(
    n_signal1: int = 100_000,
    n_signal2: int = 100_000,
    n_bkg: int = 500_000,
    xs_signal1: float = 0.5,
    xs_signal2: float = 0.1,
    xs_bkg1: float = 100,
    xs_bkg2: float = 80,
    xs_bkg3: float = 50,
    xs_bkg4: float = 20,
    xs_bkg5: float = 10,
    lumi: float = 100.0,
    noise_scale: float = 0.3,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:

    """
    Generate 3D Gaussian data for 2 signal and 5 background classes.

    For each point, compute likelihood-ratio-based 3-class scores:
    [score_signal1, score_signal2, score_background].

    Parameters
    ----------
    n_signal1 : int, optional
        Number of events for signal1. Default is 100000.
    n_signal2 : int, optional
        Number of events for signal2. Default is 100000.
    n_bkg : int, optional
        Total number of background events. Default is 500000.
    xs_signal1 : float, optional
        Cross-section for signal1. Default is 0.5.
    xs_signal2 : float, optional
        Cross-section for signal2. Default is 0.1.
    xs_bkg1 : float, optional
        Cross-section for background1. Default is 100.
    xs_bkg2 : float, optional
        Cross-section for background2. Default is 80.
    xs_bkg3 : float, optional
        Cross-section for background3. Default is 50.
    xs_bkg4 : float, optional
        Cross-section for background4. Default is 20.
    xs_bkg5 : float, optional
        Cross-section for background5. Default is 10.
    lumi : float, optional
        Luminosity for scaling event weights. Default is 100.0.
    noise_scale : float, optional
        Scale of multiplicative noise applied to the data. Default is 0.2.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary of DataFrames, each containing the generated toy data
        with columns:
        - 'NN_output': 3-vector of scores\
        [score_signal1, score_signal2, score_background].
        - 'weight': Event weight.
    """

    if seed is not None:
        np.random.seed(seed)

    total_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5
    n_bkg1 = int(n_bkg * xs_bkg1 / total_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / total_xs_bkg)
    n_bkg3 = int(n_bkg * xs_bkg3 / total_xs_bkg)
    n_bkg4 = int(n_bkg * xs_bkg4 / total_xs_bkg)
    n_bkg5 = n_bkg - (n_bkg1 + n_bkg2 + n_bkg3 + n_bkg4)

    processes = [
        "signal1", "signal2",
        "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"
    ]
    counts = dict(
        signal1=n_signal1, signal2=n_signal2,
        bkg1=n_bkg1, bkg2=n_bkg2, bkg3=n_bkg3, bkg4=n_bkg4, bkg5=n_bkg5,
    )
    xs = dict(
        signal1=xs_signal1, signal2=xs_signal2,
        bkg1=xs_bkg1, bkg2=xs_bkg2, bkg3=xs_bkg3, bkg4=xs_bkg4, bkg5=xs_bkg5,
    )

    raw = {
        p: _sample(p, counts[p], seed + i if seed else None)
        for i, p in enumerate(processes)
    }
    # apply noise to the sampled data to mimic detector effects
    for p in processes:
        raw[p] *= np.random.normal(1.0, noise_scale, size=raw[p].shape)

    pdfs = {p: multivariate_normal(MEANS[p], COV) for p in processes}

    bkg_procs = [p for p in processes if p.startswith("bkg")]
    total_bkg_xs = sum(xs[p] for p in bkg_procs)

    def pb(X):
        return sum((xs[p] / total_bkg_xs) * pdfs[p].pdf(X) for p in bkg_procs)

    data = {}
    for proc in processes:
        X = raw[proc]
        w = xs[proc] * lumi / counts[proc]

        p1 = pdfs["signal1"].pdf(X)
        p2 = pdfs["signal2"].pdf(X)
        pB = pb(X)
        tot = p1 + p2 + pB + 1e-12

        nn_output = np.column_stack((p1 / tot, p2 / tot, pB / tot))
        data[proc] = pd.DataFrame({
            "NN_output": list(nn_output),
            "weight": w,
        })
    return data


def generate_toy_data_1D(
    n_signal: int = 100_000,
    n_bkg: int = 100_000,
    xs_signal: float = 0.5,
    xs_bkg1: float = 100,
    xs_bkg2: float = 80,
    xs_bkg3: float = 50,
    xs_bkg4: float = 20,
    xs_bkg5: float = 10,
    lumi: float = 100.0,
    noise_scale: float = 0.3,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate 1D toy data for signal and background events.

    Parameters
    ----------
    n_signal : int, optional
        Number of signal events to generate. Default is 100000.
    n_bkg : int, optional
        Number of background events to generate. Default is 300000.
    xs_signal : float, optional
        Cross-section for signal events. Default is 0.5.
    xs_bkg1 : float, optional
        Cross-section for the first background component. Default is 50.
    xs_bkg2 : float, optional
        Cross-section for the second background component. Default is 15.
    xs_bkg3 : float, optional
        Cross-section for the third background component. Default is 10.
    xs_bkg4 : float, optional
        Cross-section for the fourth background component. Default is 20.
    xs_bkg5 : float, optional
        Cross-section for the fifth background component. Default is 10.
    lumi : float, optional
        Luminosity for scaling event weights. Default is 100.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary of DataFrames, each containing the generated toy data
        with columns "NN_output" and "weight".
    """

    if seed is not None:
        np.random.seed(seed)

    tot_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5
    n_bkg1 = int(n_bkg * xs_bkg1 / tot_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / tot_xs_bkg)
    n_bkg3 = int(n_bkg * xs_bkg3 / tot_xs_bkg)
    n_bkg4 = int(n_bkg * xs_bkg4 / tot_xs_bkg)
    n_bkg5 = n_bkg - (n_bkg1 + n_bkg2 + n_bkg3 + n_bkg4)

    counts = dict(
        signal=n_signal,
        bkg1=n_bkg1,
        bkg2=n_bkg2,
        bkg3=n_bkg3,
        bkg4=n_bkg4,
        bkg5=n_bkg5,
    )
    xs = dict(
        signal=xs_signal,
        bkg1=xs_bkg1,
        bkg2=xs_bkg2,
        bkg3=xs_bkg3,
        bkg4=xs_bkg4,
        bkg5=xs_bkg5,
    )

    X = {
        "signal": _sample("signal1", n_signal, seed),
        "bkg1":   _sample("bkg1",    n_bkg1,  seed + 1 if seed else None),
        "bkg2":   _sample("bkg2",    n_bkg2,  seed + 2 if seed else None),
        "bkg3":   _sample("bkg3",    n_bkg3,  seed + 3 if seed else None),
        "bkg4":   _sample("bkg4",    n_bkg4,  seed + 4 if seed else None),
        "bkg5":   _sample("bkg5",    n_bkg5,  seed + 5 if seed else None),
    }
    # apply noise to the sampled data to mimic detector effects
    for proc in X:
        X[proc] *= np.random.normal(1.0, noise_scale, size=X[proc].shape)

    pdf_sig = multivariate_normal(MEANS["signal1"], COV)
    pdf_bkg = {
        p: multivariate_normal(MEANS[p], COV)
        for p in ("bkg1", "bkg2", "bkg3", "bkg4", "bkg5")
    }
    total_bkg_xs = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5

    def _pb(x):
        return sum((xs[p] / total_bkg_xs) * pdf_bkg[p].pdf(x) for p in pdf_bkg)

    data = {}
    for proc in ("signal", "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"):
        Xp = X[proc]
        ps = pdf_sig.pdf(Xp)
        pb = _pb(Xp)

        lr = ps / (pb + 1e-12)
        disc = lr / (1.0 + lr)        # map to (0,1)

        data[proc] = pd.DataFrame({
            "NN_output": disc,
            "weight": xs[proc] * lumi / counts[proc],
        })

    return data
