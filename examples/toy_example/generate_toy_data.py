import numpy as np
import pandas as pd

def sample_background(n_events, lam):
    """
    Sample NN outputs for a background using an exponential decay distribution.
    The PDF is proportional to exp(-lam * x) for x in [0,1].
    Uses inverse transform sampling.
    """
    u = np.random.uniform(0, 1, n_events)
    # Normalization factor: integral_0^1 exp(-lam * x) dx = (1 - exp(-lam))/lam
    # Inverse CDF:
    x = -1.0 / lam * np.log(1 - u * (1 - np.exp(-lam)))
    return x

def sample_signal(n_events, lam):
    """
    Sample NN outputs for a signal using a reversed exponential distribution.
    The PDF is proportional to exp(-lam*(1 - x)) for x in [0,1].
    Uses inverse transform sampling.
    """
    u = np.random.uniform(0, 1, n_events)
    # For reversed exponential, let z = 1-x ~ Exponential(lam) in [0,1]
    z = -1.0 / lam * np.log(1 - u * (1 - np.exp(-lam)))
    x = 1 - z
    return x

def generate_toy_data(n_signal=1000, 
                      n_bkg1=1000, n_bkg2=1000, n_bkg3=1000,
                      xs_signal=0.5, xs_bkg1=100, xs_bkg2=10, xs_bkg3=1,
                      lumi=100, seed=None):
    """
    Generate toy NN output data for one signal and three backgrounds.
    
    Parameters:
      - n_signal: number of generated events for the signal.
      - n_bkgX: number of generated events for each background.
      - xs_signal: cross section for the signal in pb (500 fb = 0.5 pb).
      - xs_bkg1, xs_bkg2, xs_bkg3: cross sections for the backgrounds in pb.
      - lumi: integrated luminosity in /fb.
      - seed: (optional) seed for reproducibility.
      
    Returns:
      A dictionary mapping process names to pandas DataFrames with a column "NN_output".
    """
    if seed is not None:
        np.random.seed(seed)
    
    # lambda parameters for the exponential distributions
    lam_bkg1 = 10    # steeper decay / easy background
    lam_bkg2 = 5    # medium friendly background
    lam_bkg3 = 2    # analyzers hate this background

    # For the signal, we want most events near 1. Use a lambda value to concentrate near 1.
    lam_signal = 5

    # Generate NN outputs
    nn_signal = sample_signal(n_signal, lam_signal)
    nn_bkg1 = sample_background(n_bkg1, lam_bkg1)
    nn_bkg2 = sample_background(n_bkg2, lam_bkg2)
    nn_bkg3 = sample_background(n_bkg3, lam_bkg3)

    # Optionally, one could compute per-event weights based on cross section and lumi,
    # e.g., weight = (xs * lumi) / n_generated
    weight_signal = xs_signal * lumi / n_signal
    weight_bkg1   = xs_bkg1   * lumi / n_bkg1
    weight_bkg2   = xs_bkg2   * lumi / n_bkg2
    weight_bkg3   = xs_bkg3   * lumi / n_bkg3

    # Create DataFrames (here we only store the NN output, but you could add weights if needed)
    df_signal = pd.DataFrame({"NN_output": nn_signal, "weight": weight_signal})
    df_bkg1 = pd.DataFrame({"NN_output": nn_bkg1, "weight": weight_bkg1})
    df_bkg2 = pd.DataFrame({"NN_output": nn_bkg2, "weight": weight_bkg2})
    df_bkg3 = pd.DataFrame({"NN_output": nn_bkg3, "weight": weight_bkg3})

    data = {
        "signal": df_signal,
        "bkg1": df_bkg1,
        "bkg2": df_bkg2,
        "bkg3": df_bkg3
    }

    return data

# If this module is executed directly, generate toy data and print summary stats.
if __name__ == "__main__":
    toy_data = generate_toy_data(seed=42)
    for proc, df in toy_data.items():
        print(f"{proc}: {len(df)} events, NN_output min={df['NN_output'].min():.3f}, max={df['NN_output'].max():.3f}")
