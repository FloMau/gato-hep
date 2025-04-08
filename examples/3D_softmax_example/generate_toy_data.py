import numpy as np
import pandas as pd
import tensorflow as tf

def generate_toy_data_multiclass(n_signal=1000, n_bkg1=1000, n_bkg2=1000,
                                 # means chosen so that, after softmax, the highest probability is for the desired class.
                                 mean_signal=[1.5, -1.5, -1.5],
                                 mean_bkg1=[-1.0, 0.0, -2.0],
                                 mean_bkg2=[-2.0, -2.0, 0.0],
                                 # common covariance introduces moderate correlations
                                 cov=[[1.0, 0.3, 0.3],
                                      [0.3, 1.0, 0.3],
                                      [0.3, 0.3, 1.0]],
                                 xs_signal=0.5, xs_bkg1=100, xs_bkg2=10,
                                 lumi=100, seed=None):
    """
    Generate toy NN output data for three processes: signal, bkg1, and bkg2.
    
    Each event is a 3D vector, which is passed through a softmax so that the output
    is a probability vector over the three classes. The distributions are chosen such that:
      - signal events are biased to have high probability in the first component,
      - bkg1 events are biased to have high probability in the second component,
      - bkg2 events are biased to have high probability in the third component.
    
    Parameters:
      - n_signal, n_bkg1, n_bkg2: Number of events for each process.
      - mean_signal, mean_bkg1, mean_bkg2: Mean vectors for the multivariate normal distributions.
      - cov: Covariance matrix (same for all processes here).
      - xs_signal, xs_bkg1, xs_bkg2: Cross sections (in pb) for signal and backgrounds.
      - lumi: Integrated luminosity in /fb.
      - seed: (optional) random seed for reproducibility.
      
    Returns:
      A dictionary mapping process names to pandas DataFrames with columns:
        - "NN_output": a list/array of 3 softmax-activated values,
        - "weight": per-event weight.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate raw 3D outputs for each process
    raw_signal = np.random.multivariate_normal(mean_signal, cov, n_signal)
    raw_bkg1   = np.random.multivariate_normal(mean_bkg1, cov, n_bkg1)
    raw_bkg2   = np.random.multivariate_normal(mean_bkg2, cov, n_bkg2)

    # Apply softmax row-wise
    signal_softmax = tf.nn.softmax(tf.constant(raw_signal, dtype=tf.float32), axis=1).numpy()
    bkg1_softmax   = tf.nn.softmax(tf.constant(raw_bkg1, dtype=tf.float32), axis=1).numpy()
    bkg2_softmax   = tf.nn.softmax(tf.constant(raw_bkg2, dtype=tf.float32), axis=1).numpy()

    # Compute per-event weights based on cross section and lumi
    weight_signal = xs_signal * lumi / n_signal
    weight_bkg1   = xs_bkg1   * lumi / n_bkg1
    weight_bkg2   = xs_bkg2   * lumi / n_bkg2

    # Create DataFrames; we store the NN_output as a list for each event.
    df_signal = pd.DataFrame({
        "NN_output": list(signal_softmax),
        "weight": weight_signal
    })
    df_bkg1 = pd.DataFrame({
        "NN_output": list(bkg1_softmax),
        "weight": weight_bkg1
    })
    df_bkg2 = pd.DataFrame({
        "NN_output": list(bkg2_softmax),
        "weight": weight_bkg2
    })

    data = {
        "signal": df_signal,
        "bkg1": df_bkg1,
        "bkg2": df_bkg2
    }
    
    return data

# If executed directly, generate toy data and print summary statistics.
if __name__ == "__main__":
    toy_data = generate_toy_data_multiclass(seed=42)
    for proc, df in toy_data.items():
        # Extract the NN_output as a 2D array for summary stats.
        outputs = np.stack(df["NN_output"].values)
        print(f"{proc}: {len(df)} events")
        print(f"  NN_output: mean = {outputs.mean(axis=0)}, std = {outputs.std(axis=0)}")
