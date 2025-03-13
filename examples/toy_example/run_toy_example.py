import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf
import hist

# Append the repo root to sys.path so that we can import our core modules.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffcat_optimizer.plotting_utils import plot_stacked_histograms
from diffcat_optimizer.differentiable_categories import asymptotic_significance, soft_bin_weights, DifferentiableCutModel
from generate_toy_data import generate_toy_data

# ------------------------------------------------------------------------------
# Helper: Create a fixed histogram from 1D data.
# ------------------------------------------------------------------------------
def create_hist(data, weights=None, bins=50, low=0.0, high=1.0, name="NN_output"):
    # If bins is an integer, we do regular binning:
    if isinstance(bins, int):
        h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    # Otherwise, assume bins is an array of edges:
    else:
        h = hist.Hist.new.Var(bins, name=name).Weight()
    if weights is not None:
        h.fill(data, weight=weights)
    else:
        h.fill(data)
    return h

# ------------------------------------------------------------------------------
# Fixed binning significance: rebin the fixed histograms and compute overall significance.
# ------------------------------------------------------------------------------
def compute_significance(h_signal, h_bkg_list):
    # Sum background counts bin-by-bin.
    B_vals = h_bkg_list[0].values() + h_bkg_list[1].values() + h_bkg_list[2].values()
    S_vals = h_signal.values()
    S_tensor = tf.constant(S_vals, dtype=tf.float32)
    B_tensor = tf.constant(B_vals, dtype=tf.float32)
    Z_bins = asymptotic_significance(S_tensor, B_tensor)
    Z_overall = np.sqrt(np.sum(Z_bins.numpy()**2))
    return Z_overall

# ------------------------------------------------------------------------------
# Differentiable model: inline subclass that optimizes bin boundaries.
# ------------------------------------------------------------------------------
class one_dimensional_binning_optimiser(DifferentiableCutModel):
    """
    A toy model that optimizes the NN output bin boundaries.
    It assumes the input data_dict has key "signal" for signal and others for background.
    """
    def __init__(self, n_cats, steepness=5000.0):
        # n_cats is the number of bins desired.
        variables_config = [
            {"name": "NN_output", "n_cats": n_cats, "steepness": steepness}
        ]
        super().__init__(variables_config, name="ToyOptModel")
    
    def call(self, data_dict):
        # Use the NN_output variable.
        n_bins = self.variables_config[0]["n_cats"]
        steepness = self.variables_config[0].get("steepness", 50.0)
        raw_boundaries = self.raw_boundaries_list[0]
        # Initialize lists for yields.
        signal_yields = [0.0] * n_bins
        background_yields = [0.0] * n_bins
        # Loop over processes.
        for proc, df in data_dict.items():
            if df.empty:
                continue
            x = tf.constant(df["NN_output"].values, dtype=tf.float32)
            w = tf.constant(df["weight"].values, dtype=tf.float32)
            memberships = soft_bin_weights(x, raw_boundaries, steepness=steepness)
            for i, m in enumerate(memberships):
                yield_i = tf.reduce_sum(m * w)
                if proc == "signal":
                    signal_yields[i] += yield_i
                else:
                    background_yields[i] += yield_i
        S = tf.convert_to_tensor(signal_yields, dtype=tf.float32)
        B = tf.convert_to_tensor(background_yields, dtype=tf.float32)
        Z_bins = asymptotic_significance(S, B)
        Z_overall = tf.sqrt(tf.reduce_sum(tf.square(Z_bins)))
        return -Z_overall  # Return negative significance as loss.

# ------------------------------------------------------------------------------
# Main: Generate data, run fixed binning and optimization, then compare.
# ------------------------------------------------------------------------------
def main():
    # 1. Generate toy data
    data = generate_toy_data(
        n_signal=100000,
        n_bkg1=100000, n_bkg2=100000, n_bkg3=100000,
        xs_signal=0.5,    # 500 fb = 0.5 pb
        xs_bkg1=100, xs_bkg2=50, xs_bkg3=15,
        lumi=100,         # in /fb
        seed=42
    )

    # Create fixed histograms (with equidistant binning using 50 bins as baseline).
    n_bins = 50
    low = 0.0
    high = 1.0
    hist_signal = create_hist(data["signal"]["NN_output"], weights=data["signal"]["weight"], bins=n_bins, low=low, high=high, name="Signal")
    hist_bkg1 = create_hist(data["bkg1"]["NN_output"], weights=data["bkg1"]["weight"], bins=n_bins, low=low, high=high, name="Bkg1")
    hist_bkg2 = create_hist(data["bkg2"]["NN_output"], weights=data["bkg2"]["weight"], bins=n_bins, low=low, high=high, name="Bkg2")
    hist_bkg3 = create_hist(data["bkg3"]["NN_output"], weights=data["bkg3"]["weight"], bins=n_bins, low=low, high=high, name="Bkg3")
    bkg_hists = [hist_bkg1, hist_bkg2, hist_bkg3]


    # plot the backgrounds:
    process_labels = ["Background 1", "Background 2", "Background 3"]
    signal_labels = ["Signal x 100"]

    # For demonstration, we compare three binning schemes.
    binning_options = [5, 10, 50]
    equidistant_significances = {}
    optimized_significances = {}
    
    for nbins in binning_options:
        # --- Fixed binning significance ---
        nbins_hist = hist_signal.axes[0].size
        factor = int(nbins_hist / nbins)
        hist_signal_rb = hist_signal[::hist.rebin(factor)]
        bkg_hists_rb = [h[::hist.rebin(factor)] for h in bkg_hists]
    
        Z_equidistant = compute_significance(hist_signal_rb, bkg_hists_rb)
        equidistant_significances[nbins] = Z_equidistant
        print(f"Fixed binning ({nbins} bins): Overall significance = {Z_equidistant:.3f}")


        fixed_plot_filename = f"examples/toy_example/NN_output_distribution_fixed_{nbins}bins.pdf"
        plot_stacked_histograms(
            stacked_hists=bkg_hists_rb,
            process_labels=process_labels,
            signal_hists=[hist_signal_rb * 100],
            signal_labels=signal_labels,
            output_filename=fixed_plot_filename,
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=False
        )
        plot_stacked_histograms(
            stacked_hists=bkg_hists_rb,
            process_labels=process_labels,
            signal_hists=[hist_signal_rb * 100],
            signal_labels=signal_labels,
            output_filename=fixed_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=True
        )
        print(f"Fixed binning ({nbins} bins) plot saved as {fixed_plot_filename}")

        # --- Optimization: create a model instance with n_cats = nbins ---
        opt_model = one_dimensional_binning_optimiser(n_cats=nbins, steepness=50.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        # Run a simple optimization loop.
        epochs = 100
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = opt_model.call(data)
            grads = tape.gradient(loss, opt_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, opt_model.trainable_variables))
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  [n_bins={nbins}] Epoch {epoch}: Loss = {loss.numpy():.3f}")
                print("Effective boundaries:", opt_model.get_effective_boundaries())

        # Now, rebuild optimized histograms using effective boundaries.
        eff_boundaries = opt_model.get_effective_boundaries()["NN_output"]
        print(f"Optimized boundaries for {nbins} bins: {eff_boundaries}")

        opt_bin_edges = np.concatenate(([low], np.array(eff_boundaries), [high]))
        h_signal_opt = create_hist(data["signal"]["NN_output"], weights=data["signal"]["weight"], bins=opt_bin_edges, name="Signal_opt")
        h_bkg1_opt = create_hist(data["bkg1"]["NN_output"], weights=data["bkg1"]["weight"], bins=opt_bin_edges, name="Bkg1_opt")
        h_bkg2_opt = create_hist(data["bkg2"]["NN_output"], weights=data["bkg2"]["weight"], bins=opt_bin_edges, name="Bkg2_opt")
        h_bkg3_opt = create_hist(data["bkg3"]["NN_output"], weights=data["bkg3"]["weight"], bins=opt_bin_edges, name="Bkg3_opt")
        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt, h_bkg3_opt]
        # Compute significance from these optimized histograms.

        Z_opt = compute_significance(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt
        # optimized_hists_dict[nbins] = (h_signal_opt, opt_bkg_hists)
        print(f"Optimized binning ({nbins} bins): Overall significance = {Z_opt:.3f}")

        opt_plot_filename = f"examples/toy_example/NN_output_distribution_optimized_{nbins}bins.pdf"
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename,
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=False
        )

        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=True
        )
        print(f"Optimized binning ({nbins} bins) plot saved as {opt_plot_filename}")



    # --- Comparison plot ---
    fig_comp, ax_comp = plt.subplots(figsize=(8, 6))
    x_vals = np.array(binning_options)
    Z_equidistant_vals = [equidistant_significances[nb] for nb in binning_options]
    opt_Z_vals = [optimized_significances[nb] for nb in binning_options]
    ax_comp.plot(x_vals, Z_equidistant_vals, marker='o', linestyle='-', label="Fixed (Equidistant)")
    ax_comp.plot(x_vals, opt_Z_vals, marker='s', linestyle='--', label="Optimized")
    ax_comp.set_xlabel("Number of bins", fontsize=14)
    ax_comp.set_ylabel("Overall significance", fontsize=14)
    ax_comp.set_title("Comparison: Fixed vs Optimized Binning", fontsize=16)
    ax_comp.legend(fontsize=12)
    plt.tight_layout()
    comp_plot_filename = "examples/toy_example/Significance_Comparison.pdf"
    fig_comp.savefig(comp_plot_filename)
    plt.close(fig_comp)
    print(f"Comparison plot saved as {comp_plot_filename}")

if __name__ == "__main__":
    main()
