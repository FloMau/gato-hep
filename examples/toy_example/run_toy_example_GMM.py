import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf
import hist
import tensorflow_probability as tfp
tfd = tfp.distributions

# Append the repo root to sys.path so that we can import our core modules.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history
from diffcat_optimizer.differentiable_categories import asymptotic_significance, DiffCatModelMultiDimensional, low_bkg_penalty, compute_significance_from_hists
from diffcat_optimizer.utils import df_dict_to_tensors, create_hist
from generate_toy_data import generate_toy_data_gauss


class one_dimensional_binning_optimiser(DiffCatModelMultiDimensional):
    def __init__(self, n_cats, temperature=1.0, name="el_gato"):
        super().__init__(
            # variables_config=[{"name":"NN_output", "n_cats":n_cats}],
            n_cats=n_cats,
            dim=1,
            temperature=temperature,
            name=name
        )

    def call(self, data_dict):
        # pull out our params
        log_mix    = tf.nn.log_softmax(self.mixture_logits)        # [n_cats]
        scale_tril = self.get_scale_tril()                         # [n_cats,1,1]
        means      = tf.math.sigmoid(self.means)                   # [n_cats,1]

        # accumulators
        S = tf.zeros([self.n_cats], dtype=tf.float32)
        B = tf.zeros([self.n_cats], dtype=tf.float32)

        for proc, t in data_dict.items():
            # x: [N] → [N,1] for the MVN
            x = tf.expand_dims(t["NN_output"], 1)
            w = t["weight"]                                          # [N]

            # compute per‐component log-likelihood
            log_probs = []
            for i in range(self.n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=means[i],
                    scale_tril=scale_tril[i]
                )
                log_probs.append(dist.log_prob(x))                   # [N]
            log_probs = tf.stack(log_probs, axis=1)                 # [N,n_cats]
            # posterior memberships
            log_joint       = (log_probs + log_mix)
            memberships = tf.nn.softmax(log_joint / self.temperature, axis=1)             # [N,n_cats]

            # sum into yields
            yields = tf.reduce_sum(memberships * tf.expand_dims(w,1), axis=0)  # [n_cats]

            if proc == "signal":
                S += yields
            else:
                B += yields

        # Asimov per-bin + quadrature
        Z_bins = asymptotic_significance(S, B)                     # [n_cats]
        Z_tot  = tf.sqrt(tf.reduce_sum(Z_bins**2))                 # scalar

        return -Z_tot, B

# ------------------------------------------------------------------------------
# Main: Generate data, run fixed binning and optimization, then compare.
# ------------------------------------------------------------------------------
def main():
    # 1. Generate toy data
    data = generate_toy_data_gauss(
        n_signal=100000,
        n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
        xs_signal=0.5,    # 500 fb = 0.5 pb
        xs_bkg1=50, xs_bkg2=15, xs_bkg3=10,
        lumi=100,         # in /fb
        seed=42
    )

    # Create fixed histograms (with equidistant binning using 1500 bins as baseline, will be rebinned afterwards).
    n_bins = 1500
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

    # For demonstration, we compare multiple binning schemes.
    equidistant_binning_options = [2, 5, 10, 20]
    gato_binning_options = [2, 5, 10]
    equidistant_significances = {}
    optimized_significances = {}

    path_plots = "examples/toy_example/PlotsGMM/"
    os.makedirs(path_plots, exist_ok=True)
    fixed_plot_filename = path_plots + f"toy_data.pdf"
    plot_stacked_histograms(
        stacked_hists=[bkg_hist[::hist.rebin(30)] for bkg_hist in bkg_hists],
        process_labels=process_labels,
        signal_hists=[hist_signal[::hist.rebin(30)] * 100],
        signal_labels=signal_labels,
        output_filename=fixed_plot_filename,
        axis_labels=("Toy discriminant", "Events"),
        normalize=False,
        log=False
    )
    plot_stacked_histograms(
        stacked_hists=[bkg_hist[::hist.rebin(30)] for bkg_hist in bkg_hists],
        process_labels=process_labels,
        signal_hists=[hist_signal[::hist.rebin(30)] * 100],
        signal_labels=signal_labels,
        output_filename=fixed_plot_filename.replace(".pdf", "_log.pdf"),
        axis_labels=("Toy discriminant", "Events"),
        normalize=False,
        log=True
    )


    for nbins in equidistant_binning_options:
        # --- Fixed binning significance ---
        nbins_hist = hist_signal.axes[0].size
        factor = int(nbins_hist / nbins)
        hist_signal_rb = hist_signal[::hist.rebin(factor)]
        bkg_hists_rb = [h[::hist.rebin(factor)] for h in bkg_hists]

        Z_equidistant = compute_significance_from_hists(hist_signal_rb, bkg_hists_rb)
        equidistant_significances[nbins] = Z_equidistant
        print(f"Fixed binning ({nbins} bins): Overall significance = {Z_equidistant:.3f}")

        fixed_plot_filename = path_plots + f"NN_output_distribution_fixed_{nbins}bins.pdf"
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

    for nbins in gato_binning_options:

        # --- 3) your training step stays pure tf.function ---
        @tf.function
        def train_step(model, tensor_data, optimizer, lam=0.0):
            with tf.GradientTape() as tape:
                loss, B = model.call(tensor_data)
                penalty = low_bkg_penalty(B, threshold=10, steepness=10)
                total_loss = loss + lam * penalty
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, penalty

        # --- Optimization: create a model instance with n_cats = nbins ---
        model = one_dimensional_binning_optimiser(n_cats=nbins, temperature=0.1)
        lam = 0.05
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9 if lam==0 else 0.5)

        loss_history = []
        regularisation_history = []
        boundary_history = []
        epochs = 350

        tensor_data = df_dict_to_tensors(data)

        for epoch in range(epochs):

            total_loss, loss, penalty = train_step(model, tensor_data, optimizer, lam=lam)

            # Save the history
            loss_history.append(loss.numpy())
            regularisation_history.append(penalty.numpy())
            # Save the current boundaries in [0,1]
            boundaries_ = model.get_effective_boundaries_1d()
            boundary_history.append(boundaries_)

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[n_bins={nbins}] Epoch {epoch}: total_loss = {total_loss.numpy():.3f}, base_loss={loss.numpy():.3f}")
                print("Effective boundaries:", model.get_effective_boundaries_1d())


        # Now, rebuild optimized histograms using effective boundaries.
        eff_boundaries = model.get_effective_boundaries_1d()
        print(f"Optimized boundaries for {nbins} bins: {eff_boundaries}")

        opt_bin_edges = np.concatenate(([low], np.array(eff_boundaries), [high]))
        h_signal_opt = create_hist(data["signal"]["NN_output"], weights=data["signal"]["weight"], bins=opt_bin_edges, name="Signal_opt")
        h_bkg1_opt = create_hist(data["bkg1"]["NN_output"], weights=data["bkg1"]["weight"], bins=opt_bin_edges, name="Bkg1_opt")
        h_bkg2_opt = create_hist(data["bkg2"]["NN_output"], weights=data["bkg2"]["weight"], bins=opt_bin_edges, name="Bkg2_opt")
        h_bkg3_opt = create_hist(data["bkg3"]["NN_output"], weights=data["bkg3"]["weight"], bins=opt_bin_edges, name="Bkg3_opt")
        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt, h_bkg3_opt]
        # Compute significance from these optimized histograms.

        Z_opt = compute_significance_from_hists(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt
        # optimized_hists_dict[nbins] = (h_signal_opt, opt_bkg_hists)
        print(f"Optimized binning ({nbins} bins): Overall significance = {Z_opt:.3f}")

        opt_plot_filename = path_plots + f"NN_output_distribution_optimized_{nbins}bins.pdf"
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename,
            axis_labels=("Toy NN output", "Events"),
            normalize=False,
            log=False
        )

        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Toy NN output", "Events"),
            normalize=False,
            log=True
        )
        print(f"Optimized binning ({nbins} bins) plot saved as {opt_plot_filename}")

        # --- Now plot the history using your single function:
        # Plot the loss
        loss_plot_name = path_plots + f"history_loss_{nbins}bins.pdf"
        plot_history(
            history_data=loss_history,
            output_filename=loss_plot_name,
            y_label="Negative significance",
            x_label="Epoch",
            boundaries=False,
            title=f"Loss history (nbins={nbins})"
        )
        regularisation_plot_name = path_plots + f"history_penalty_{nbins}bins.pdf"
        plot_history(
            history_data=regularisation_history,
            output_filename=regularisation_plot_name,
            y_label="Low bkg. penalty",
            x_label="Epoch",
            boundaries=False,
            title=f"Regularisation history (nbins={nbins})"
        )

        # Plot the boundary evolution
        bndry_plot_name = path_plots + f"history_boundaries_{nbins}bins.pdf"
        plot_history(
            history_data=boundary_history,
            output_filename=bndry_plot_name,
            y_label="Boundary position",
            x_label="Epoch",
            boundaries=True,
            title=f"Boundary evolution (nbins={nbins})"
        )

    # --- Comparison plot ---
    fig_comp, ax_comp = plt.subplots(figsize=(8, 6))
    Z_equidistant_vals = [equidistant_significances[nb] for nb in equidistant_binning_options]
    opt_Z_vals = [optimized_significances[nb] for nb in gato_binning_options]
    ax_comp.plot(equidistant_binning_options, Z_equidistant_vals, marker='o', linestyle='-', label="Equidistant binning")
    ax_comp.plot(gato_binning_options, opt_Z_vals, marker='s', linestyle='--', label="GATO binning")
    ax_comp.set_xlabel("Number of bins", fontsize=22)
    ax_comp.set_ylabel("Overall significance", fontsize=22)
    ax_comp.legend(fontsize=18)
    ax_comp.set_xlim(0, ax_comp.get_xlim()[1])
    ax_comp.set_ylim(0, ax_comp.get_ylim()[1])
    plt.tight_layout()
    comp_plot_filename = path_plots + "significanceComparison.pdf"
    fig_comp.savefig(comp_plot_filename)
    plt.close(fig_comp)
    print(f"Comparison plot saved as {comp_plot_filename}")

if __name__ == "__main__":
    main()
