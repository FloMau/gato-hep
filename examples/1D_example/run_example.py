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

from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history, plot_yield_vs_uncertainty, plot_significance_comparison
from diffcat_optimizer.differentiable_categories import asymptotic_significance, gato_gmm_model, low_bkg_penalty, compute_significance_from_hists, high_bkg_uncertainty_penalty
from diffcat_optimizer.utils import df_dict_to_tensors, create_hist
from diffcat_optimizer.data_generation import generate_toy_data_1D


class gato_1D(gato_gmm_model):
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
        B_sumw2 = tf.zeros([self.n_cats], dtype=tf.float32)

        for proc, t in data_dict.items():
            # x: [N] → [N,1] for the GMM
            x = tf.expand_dims(t["NN_output"], 1)
            w = t["weight"]
            w2 = t["weight"]**2

            # compute per‐component log-likelihood
            log_probs = []
            for i in range(self.n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=means[i],
                    scale_tril=scale_tril[i]
                )
                log_probs.append(dist.log_prob(x))                             # [N]
            log_probs = tf.stack(log_probs, axis=1)                            # [N,n_cats]
            # posterior memberships
            log_joint = (log_probs + log_mix)
            memberships = tf.nn.softmax(log_joint / self.temperature, axis=1)  # [N,n_cats]

            # sum into yields
            yields = tf.reduce_sum(memberships * tf.expand_dims(w,1), axis=0)  # [n_cats]
            sumw2 = tf.reduce_sum(memberships * tf.expand_dims(w2,1), axis=0)  # [n_cats]

            if proc == "signal":
                S += yields
            else:
                B += yields
                B_sumw2 += sumw2

        # Asimov per-bin + quadrature
        Z_bins = asymptotic_significance(S, B)                     # [n_cats]
        Z_tot  = tf.sqrt(tf.reduce_sum(Z_bins**2))                 # scalar

        return -Z_tot, B, B_sumw2

# ------------------------------------------------------------------------------
# Main: Generate data, run fixed binning and optimization, then compare.
# ------------------------------------------------------------------------------
def main():
    # 1. Generate toy data
    data = generate_toy_data_1D(
        # n_signal=100000,
        # n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
        n_signal=int(100000 / 50),
        n_bkg1=int(200000 / 50), n_bkg2=int(100000 / 50), n_bkg3=int(100000 / 50),
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
    gato_binning_options = [10]
    equidistant_significances = {}
    optimized_significances = {}

    path_plots = "examples/1D_example/Plots/"
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

    # --- Fixed binning significance ---
    for nbins in equidistant_binning_options:
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

    # GATO
    tensor_data = df_dict_to_tensors(data)
    for nbins in gato_binning_options:

        @tf.function
        def train_step(model, tensor_data, optimizer, lam_yield=0.0, lam_unc=0.0):
            with tf.GradientTape() as tape:
                # assume your call() now returns (loss, B, B_sumsq)
                loss, B, B_sumw2 = model.call(tensor_data)

                penalty_yield  = low_bkg_penalty(B, threshold=10.0, steepness=10.0)
                penalty_unc  = high_bkg_uncertainty_penalty(B_sumw2, B, rel_threshold=0.2)

                # you can give them different weights
                total_loss = loss + lam_yield*penalty_yield + lam_unc*penalty_unc
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, penalty_yield, penalty_unc

        # --- Optimization: create a model instance with n_cats = nbins ---
        model = gato_1D(n_cats=nbins, temperature=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9) # if lam_yield==0 else 0.5)

        loss_history = []
        penalty_yield_history = []
        penalty_unc_history = []
        boundary_history = []

        epochs = 10
        lam_yield = 0.0
        lam_unc = 10000


        for epoch in range(epochs):

            total_loss, loss, penalty_yield, penalty_unc = train_step(model, tensor_data, optimizer, lam_yield=lam_yield, lam_unc=lam_unc)

            # Save the history
            loss_history.append(loss.numpy())
            penalty_yield_history.append(penalty_yield.numpy())
            penalty_unc_history.append(penalty_unc.numpy())
            # Save the current boundaries in [0,1]
            boundaries_ = model.get_effective_boundaries_1d()
            boundary_history.append(boundaries_)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[n_bins={nbins}] Epoch {epoch}: total_loss = {total_loss.numpy():.3f}, base_loss={loss.numpy():.3f}")
                print("Effective boundaries:", model.get_effective_boundaries_1d())


        # Rebuild optimized histograms using effective boundaries
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
        regularisation_plot_name = path_plots + f"history_penalty_yield{nbins}bins.pdf"
        plot_history(
            history_data=penalty_yield_history,
            output_filename=regularisation_plot_name,
            y_label="Low bkg. penalty",
            x_label="Epoch",
            boundaries=False,
            title=f"Regularisation history (nbins={nbins})"
        )
        regularisation_plot_name = path_plots + f"history_penalty_unc_{nbins}bins.pdf"
        plot_history(
            history_data=penalty_unc_history,
            output_filename=regularisation_plot_name,
            y_label="High bkg. unc. penalty",
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

        B_sorted, rel_unc_sorted, _ = model.compute_hard_bkg_stats(tensor_data)
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc_sorted,
            output_filename=path_plots + f"yield_vs_uncertainty_{nbins}bins_sorted.pdf",
        )

    plot_significance_comparison(
        equidistant_binning_options,
        [equidistant_significances[nb] for nb in equidistant_binning_options],
        gato_binning_options,
        [optimized_significances[nb]  for nb in gato_binning_options],
        output_filename=path_plots + "significanceComparison.pdf",
    )

if __name__ == "__main__":
    main()
