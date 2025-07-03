import argparse
import os

import hist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Import from the installed/packaged gato modules
from gatohep.data_generation import generate_toy_data_1D
from gatohep.losses import (
    high_bkg_uncertainty_penalty,
    low_bkg_penalty,
)
from gatohep.models import (
    gato_gmm_model,
)
from gatohep.plotting_utils import (
    plot_gmm_1d,  # if your 1D example uses this helper
    plot_history,
    plot_significance_comparison,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
)
from gatohep.utils import (
    align_boundary_tracks,  # if this helper was in utils
    asymptotic_significance,
    compute_significance_from_hists,
    create_hist,
    df_dict_to_tensors,  # if used in your 1D script
)


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
        log_mix = tf.nn.log_softmax(self.mixture_logits)        # [n_cats]
        scale_tril = self.get_scale_tril()                         # [n_cats,1,1]
        means = tf.math.sigmoid(self.means)                   # [n_cats,1]

        # accumulators
        S = tf.zeros([self.n_cats], dtype=tf.float32)
        B = tf.zeros([self.n_cats], dtype=tf.float32)
        B_sumw2 = tf.zeros([self.n_cats], dtype=tf.float32)

        for proc, t in data_dict.items():
            # x: [N] -> [N,1] for the GMM
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
                log_probs.append(dist.log_prob(x))   # [N]
            log_probs = tf.stack(log_probs, axis=1)  # [N,n_cats]
            # posterior memberships
            log_joint = (log_probs + log_mix)
            memberships = tf.nn.softmax(
                log_joint / self.temperature, axis=1
            )  # [N,n_cats]

            # sum into yields
            yields = tf.reduce_sum(
                memberships * tf.expand_dims(w,1), axis=0
            )  # [n_cats]
            sumw2 = tf.reduce_sum(
                memberships * tf.expand_dims(w2,1), axis=0
            )  # [n_cats]

            if proc == "signal":
                S += yields
            else:
                B += yields
                B_sumw2 += sumw2

        # Asimov per-bin + quadrature
        Z_bins = asymptotic_significance(S, B)        # [n_cats]
        Z_tot = tf.sqrt(tf.reduce_sum(Z_bins**2))     # scalar

        return -Z_tot, B, B_sumw2


# main: Generate data, run fixed binning and optimization, then compare.
def main():

    parser = argparse.ArgumentParser(description="1-D GATO optimisation on toy data")
    parser.add_argument("--epochs",
                        type=int,
                        default=250,
                        help="number of training epochs (default: 250)")

    parser.add_argument(
        "--gato-bins",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        metavar="N",
        help="List of target bin counts for the GATO run (default: 3,5,10)"
    )

    parser.add_argument(
        "--lam-yield",
        type=float,
        default=0.0,
        help=r"lambda for the low-background-yield penalty (default: 0)"
    )

    parser.add_argument(
        "--lam-unc",
        type=float,
        default=0.0,
        help=r"lambda for the high-uncertainty penalty (default: 0)"
    )

    parser.add_argument(
        "--thr-yield",
        type=float,
        default=5.0,
        help="Threshold (events) below which the low-yield "
        "penalty turns on (default: 10)"
    )

    parser.add_argument(
        "--thr-unc",
        type=float,
        default=0.20,
        help="Relative uncertainty above which the uncertainty "
        "penalty turns on (default: 0.20)"
    )

    parser.add_argument(
        "--n-bkg",
        type=int,
        default=300000,
        help="Total number of background events to generate."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="Plots",
        help="Suffix for the output directory. Default: \"Plots\""
    )

    args = parser.parse_args()

    gato_binning_options = args.gato_bins
    epochs = args.epochs
    lam_yield = args.lam_yield
    lam_unc = args.lam_unc
    yield_thr = args.thr_yield
    unc_thr = args.thr_unc
    n_bkg = args.n_bkg

    # 1. Generate toy data
    data = generate_toy_data_1D(
        n_signal=100000,
        n_bkg=n_bkg,
        xs_signal=0.5, xs_bkg1=50, xs_bkg2=15, xs_bkg3=10,  # in pb
        lumi=100,  # in /fb
        seed=42
    )

    # Create fixed histograms, will be rebinned afterwards
    n_bins = 1500
    low = 0.0
    high = 1.0
    hist_signal = create_hist(
        data["signal"]["NN_output"],
        weights=data["signal"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Signal"
    )
    hist_bkg1 = create_hist(
        data["bkg1"]["NN_output"],
        weights=data["bkg1"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg1"
    )
    hist_bkg2 = create_hist(
        data["bkg2"]["NN_output"],
        weights=data["bkg2"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg2"
    )
    hist_bkg3 = create_hist(
        data["bkg3"]["NN_output"],
        weights=data["bkg3"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg3"
    )
    bkg_hists = [hist_bkg1, hist_bkg2, hist_bkg3]

    # plot the backgrounds:
    process_labels = ["Background 1", "Background 2", "Background 3"]
    signal_labels = ["Signal x 100"]

    # For demonstration, we compare multiple binning schemes.
    equidistant_binning_options = [2, 5, 10, 20]
    equidistant_significances = {}
    optimized_significances = {}

    path_plots = f"examples/1D_example/{args.out}/"
    os.makedirs(path_plots, exist_ok=True)
    fixed_plot_filename = path_plots + "toy_data.pdf"
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
        print(
            f"Fixed binning ({nbins} bins): Overall significance = {Z_equidistant:.3f}"
        )

        fixed_plot_filename = path_plots + \
            f"NN_output_distribution_fixed_{nbins}bins.pdf"
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

    # GATO part
    tensor_data = df_dict_to_tensors(data)
    for nbins in gato_binning_options:

        @tf.function
        def train_step(
            model,
            tensor_data,
            optimizer,
            lam_yield=0.0,
            lam_unc=0.0,
            threshold_yield=5,
            rel_threshold_unc=0.2
        ):
            with tf.GradientTape() as tape:
                # assume your call() now returns (loss, B, B_sumsq)
                loss, B, B_sumw2 = model.call(tensor_data)

                penalty_yield = low_bkg_penalty(B, threshold=threshold_yield)
                penalty_unc = high_bkg_uncertainty_penalty(
                    B_sumw2, B, rel_threshold=rel_threshold_unc
                )

                # you can give them different weights
                total_loss = loss + lam_yield*penalty_yield + lam_unc*penalty_unc
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, penalty_yield, penalty_unc

        # --- Optimization: create a model instance with n_cats = nbins ---
        model = gato_1D(n_cats=nbins, temperature=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9)

        loss_history = []
        penalty_yield_history = []
        penalty_unc_history = []
        boundary_history = []
        boundary_history_raw = []

        for epoch in range(epochs):

            total_loss, loss, penalty_yield, penalty_unc = train_step(
                model, tensor_data, optimizer, lam_yield=lam_yield,
                lam_unc=lam_unc, threshold_yield=yield_thr, rel_threshold_unc=unc_thr
            )

            # Save the history
            loss_history.append(loss.numpy())
            penalty_yield_history.append(penalty_yield.numpy())
            penalty_unc_history.append(penalty_unc.numpy())
            # Save the current boundaries in [0,1]
            # cuts, order = model.get_effective_boundaries_1d(return_mapping=True)
            # boundary_history.append((cuts, order))
            raw_cuts = model.get_effective_boundaries_1d()     # list(float)
            boundary_history_raw.append(raw_cuts)                # keep raw

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"[n_bins={nbins}] Epoch {epoch}: \
                    total_loss = {total_loss.numpy():.3f}, \
                    base_loss={loss.numpy():.3f}"
                )
                print("Effective boundaries:", model.get_effective_boundaries_1d())
            # save the trained GATO model
        checkpoint_dir = os.path.join(path_plots, f"checkpoints/{nbins}_bins")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(checkpoint_dir)
        # Rebuild optimized histograms using effective boundaries
        eff_boundaries = model.get_effective_boundaries_1d()
        print(f"Optimized boundaries for {nbins} bins: {eff_boundaries}")

        opt_bin_edges = np.concatenate(([low], np.array(eff_boundaries), [high]))
        h_signal_opt = create_hist(
            data["signal"]["NN_output"],
            weights=data["signal"]["weight"],
            bins=opt_bin_edges,
            name="Signal_opt"
        )
        h_bkg1_opt = create_hist(
            data["bkg1"]["NN_output"],
            weights=data["bkg1"]["weight"],
            bins=opt_bin_edges,
            name="Bkg1_opt"
        )
        h_bkg2_opt = create_hist(
            data["bkg2"]["NN_output"],
            weights=data["bkg2"]["weight"],
            bins=opt_bin_edges,
            name="Bkg2_opt"
        )
        h_bkg3_opt = create_hist(
            data["bkg3"]["NN_output"],
            weights=data["bkg3"]["weight"],
            bins=opt_bin_edges,
            name="Bkg3_opt"
        )
        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt, h_bkg3_opt]

        # Compute significance from these optimized histograms.
        Z_opt = compute_significance_from_hists(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt

        print(f"Optimized binning ({nbins} bins): Overall significance = {Z_opt:.3f}")

        opt_plot_filename = path_plots + \
            f"NN_output_distribution_optimized_{nbins}bins.pdf"
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
        )
        regularisation_plot_name = path_plots + f"history_penalty_yield{nbins}bins.pdf"
        plot_history(
            history_data=penalty_yield_history,
            output_filename=regularisation_plot_name,
            y_label="Low bkg. penalty",
            x_label="Epoch",
            boundaries=False,
        )
        regularisation_plot_name = path_plots + f"history_penalty_unc_{nbins}bins.pdf"
        plot_history(
            history_data=penalty_unc_history,
            output_filename=regularisation_plot_name,
            y_label="High bkg. unc. penalty",
            x_label="Epoch",
            boundaries=False,
        )

        boundary_history = align_boundary_tracks(
            boundary_history_raw, dist_tol=0.05
        )   # ndarray (epochs, n_tracks)
        boundary_plot_name = path_plots + f"history_boundaries_{nbins}bins.pdf"
        plot_history(
            history_data=boundary_history,
            output_filename=boundary_plot_name,
            y_label="Boundary position",
            x_label="Epoch",
            boundaries=True,
        )

        B_sorted, rel_unc_sorted, _ = model.compute_hard_bkg_stats(tensor_data)
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc_sorted,
            output_filename=path_plots + f"yield_vs_uncertainty_{nbins}bins_sorted.pdf",
        )
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc_sorted,
            log=True,
            output_filename=path_plots +
            f"yield_vs_uncertainty_{nbins}bins_sorted_log.pdf",
        )
        # plot the learned GMM in 1D
        plot_gmm_1d(
            model,
            output_filename=os.path.join(path_plots, f"gmm_components_{nbins}bins.pdf"),
            x_range=(low, high),
            n_points=1000
        )

    plot_significance_comparison(
        {"": {nb: equidistant_significances[nb] for nb in equidistant_binning_options}},
        {"": {nb: optimized_significances[nb] for nb in gato_binning_options}},
        output_filename=path_plots + "significanceComparison.pdf",
    )


if __name__ == "__main__":
    main()
