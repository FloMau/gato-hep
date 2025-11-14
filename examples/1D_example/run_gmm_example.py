import argparse
from gatohep.data_generation import generate_toy_data_1D
from gatohep.losses import (
    high_bkg_uncertainty_penalty,
    low_bkg_penalty,
)
from gatohep.models import (
    gato_gmm_model,
)
from gatohep.plotting_utils import (
    plot_bias_history,
    plot_gmm_1d,
    plot_history,
    plot_significance_comparison,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
)
from gatohep.utils import (
    LearningRateScheduler,
    TemperatureScheduler,
    asymptotic_significance,
    compute_significance_from_hists,
    create_hist,
    df_dict_to_tensors,
)
import hist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

tfd = tfp.distributions


# Define a 1D GATO model for the toy example inheriting from gato_gmm_model
class gato_1D(gato_gmm_model):
    def __init__(self, n_cats, temperature=1.0, mean_norm="sigmoid"):
        super().__init__(
            n_cats=n_cats,
            dim=1,
            temperature=temperature,
            mean_norm=mean_norm,
            mean_range=(0.0, 1.0),
        )  # dummy NN output is already in (0,1)

    def call(self, data_dict):
        """
        Compute the loss for two signals vs. backgrounds
        using the generic helpers from the base class.
        """
        significances, bkg_yield, bkg_sum_w2 = self.get_differentiable_significance(
            data_dict,
            signal_labels=["signal"],
            return_details=True,
        )
        loss = -significances["signal"]
        return loss, bkg_yield, bkg_sum_w2


# main: Generate data, run fixed binning and optimization, then compare.
def main():
    parser = argparse.ArgumentParser(description="1-D GATO optimisation on toy data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="number of training epochs (default: 250)",
    )

    parser.add_argument(
        "--gato-bins",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        metavar="N",
        help="List of target bin counts for the GATO run (default: 3,5,10)",
    )

    parser.add_argument(
        "--lam-yield",
        type=float,
        default=0.0,
        help=r"lambda for the low-background-yield penalty (default: 0)",
    )

    parser.add_argument(
        "--lam-unc",
        type=float,
        default=0.0,
        help=r"lambda for the high-uncertainty penalty (default: 0)",
    )

    parser.add_argument(
        "--thr-yield",
        type=float,
        default=5.0,
        help="Threshold (events) below which the low-yield "
        "penalty turns on (default: 10)",
    )

    parser.add_argument(
        "--thr-unc",
        type=float,
        default=0.20,
        help="Relative uncertainty above which the uncertainty "
        "penalty turns on (default: 0.20)",
    )

    parser.add_argument(
        "--n-bkg",
        type=int,
        default=300000,
        help="Total number of background events to generate.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="Plots",
        help='Suffix for the output directory. Default: "Plots"',
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
        seed=42,
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
        name="Signal",
    )
    bkg_processes = [f"bkg{i}" for i in range(1, 6)]
    bkg_hists = [
        create_hist(
            data[proc]["NN_output"],
            weights=data[proc]["weight"],
            bins=n_bins,
            low=low,
            high=high,
            name=f"{proc.capitalize()}",
        )
        for proc in bkg_processes
    ]

    # plot the backgrounds:
    process_labels = [f"Background {i}" for i in range(1, len(bkg_processes) + 1)]
    signal_labels = ["Signal x 100"]

    # For demonstration, we compare multiple binning schemes.
    equidistant_binning_options = [2, 5, 10, 20]
    equidistant_significances = {}
    optimized_significances = {}

    path_plots = f"examples/1D_example/{args.out}/"
    os.makedirs(path_plots, exist_ok=True)
    fixed_plot_filename = path_plots + "toy_data.pdf"
    plot_stacked_histograms(
        stacked_hists=[bkg_hist[:: hist.rebin(30)] for bkg_hist in bkg_hists],
        process_labels=process_labels,
        signal_hists=[hist_signal[:: hist.rebin(30)] * 100],
        signal_labels=signal_labels,
        output_filename=fixed_plot_filename,
        axis_labels=("Toy discriminant", "Events"),
        normalize=False,
        log=False,
    )
    plot_stacked_histograms(
        stacked_hists=[bkg_hist[:: hist.rebin(30)] for bkg_hist in bkg_hists],
        process_labels=process_labels,
        signal_hists=[hist_signal[:: hist.rebin(30)] * 100],
        signal_labels=signal_labels,
        output_filename=fixed_plot_filename.replace(".pdf", "_log.pdf"),
        axis_labels=("Toy discriminant", "Events"),
        normalize=False,
        log=True,
    )

    # Fixed binning significance
    for nbins in equidistant_binning_options:
        nbins_hist = hist_signal.axes[0].size
        factor = int(nbins_hist / nbins)
        hist_signal_rb = hist_signal[:: hist.rebin(factor)]
        bkg_hists_rb = [h[:: hist.rebin(factor)] for h in bkg_hists]

        Z_equidistant = compute_significance_from_hists(hist_signal_rb, bkg_hists_rb)
        equidistant_significances[nbins] = Z_equidistant
        print(
            f"Fixed binning ({nbins} bins): Overall significance = {Z_equidistant:.3f}"
        )

        fixed_plot_filename = (
            path_plots + f"NN_output_distribution_fixed_{nbins}bins.pdf"
        )
        plot_stacked_histograms(
            stacked_hists=bkg_hists_rb,
            process_labels=process_labels,
            signal_hists=[hist_signal_rb * 100],
            signal_labels=signal_labels,
            output_filename=fixed_plot_filename,
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=False,
        )
        plot_stacked_histograms(
            stacked_hists=bkg_hists_rb,
            process_labels=process_labels,
            signal_hists=[hist_signal_rb * 100],
            signal_labels=signal_labels,
            output_filename=fixed_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Toy NN output", "Toy events"),
            normalize=False,
            log=True,
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
            rel_threshold_unc=0.2,
        ):
            with tf.GradientTape() as tape:
                # assume your call() now returns (loss, B, B_sumsq)
                loss, B, B_sumw2 = model.call(tensor_data)

                penalty_yield = low_bkg_penalty(B, threshold=threshold_yield)
                penalty_unc = high_bkg_uncertainty_penalty(
                    B_sumw2, B, rel_threshold=rel_threshold_unc
                )
                total_loss = loss + lam_yield * penalty_yield + lam_unc * penalty_unc
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, penalty_yield, penalty_unc

        # --- Optimization: create a model instance with n_cats = nbins ---
        model = gato_1D(n_cats=nbins, temperature=1.0)

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
        lr_scheduler = LearningRateScheduler(
            optimizer,
            lr_initial=0.1,
            lr_final=0.001,
            total_epochs=epochs,
            mode="cosine",
        )

        # temperature scheduler
        temperature_scheduler = TemperatureScheduler(
            model,
            t_initial=1.0,
            t_final=0.05,
            total_epochs=args.epochs,
            mode="cosine",
        )

        loss_history = []
        penalty_yield_history = []
        penalty_unc_history = []
        mean_bias_history = []
        bias_epochs = []
        temperature_history = []
        for epoch in range(epochs):
            lr_scheduler.update(epoch)
            temperature_scheduler.update(epoch)

            total_loss, loss, penalty_yield, penalty_unc = train_step(
                model,
                tensor_data,
                optimizer,
                lam_yield=lam_yield,
                lam_unc=lam_unc,
                threshold_yield=yield_thr,
                rel_threshold_unc=unc_thr,
            )

            # Save the history
            loss_history.append(loss.numpy())
            penalty_yield_history.append(penalty_yield.numpy())
            penalty_unc_history.append(penalty_unc.numpy())
            if epoch % 25 == 0 or epoch == epochs - 1:
                bias_vec = model.get_bias(tensor_data)
                mean_bias_history.append(float(np.mean(np.abs(bias_vec))))
                bias_epochs.append(epoch)
                temperature_history.append(float(model.temperature))

            if epoch % 10 == 0 or epoch == epochs - 1:
                lr_value = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
                if hasattr(lr_value, "numpy"):
                    lr_value = float(lr_value.numpy())
                else:
                    lr_value = float(lr_value)
                print(
                    f"[n_bins={nbins}] Epoch {epoch}: total_loss = {total_loss.numpy():.3f}, "
                    f"base_loss = {loss.numpy():.3f}, lr = {lr_value:.5f}"
                )
                print("Effective boundaries:", model.get_effective_boundaries_1d())
        # save the trained GATO model
        checkpoint_dir = os.path.join(
            path_plots, "checkpoints", f"{nbins}_bins"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(checkpoint_dir)
        # Rebuild optimized histograms using effective boundaries
        eff_boundaries = model.get_effective_boundaries_1d()
        print(f"Optimized boundaries for {nbins} bins: {eff_boundaries}")

        # check bias due to finite temperature in training
        bias = model.get_bias(tensor_data)
        print(f"T = {model.temperature:4.2f};  per-bin bias: {bias}")

        opt_bin_edges = np.concatenate(([low], np.array(eff_boundaries), [high]))
        h_signal_opt = create_hist(
            data["signal"]["NN_output"],
            weights=data["signal"]["weight"],
            bins=opt_bin_edges,
            name="Signal_opt",
        )
        opt_bkg_hists = [
            create_hist(
                data[proc]["NN_output"],
                weights=data[proc]["weight"],
                bins=opt_bin_edges,
                name=f"{proc}_opt",
            )
            for proc in bkg_processes
        ]

        # Compute significance from these optimized histograms.
        Z_opt = compute_significance_from_hists(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt

        print(f"Optimized binning ({nbins} bins): Overall significance = {Z_opt:.3f}")

        bias_plot_base = path_plots + f"bias_history_{nbins}bins"
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + ".pdf",
            epochs=bias_epochs,
            temp_points=temperature_history,
            temp_label="Temperature",
        )
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + "_log.pdf",
            epochs=bias_epochs,
            temp_points=temperature_history,
            temp_label="Temperature",
            log_scale=True,
        )

        opt_plot_filename = (
            path_plots + f"NN_output_distribution_optimized_{nbins}bins.pdf"
        )
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename,
            axis_labels=("Toy NN output", "Events"),
            normalize=False,
            log=False,
        )

        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=process_labels,
            signal_hists=[h_signal_opt * 100],
            signal_labels=signal_labels,
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Toy NN output", "Events"),
            normalize=False,
            log=True,
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
            output_filename=path_plots
            + f"yield_vs_uncertainty_{nbins}bins_sorted_log.pdf",
        )
        # plot the learned GMM in 1D
        plot_gmm_1d(
            model,
            output_filename=os.path.join(path_plots, f"gmm_components_{nbins}bins.pdf"),
            x_range=(low, high),
            n_points=1000,
        )

    plot_significance_comparison(
        {"": {nb: equidistant_significances[nb] for nb in equidistant_binning_options}},
        {"": {nb: optimized_significances[nb] for nb in gato_binning_options}},
        output_filename=path_plots + "significanceComparison.pdf",
    )


if __name__ == "__main__":
    main()
