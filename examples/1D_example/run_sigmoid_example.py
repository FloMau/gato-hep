import argparse
from gatohep.data_generation import generate_toy_data_1D
from gatohep.losses import (
    high_bkg_uncertainty_penalty,
    low_bkg_penalty,
)
from gatohep.models import gato_sigmoid_model
from gatohep.plotting_utils import (
    plot_bias_history,
    plot_history,
    plot_significance_comparison,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
)
from gatohep.utils import (
    LearningRateScheduler,
    SteepnessScheduler,
    asymptotic_significance,
    compute_significance_from_hists,
    create_hist,
    df_dict_to_tensors,
)
import hist
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# Define the 1D GATO model using sigmoid boundaries
class gato_1D(gato_sigmoid_model):
    """
    One-dimensional GATO model that uses monotonic sigmoid boundaries.

    The class replicates the call/significance logic of the 1-D GMM example
    so the training script needs only to replace the model import.

    Parameters
    ----------
    n_cats : int
        Number of bins (Gaussian components in the GMM analogue).
    steepness : float, optional
        Initial slope k of the sigmoids.  Can be annealed externally.
    """

    def __init__(self, n_cats: int, *, steepness: float = 50.0):
        variables = [
            {"name": "NN_output", "bins": n_cats, "range": (0.0, 1.0)},
        ]
        super().__init__(variables_config=variables, global_steepness=steepness)

    def call(self, data_dict):
        """
        Compute the loss and background yields for one optimisation step.

        The loss is  -sqrt(Z1 * Z2)  where Z1, Z2 are the per-signal
        significances obtained with asymptotic formulae.

        Parameters
        ----------
        data_dict : dict
            Mapping from process name to a dictionary with keys
            ``"NN_output"`` (tensor, shape Nx1) and ``"weight"`` (tensor).

        Returns
        -------
        loss : tf.Tensor  scalar
            Negative combined significance (to minimise).
        bkg_y : tf.Tensor  (n_cats,)
            Background yields per bin.
        bkg_w2 : tf.Tensor (n_cats,)
            Sum of squared weights per bin for the background.
        """
        gamma = self.get_probs(data_dict)  # soft assignments

        sig_yield = tf.zeros(self.n_cats, tf.float32)
        bkg_yield = tf.zeros(self.n_cats, tf.float32)
        bkg_sumw2 = tf.zeros(self.n_cats, tf.float32)

        for proc, g in gamma.items():
            w = data_dict[proc]["weight"]
            w2 = w**2

            y = tf.reduce_sum(g * w[:, None], axis=0)
            y2 = tf.reduce_sum(g * w2[:, None], axis=0)

            if proc.startswith("signal"):
                sig_yield += y
            else:
                bkg_yield += y
                bkg_sumw2 += y2

        # Asimov per-bin + quadrature
        loss = -tf.sqrt(
            tf.reduce_sum(asymptotic_significance(sig_yield, bkg_yield) ** 2)
        )
        return loss, bkg_yield, bkg_sumw2


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
        default="PlotsSigmoidModel",
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
        xs_signal=0.5,
        xs_bkg1=100,
        xs_bkg2=80,
        xs_bkg3=50,  # in pb
        lumi=100,  # in /fb
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
    hist_bkg1 = create_hist(
        data["bkg1"]["NN_output"],
        weights=data["bkg1"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg1",
    )
    hist_bkg2 = create_hist(
        data["bkg2"]["NN_output"],
        weights=data["bkg2"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg2",
    )
    hist_bkg3 = create_hist(
        data["bkg3"]["NN_output"],
        weights=data["bkg3"]["weight"],
        bins=n_bins,
        low=low,
        high=high,
        name="Bkg3",
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
        model = gato_1D(n_cats=nbins, steepness=50.0)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.5, rho=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.9, nesterov=True)

        lr_scheduler = LearningRateScheduler(
            optimizer,
            lr_initial=0.5,
            lr_final=0.001,
            total_epochs=epochs,
            mode="cosine",
        )

        # steepness scheduler
        steepness_scheduler = SteepnessScheduler(
            model,
            t_initial=50.0,  # starting k
            t_final=10000.0,  # final k
            total_epochs=args.epochs,
            mode="cosine",  # or "exponential"
        )

        loss_history = []
        penalty_yield_history = []
        penalty_unc_history = []
        boundary_history = []
        mean_bias_history = []
        bias_epochs = []
        steepness_history = []
        for epoch in range(epochs):
            lr_scheduler.update(epoch)
            steepness_scheduler.update(epoch)  # adjust all k_j in-place

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
            boundary_history.append(
                model.calculate_boundaries().numpy()
            )  # list(ndarray)
            if epoch % 25 == 0 or epoch == epochs - 1:
                bias_vec = model.get_bias(tensor_data)
                mean_bias_history.append(float(np.mean(np.abs(bias_vec))))
                bias_epochs.append(epoch)
                steepness_history.append(float(model.var_cfg[0]["k"].numpy()))

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
                print("Effective boundaries:", model.calculate_boundaries())
        # save the trained GATO model
        checkpoint_dir = os.path.join(path_plots, f"checkpoints/{nbins}_bins")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(checkpoint_dir)
        # Rebuild optimized histograms using effective boundaries
        eff_boundaries = model.calculate_boundaries()
        print(f"Optimized boundaries for {nbins} bins: {eff_boundaries}")

        # check bias due to finite temperature in training
        bias = model.get_bias(tensor_data)
        print(f"Steepness = {float(model.var_cfg[0]['k'].numpy()):.1f};  per-bin bias: {bias}")

        opt_bin_edges = np.concatenate(([low], np.array(eff_boundaries), [high]))
        h_signal_opt = create_hist(
            data["signal"]["NN_output"],
            weights=data["signal"]["weight"],
            bins=opt_bin_edges,
            name="Signal_opt",
        )
        h_bkg1_opt = create_hist(
            data["bkg1"]["NN_output"],
            weights=data["bkg1"]["weight"],
            bins=opt_bin_edges,
            name="Bkg1_opt",
        )
        h_bkg2_opt = create_hist(
            data["bkg2"]["NN_output"],
            weights=data["bkg2"]["weight"],
            bins=opt_bin_edges,
            name="Bkg2_opt",
        )
        h_bkg3_opt = create_hist(
            data["bkg3"]["NN_output"],
            weights=data["bkg3"]["weight"],
            bins=opt_bin_edges,
            name="Bkg3_opt",
        )
        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt, h_bkg3_opt]

        # Compute significance from these optimized histograms.
        Z_opt = compute_significance_from_hists(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt

        print(f"Optimized binning ({nbins} bins): Overall significance = {Z_opt:.3f}")

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

        bias_plot_base = path_plots + f"bias_history_{nbins}bins"
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + ".pdf",
            epochs=bias_epochs,
            temp_points=steepness_history,
            temp_label="Steepness",
        )
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + "_log.pdf",
            epochs=bias_epochs,
            temp_points=steepness_history,
            temp_label="Steepness",
            log_scale=True,
        )

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

        boundary_plot_name = path_plots + f"history_boundaries_{nbins}bins.pdf"
        plot_history(
            history_data=boundary_history,
            output_filename=boundary_plot_name,
            y_label="Boundary position",
            x_label="Epoch",
            boundaries=True,
        )

        (
            B,
            rel_unc,
        ) = model.compute_hard_bkg_stats(tensor_data)
        plot_yield_vs_uncertainty(
            B,
            rel_unc,
            output_filename=path_plots + f"yield_vs_uncertainty_{nbins}bins_sorted.pdf",
        )
        plot_yield_vs_uncertainty(
            B,
            rel_unc,
            log=True,
            output_filename=path_plots
            + f"yield_vs_uncertainty_{nbins}bins_sorted_log.pdf",
        )

    plot_significance_comparison(
        {"": {nb: equidistant_significances[nb] for nb in equidistant_binning_options}},
        {"": {nb: optimized_significances[nb] for nb in gato_binning_options}},
        output_filename=path_plots + "significanceComparison.pdf",
    )


if __name__ == "__main__":
    main()
