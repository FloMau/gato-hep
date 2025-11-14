import argparse
import os

import numpy as np
import tensorflow as tf

from gatohep.losses import high_bkg_uncertainty_penalty, low_bkg_penalty
from gatohep.models import gato_gmm_model
from gatohep.plotting_utils import (
    make_gif,
    plot_bias_history,
    plot_bin_boundaries_2D,
    plot_category_mass_spectra,
    plot_history,
    plot_inclusive_mass,
    plot_yield_vs_uncertainty,
)
from gatohep.utils import (
    LearningRateScheduler,
    TemperatureScheduler,
    build_category_mass_maps,
    compute_mass_reweight_factors,
    convert_mass_data_to_tensors,
    generate_resonance_toy_data,
    slice_to_2d_features,
)


class DiphotonSoftmax(gato_gmm_model):
    def __init__(self, n_cats, temperature=0.5, mass_sigma=1.5):
        super().__init__(
            n_cats=n_cats,
            dim=2,
            temperature=temperature,
            mean_norm="softmax",
            name="gato_diphoton",
        )
        self.mass_center = tf.constant(125.0, dtype=tf.float32)
        self.mass_sigma = tf.constant(float(mass_sigma), dtype=tf.float32)
        self.mass_sig_low = self.mass_center - self.mass_sigma
        self.mass_sig_high = self.mass_center + self.mass_sigma

    def call(self, data_dict, reweight=None, reweight_processes=None):
        masked = {}
        for proc, tensors in data_dict.items():
            weights = tensors["weight"]
            if proc in ("signal1", "signal2"):
                masses = tensors["mass"]
                window_mask = tf.cast(
                    tf.logical_and(
                        masses >= self.mass_sig_low, masses <= self.mass_sig_high
                    ),
                    tf.float32,
                )
                weights = weights * window_mask
            masked[proc] = {
                "NN_output": tensors["NN_output"],
                "weight": weights,
            }

        significances, bkg_yield, bkg_sum_w2 = self.get_differentiable_significance(
            masked,
            signal_labels=["signal1", "signal2"],
            background_reweight=reweight,
            reweight_processes=reweight_processes,
            return_details=True,
        )
        z1 = significances["signal1"]
        z2 = significances["signal2"]
        loss = -tf.sqrt(z1 * z2)
        return loss, bkg_yield, bkg_sum_w2, z1, z2


@tf.function
def train_step(model, data, opt, reweight, lamY, lamU, thrY, thrU):
    with tf.GradientTape() as tape:
        loss, B_sig, B_sig_w2, z1, z2 = model.call(data, reweight)
        penalty_y = low_bkg_penalty(B_sig, threshold=thrY)
        penalty_u = high_bkg_uncertainty_penalty(B_sig_w2, B_sig, rel_threshold=thrU)
        total = loss + lamY * penalty_y + lamU * penalty_u
    grads = tape.gradient(total, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return total, loss, penalty_y, penalty_u, z1, z2


def main():
    parser = argparse.ArgumentParser(
        description="Diphoton bump-hunt optimisation with GATO."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gato-bins", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--lam-yield", type=float, default=0.0)
    parser.add_argument("--lam-unc", type=float, default=0.0)
    parser.add_argument("--thr-yield", type=float, default=10)
    parser.add_argument("--thr-unc", type=float, default=0.1)
    parser.add_argument("--n-bkg", type=int, default=1_000_000)
    parser.add_argument("--n-signal", type=int, default=100_000)
    parser.add_argument("--rewt-interval", type=int, default=50)
    parser.add_argument("--mass-sigma", type=float, default=1.5)
    parser.add_argument("--out", type=str, default="PlotsDiphotonBumpHunt")
    args = parser.parse_args()

    path_plots = os.path.join("examples", "bumphunt_example", args.out)
    os.makedirs(path_plots, exist_ok=True)

    data_full = generate_resonance_toy_data(
        n_signal1=args.n_signal,
        n_signal2=args.n_signal,
        n_bkg=args.n_bkg,
        mass_sigma=args.mass_sigma,
        background_slopes=(0.05, 0.04, 0.035, 0.03, 0.025),
    )
    data_2d = slice_to_2d_features(data_full)
    tensor_data = convert_mass_data_to_tensors(data_2d)

    plot_inclusive_mass(data_2d, path_plots, sig_scales=(50, 250))

    sig_low = 125.0 - args.mass_sigma
    sig_high = 125.0 + args.mass_sigma

    for n_cats in args.gato_bins:
        print(f"\n--- Optimising {n_cats} bins ---")
        model = DiphotonSoftmax(
            n_cats=n_cats, temperature=1.0, mass_sigma=args.mass_sigma
        )
        optimizer = tf.keras.optimizers.RMSprop(0.05)
        lr_scheduler = LearningRateScheduler(
            optimizer,
            lr_initial=0.05,
            lr_final=0.001,
            total_epochs=args.epochs,
            mode="cosine",
        )
        temp_scheduler = TemperatureScheduler(
            model,
            t_initial=1.0,
            t_final=0.1,
            total_epochs=args.epochs,
            mode="cosine",
        )

        reweight = tf.ones(n_cats, dtype=tf.float32)
        loss_history = []
        penalty_y_hist = []
        penalty_u_hist = []
        bias_history = []
        bias_epochs = []
        temp_history = []
        path_bins = os.path.join(path_plots, f"gato_{n_cats}bins")
        os.makedirs(path_bins, exist_ok=True)
        frames_dir = os.path.join(path_bins, "boundary_frames")
        os.makedirs(frames_dir, exist_ok=True)
        boundary_frames = []

        for epoch in range(args.epochs):
            if epoch % max(1, args.rewt_interval) == 0:
                factors = compute_mass_reweight_factors(
                    model,
                    data_2d,
                    signal_labels=["signal1", "signal2"],
                    mass_sig_low=sig_low,
                    mass_sig_high=sig_high,
                )
                reweight = tf.constant(factors, dtype=tf.float32)
                print(f"Updated reweight factors: {factors}")

            _, loss, penY, penU, z1, z2 = train_step(
                model,
                tensor_data,
                optimizer,
                reweight,
                args.lam_yield,
                args.lam_unc,
                args.thr_yield,
                args.thr_unc,
            )
            lr_scheduler.update(epoch)
            temp_scheduler.update(epoch)

            loss_history.append(float(loss.numpy()))
            penalty_y_hist.append(float(penY.numpy()))
            penalty_u_hist.append(float(penU.numpy()))

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                lr_value = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
                lr_value = float(lr_value.numpy()) if hasattr(lr_value, "numpy") else float(lr_value)
                temperature = model.temperature
                temp_history.append(temperature)
                bias_input = {
                    p: {
                        "NN_output": tensor_data[p]["NN_output"],
                        "weight": tensor_data[p]["weight"],
                    }
                    for p in tensor_data
                }
                bias_vec = model.get_bias(bias_input)
                bias_history.append(float(np.mean(np.abs(bias_vec))))
                bias_epochs.append(epoch)
                boundary_fname = os.path.join(frames_dir, f"boundary_{epoch:04d}.png")
                plot_bin_boundaries_2D(
                    model,
                    list(range(n_cats)),
                    boundary_fname,
                    resolution=600,
                    annotation=f"Epoch {epoch}",
                )
                boundary_frames.append(boundary_fname)
                print(
                    f"[{epoch:04d}] loss={loss.numpy():.4f} "
                    f"Z1={z1.numpy():.3f} Z2={z2.numpy():.3f} lr={lr_value:.5f}"
                )

        ckpt_dir = os.path.join(path_plots, "checkpoints", f"{n_cats}_bins")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(ckpt_dir)

        loss_eval = model.call(tensor_data, reweight)
        _, _, _, z1_final, z2_final = loss_eval
        print(
            f"Final significances for {n_cats} bins: "
            f"Z(signal1)={float(z1_final.numpy()):.3f}, "
            f"Z(signal2)={float(z2_final.numpy()):.3f}"
        )

        plot_history(
            np.array(loss_history),
            os.path.join(path_bins, f"loss_{n_cats}.pdf"),
            y_label="Geometric mean significance",
            x_label="Epoch",
        )
        plot_history(
            np.array(penalty_y_hist),
            os.path.join(path_bins, f"penalty_yield_{n_cats}.pdf"),
            y_label="Low background penalty",
            x_label="Epoch",
        )
        plot_history(
            np.array(penalty_u_hist),
            os.path.join(path_bins, f"penalty_unc_{n_cats}.pdf"),
            y_label="High-uncertainty penalty",
            x_label="Epoch",
        )
        plot_bias_history(
            bias_history,
            os.path.join(path_bins, f"bias_history_{n_cats}.pdf"),
            epochs=bias_epochs,
            temp_points=temp_history,
            temp_label="Temperature",
        )

        assignments = model.get_bin_indices(
            {p: {"NN_output": tensor_data[p]["NN_output"]} for p in tensor_data}
        )
        assign_np = {k: v.numpy() for k, v in assignments.items()}
        per_cat_hists = build_category_mass_maps(assign_np, data_2d, n_cats)
        plot_category_mass_spectra(
            per_cat_hists,
            os.path.join(path_bins, "mass_spectra"),
            sig_scales=(2, 10),
        )

        plot_bin_boundaries_2D(
            model,
            list(range(n_cats)),
            os.path.join(path_bins, f"bin_boundaries_{n_cats}_bins.pdf"),
        )

        if boundary_frames:
            gif_path = os.path.join(path_bins, f"boundary_evolution_{n_cats}.gif")
            make_gif(boundary_frames, gif_path, interval=500)
        B_sorted, rel_unc, _ = model.compute_hard_bkg_stats(
            {p: {"NN_output": tensor_data[p]["NN_output"], "weight": tensor_data[p]["weight"]} for p in tensor_data},
            signal_labels=["signal1", "signal2"],
        )
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc,
            output_filename=os.path.join(path_bins, f"yield_unc_{n_cats}.pdf"),
        )
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc,
            output_filename=os.path.join(path_bins, f"yield_unc_{n_cats}_log.pdf"),
            log=True,
        )


if __name__ == "__main__":
    main()
