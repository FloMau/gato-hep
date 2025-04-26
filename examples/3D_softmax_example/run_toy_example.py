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

from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history, plot_learned_gaussians, visualize_bins_2d
from diffcat_optimizer.differentiable_categories import asymptotic_significance, DiffCatModelMultiDimensional, low_bkg_penalty
from generate_toy_data import generate_toy_data_multiclass

# ------------------------------------------------------------------------------
# Helper: Create a fixed histogram from 1D data.
# ------------------------------------------------------------------------------
def create_hist(data, weights=None, bins=50, low=0.0, high=1.0, name="Projection"):
    if isinstance(bins, int):
        h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    else:
        h = hist.Hist.new.Var(bins, name=name).Weight()
    if weights is not None:
        h.fill(data, weight=weights)
    else:
        h.fill(data)
    return h

def assign_bins_and_order(model, data, eps=1e-6):
    """
    Given a trained multidimensional cut model and a data dictionary,
    compute for each event the hard assignment (using argmax over joint log-probabilities),
    accumulate yields to compute a significance measure per bin,
    and then re-map the original bin indices to new indices so that the
    most significant bin gets the highest new index.
    
    Returns:
      bin_assignments: dict mapping process name -> array of new bin indices for each event.
      order: sorted array of original bin indices (ascending significance).
      sb_ratios: array of significance (S/sqrt(B)) for each original bin.
      inv_mapping: dictionary mapping new bin index -> original bin index.
    """
    n_cats = model.n_cats

    # Retrieve learned parameters.
    log_mix = tf.nn.log_softmax(model.mixture_logits)  # shape: (n_cats,)
    scale_tril = model.get_scale_tril()                 # shape: (n_cats, dim, dim)
    means = model.means                                 # shape: (n_cats, dim)

    # Accumulate yields per bin.
    S_yields = np.zeros(n_cats, dtype=np.float32)
    B_yields = np.zeros(n_cats, dtype=np.float32)

    bin_assignments = {}

    for proc, df in data.items():
        if df.empty:
            bin_assignments[proc] = np.array([])
            continue
        x = tf.constant(np.stack(df["NN_output"].values), dtype=tf.float32)
        weights = df["weight"].values  # shape: (n_events,)
    
        log_probs = []
        for i in range(n_cats):
            dist = tfd.MultivariateNormalTriL(
                loc=tf.nn.softmax(means[i]),  # as used in training
                scale_tril=scale_tril[i]
            )
            lp = dist.log_prob(x)  # shape: (n_events,)
            log_probs.append(lp)
        log_probs = tf.stack(log_probs, axis=1)  # shape: (n_events, n_cats)
    
        # Add mixture weights.
        log_joint = log_probs + log_mix  # shape: (n_events, n_cats)
    
        # Hard assignment: argmax.
        assignments = tf.argmax(log_joint, axis=1).numpy()  # shape: (n_events,)
        bin_assignments[proc] = assignments
    
        # Accumulate yields.
        for i in range(n_cats):
            mask = (assignments == i)
            yield_sum = weights[mask].sum()
            if proc == "signal":
                S_yields[i] += yield_sum
            else:
                B_yields[i] += yield_sum

    # Compute a significance measure per bin. (Here we use S/sqrt(B))
    significances = S_yields / (np.sqrt(B_yields) + eps)
    
    # order: sort original bin indices in ascending order (lowest significance first)
    order = np.argsort(significances)  # e.g., [orig_bin_low, ..., orig_bin_high]
    
    # Create new mapping: assign new indices 0...n_cats-1 in order of ascending significance.
    new_order_mapping = {orig: new for new, orig in enumerate(order)}
    # And the inverse mapping: new index -> original index.
    inv_mapping = {v: k for k, v in new_order_mapping.items()}

    # Remap bin assignments.
    for proc in bin_assignments:
        orig_assign = bin_assignments[proc]
        bin_assignments[proc] = np.vectorize(lambda i: new_order_mapping[i])(orig_assign)
    
    return bin_assignments, order, significances, inv_mapping


def fill_histogram_from_assignments(assignments, weights, nbins, name="BinAssignments"):
    """
    Given an array of integer assignments (one per event) and corresponding weights,
    fill a histogram with bins [0, nbins] using the hist library.
    """
    # Create a histogram with nbins bins, ranging from 0 to nbins.
    h = hist.Hist.new.Reg(nbins, 0, nbins, name=name).Weight()
    h.fill(assignments, weight=weights)
    return h


# ------------------------------------------------------------------------------
# Compute overall significance from fixed histograms.
# ------------------------------------------------------------------------------
def compute_significance(h_signal, bkg_hists):
    # Sum background counts bin-by-bin.
    B_vals = sum(h.values() for h in bkg_hists)
    S_vals = h_signal.values()
    S_tensor = tf.constant(S_vals, dtype=tf.float32)
    B_tensor = tf.constant(B_vals, dtype=tf.float32)
    Z_bins = asymptotic_significance(S_tensor, B_tensor)
    Z_overall = np.sqrt(np.sum(Z_bins.numpy()**2))
    return Z_overall

# ------------------------------------------------------------------------------
# Differentiable model: subclass that optimizes a Gaussian mixture for multidimensional NN output.
# ------------------------------------------------------------------------------
class DiffCatModelExample(DiffCatModelMultiDimensional):
    """
    This class simply inherits the functionality from DiffCatModelMultiDimensional.
    You can override call() here if needed. In this example, we use the full 3D NN output.
    """
    def call(self, data_dict):
        """
        Compute the loss (negative overall significance) and background yields.
        """
        n_cats = self.n_cats
        # Compute log mixture weights.
        log_mix = tf.nn.log_softmax(self.mixture_logits)  # shape: (n_cats,)
        scale_tril = self.get_scale_tril()                 # shape: (n_cats, dim, dim)
        means = self.means                                 # shape: (n_cats, dim)

        # Accumulate yields per category.
        signal_yields = [0.0] * n_cats
        background_yields = [0.0] * n_cats

        # Loop over processes.
        for proc, df in data_dict.items():
            # print("sum weights for proc", proc, np.sum(df.weight))
            if df.empty:
                continue
            # x: shape (n_events, dim)
            x = tf.constant(np.stack(df["NN_output"].values), dtype=tf.float32)
            w = tf.constant(df["weight"].values, dtype=tf.float32)

            # For each category, compute the log pdf.
            log_probs = []
            for i in range(n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=tf.nn.softmax(means[i]),
                    scale_tril=scale_tril[i]
                )
                lp = dist.log_prob(x)  # shape: (n_events,)
                log_probs.append(lp)
            log_probs = tf.stack(log_probs, axis=1)  # shape: (n_events, n_cats)

            # Joint log probability: log mix weight + log pdf.
            log_joint = log_probs + log_mix  # shape: (n_events, n_cats)

            # Temperatured softmax to get soft memberships.
            memberships = tf.nn.softmax(log_joint / self.temperature, axis=1)

            # Weighted yield per category.
            yields = tf.reduce_sum(memberships * tf.expand_dims(w, axis=1), axis=0)

            if proc == "signal":
                for i in range(n_cats):
                    signal_yields[i] += yields[i]
            else:
                for i in range(n_cats):
                    background_yields[i] += yields[i]

        S = tf.convert_to_tensor(signal_yields, dtype=tf.float32)
        B = tf.convert_to_tensor(background_yields, dtype=tf.float32)
        Z_bins = asymptotic_significance(S, B)
        Z_overall = tf.sqrt(tf.reduce_sum(tf.square(Z_bins)))

        return -Z_overall, B

    pass

# ------------------------------------------------------------------------------
# Main: Baseline fixed binning using the first component of the NN output as discriminant.
# (All events are used.)
# ------------------------------------------------------------------------------
def main():
    # Set covariance and generate multiclass toy data.
    cov = [[2.0, 0.3, 0.3],
           [0.2, 2.0, 0.2],
           [0.3, 0.3, 2.0]]
    data = generate_toy_data_multiclass(
        n_signal=100000,
        n_bkg1=200000,
        n_bkg2=100000,
        mean_signal=[1.0, -1.0, -1.5],
        mean_bkg1=[0.0, 1.5, 0.0],
        mean_bkg2=[0.0, 0.0, 1.5],
        xs_signal=0.5, xs_bkg1=50, xs_bkg2=15,
        lumi=100,
        cov=cov,
        seed=42
    )

    # Here we use the first component of the NN output as the discriminant.
    # For each process, extract the first component on the fly.
    discriminants = {}
    for proc, df in data.items():
        arr = np.stack(df["NN_output"].values)  # shape: (n_events, 3)
        discriminants[proc] = arr[:, 0]
        print(f"{proc}: {len(discriminants[proc])} events")

    # Create a directory for baseline plots.
    path_plots = "examples/3D_softmax_example/Plots/FixedBinning/"
    os.makedirs(path_plots, exist_ok=True)

    # Define a list of bin numbers to test.
    equidistant_binning_options = [3, 5, 10, 25]
    gato_binning_options = [3, 5, 10, 25]
    equidistant_significances = {}
    optimized_significances = {}

    # Loop over each binning option.
    for nbins in equidistant_binning_options:
        # Build histograms for each process using the discriminant (first component).
        h_signal = create_hist(discriminants["signal"],
                               weights=data["signal"]["weight"],
                               bins=nbins, low=0.0, high=1.0, name="Signal")
        bkg_hists = []
        bkg_labels = []
        for proc in discriminants:
            if proc == "signal":
                continue
            h = create_hist(discriminants[proc],
                            weights=data[proc]["weight"],
                            bins=nbins, low=0.0, high=1.0, name=proc)
            bkg_hists.append(h)
            bkg_labels.append(proc)
    
            # Compute overall significance.
        Z = compute_significance(h_signal, bkg_hists)
        equidistant_significances[nbins] = Z
        print(f"Fixed binning ({nbins} bins): Overall significance = {Z:.3f}")

        # Plot the histograms using the plotting utility.
        signal_labels = [r"Signal $\times 10$"]
        output_filename = os.path.join(path_plots, f"NN_output_distribution_fixed_{nbins}bins.pdf")
        plot_stacked_histograms(
            stacked_hists=bkg_hists,
            process_labels=bkg_labels,
            signal_hists=[10*h_signal],
            signal_labels=signal_labels,
            output_filename=output_filename,
            axis_labels=("NN output (signal node)", "Events"),
            normalize=False,
            log=False
        )
        # Also save a log-scale version.
        plot_stacked_histograms(
            stacked_hists=bkg_hists,
            process_labels=bkg_labels,
            signal_hists=[10*h_signal],
            signal_labels=signal_labels,
            output_filename=output_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("NN output (signal node)", "Events"),
            normalize=False,
            log=True
        )
        print(f"Saved fixed binning plot for {nbins} bins as {output_filename}")

    # Optimization: learn the Gaussian mixture clustering.
    # Here we use the full 3D NN output.
    path_plots = "examples/3D_softmax_example/Plots/gatoBinningsLam0p01/"
    for nbins in gato_binning_options:
        model = DiffCatModelExample(n_cats=nbins, dim=3, temperature=0.1)
        lam = 0.01
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        loss_history = []
        reg_history = []
        param_history = []
        epochs = 250
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss, B = model.call(data)
                reg = tf.Variable(np.array([0]))
                total_loss = loss
                if lam != 0:
                    reg = low_bkg_penalty(B, threshold=10, steepness=10)
                    total_loss += lam * reg
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_history.append(loss.numpy())
            reg_history.append(reg.numpy())
            param_history.append(model.get_effective_parameters())
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch}] total_loss={total_loss.numpy():.3f}, base_loss={loss.numpy():.3f}")
                # print("Effective parameters:", model.get_effective_parameters())

        # Retrieve effective parameters and, e.g., use the learned mixture to define bin edges.
        eff_params = model.get_effective_parameters()
        print("Final learned parameters:", eff_params)

        bin_assignments, order, significances, inv_mapping = assign_bins_and_order(model, data)

        # Assume bin_assignments is a dict mapping process -> array of bin indices,
        # and that for each process the weights are stored in data[proc]["weight"].
        h_signal_opt = fill_histogram_from_assignments(bin_assignments["signal"],data["signal"]["weight"],nbins)
        h_bkg1_opt = fill_histogram_from_assignments(bin_assignments["bkg1"],data["bkg1"]["weight"], nbins)
        h_bkg2_opt = fill_histogram_from_assignments(bin_assignments["bkg2"],data["bkg2"]["weight"], nbins)

        # visualize_bins_2d(
        #     data_dict=
        # )
        print("Signal hist:", h_signal_opt)
        print("background hist 1", h_bkg1_opt)

        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt]
        Z_opt = compute_significance(h_signal_opt, opt_bkg_hists)
        optimized_significances[nbins] = Z_opt
        print(f"Optimized binning significance: {Z_opt:.3f}")

        opt_plot_filename = os.path.join(path_plots, f"NN_output_distribution_optimized_{nbins}bins.pdf")
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg1", "Bkg2"],
            signal_hists=[h_signal_opt*10],
            signal_labels=[r"Signal $\times 10$"],
            output_filename=opt_plot_filename,
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=False
        )
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg1", "Bkg2"],
            signal_hists=[h_signal_opt*10],
            signal_labels=[r"Signal $\times 10$"],
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=True
        )
        # Optionally, plot training histories.
        loss_plot_name = os.path.join(path_plots, f"history_loss_{nbins}bins.pdf")
        plot_history(loss_history, loss_plot_name, y_label="Negative significance", x_label="Epoch", boundaries=False, title="Loss History")

        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=0,
            dim_y=1,
            output_filename=os.path.join(path_plots, f"GaussianBlobs_{nbins}Bins_dims01.pdf"),
            inv_mapping=inv_mapping
        )

        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=0,
            dim_y=2,
            output_filename=os.path.join(path_plots, f"GaussianBlobs_{nbins}Bins_dims02.pdf"),
            inv_mapping=inv_mapping
        )
        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=1,
            dim_y=2,
            output_filename=os.path.join(path_plots, f"GaussianBlobs_{nbins}Bins_dims12.pdf"),
            inv_mapping=inv_mapping
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
