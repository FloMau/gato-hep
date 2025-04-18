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

from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history, plot_learned_gaussians
from diffcat_optimizer.differentiable_categories import asymptotic_significance, DiffCatModelMultiDimensional, low_bkg_penalty
# We'll use a new generator for 5D data.
# from generate_toy_data import generate_toy_data_multiclass_5D

# ------------------------------------------------------------------------------
# 5D Toy Data Generator
# ------------------------------------------------------------------------------
def generate_toy_data_multiclass_5D(n_signal1=1000, n_signal2=1000,
                                    n_bkg1=1000, n_bkg2=1000, n_bkg3=1000,
                                    # Means chosen such that after softmax, 
                                    # signal1 is biased to component 0, signal2 to component 1,
                                    # and backgrounds to components 2, 3, and 4.
                                    mean_signal1=[1.5, -1.0, -0.5, -0.5, -0.5],
                                    mean_signal2=[-1.0, 1.5, -0.5, -0.5, -0.5],
                                    mean_bkg1=[-0.5, -0.5, 1.0, -1.0, -1.0],
                                    mean_bkg2=[-0.5, -0.5, -1.0, 1.0, -1.0],
                                    mean_bkg3=[-0.5, -0.5, -1.0, -1.0, 1.0],
                                    # Common covariance matrix (5x5), modest correlations.
                                    cov=np.eye(5) * 1.0 + 0.3*(np.ones((5,5))-np.eye(5)),
                                    xs_signal1=0.5, xs_signal2=0.1,  # second signal 5x smaller
                                    xs_bkg1=100, xs_bkg2=10, xs_bkg3=1,
                                    lumi=100, seed=None):
    """
    Generate 5D NN output toy data for two signals and three backgrounds.
    Each event is a 5D vector that is passed through softmax.
    """
    if seed is not None:
        np.random.seed(seed)
    # Generate raw outputs.
    raw_signal1 = np.random.multivariate_normal(mean_signal1, cov, n_signal1)
    raw_signal2 = np.random.multivariate_normal(mean_signal2, cov, n_signal2)
    raw_bkg1 = np.random.multivariate_normal(mean_bkg1, cov, n_bkg1)
    raw_bkg2 = np.random.multivariate_normal(mean_bkg2, cov, n_bkg2)
    raw_bkg3 = np.random.multivariate_normal(mean_bkg3, cov, n_bkg3)

    # Apply softmax row-wise.
    signal1_softmax = tf.nn.softmax(tf.constant(raw_signal1, dtype=tf.float32), axis=1).numpy()
    signal2_softmax = tf.nn.softmax(tf.constant(raw_signal2, dtype=tf.float32), axis=1).numpy()
    bkg1_softmax = tf.nn.softmax(tf.constant(raw_bkg1, dtype=tf.float32), axis=1).numpy()
    bkg2_softmax = tf.nn.softmax(tf.constant(raw_bkg2, dtype=tf.float32), axis=1).numpy()
    bkg3_softmax = tf.nn.softmax(tf.constant(raw_bkg3, dtype=tf.float32), axis=1).numpy()

    # Compute per-event weights.
    weight_signal1 = xs_signal1 * lumi / n_signal1
    weight_signal2 = xs_signal2 * lumi / n_signal2
    weight_bkg1 = xs_bkg1 * lumi / n_bkg1
    weight_bkg2 = xs_bkg2 * lumi / n_bkg2
    weight_bkg3 = xs_bkg3 * lumi / n_bkg3
    
    # Build DataFrames.
    df_signal1 = pd.DataFrame({"NN_output": list(signal1_softmax), "weight": weight_signal1})
    df_signal2 = pd.DataFrame({"NN_output": list(signal2_softmax), "weight": weight_signal2})
    df_bkg1 = pd.DataFrame({"NN_output": list(bkg1_softmax), "weight": weight_bkg1})
    df_bkg2 = pd.DataFrame({"NN_output": list(bkg2_softmax), "weight": weight_bkg2})
    df_bkg3 = pd.DataFrame({"NN_output": list(bkg3_softmax), "weight": weight_bkg3})
    
    data = {
        "signal1": df_signal1,
        "signal2": df_signal2,
        "bkg1": df_bkg1,
        "bkg2": df_bkg2,
        "bkg3": df_bkg3
    }
    return data

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

def convert_data_to_tensors(data):
    """
    Convert the dictionary of DataFrames (with 'NN_output' and 'weight')
    into a dictionary of dictionaries containing TF tensors.
    This conversion is done once and passed to the training function.
    """
    tensor_data = {}
    for proc, df in data.items():
        tensor_data[proc] = {
            "NN_output": tf.constant(np.stack(df["NN_output"].values), dtype=tf.float32),
            "weight": tf.constant(df["weight"].values, dtype=tf.float32)
        }
    return tensor_data

# ------------------------------------------------------------------------------
# Overall significance from histograms (same as before).
# ------------------------------------------------------------------------------
def compute_significance(h_signal, h_bkg_list):
    B_vals = sum(h.values() for h in h_bkg_list)
    S_vals = h_signal.values()
    S_tensor = tf.constant(S_vals, dtype=tf.float32)
    B_tensor = tf.constant(B_vals, dtype=tf.float32)
    Z_bins = asymptotic_significance(S_tensor, B_tensor)
    return np.sqrt(np.sum(Z_bins.numpy()**2))

# ------------------------------------------------------------------------------
# Differentiable model: subclass that optimizes a Gaussian mixture for 5D NN output.
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Differentiable model: DiffCatModelExample5D
# ------------------------------------------------------------------------------
class DiffCatModelExample5D(DiffCatModelMultiDimensional):
    """
    Inherits from DiffCatModelMultiDimensional with full 5D NN output.
    
    For training, we define two channels:
      - Channel 1 (signal1): signal events are those from "signal1",
        background events include all events from other processes and from "signal2".
      - Channel 2 (signal2): signal events are those from "signal2",
        background events include all events from other processes and from "signal1".
    
    Loss = -sqrt(Z1 * Z2), where
      Z1 = S1/sqrt(B1) and Z2 = S2/sqrt(B2).
    """
    def call(self, data_dict):
        n_cats = self.n_cats
        log_mix = tf.nn.log_softmax(self.mixture_logits)
        scale_tril = self.get_scale_tril()
        means = self.means

        sig1_yields = tf.zeros([n_cats], dtype=tf.float32)
        sig2_yields = tf.zeros([n_cats], dtype=tf.float32)
        bkg_yields  = tf.zeros([n_cats], dtype=tf.float32)

        # Loop over processes (data_dict now contains tensors).
        for proc, tensors in data_dict.items():
            x = tensors["NN_output"]  # shape: (n_events, 5)
            w = tensors["weight"]     # shape: (n_events,)
            log_probs = []
            for i in range(n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=tf.nn.softmax(means[i]),
                    scale_tril=scale_tril[i]
                )
                lp = dist.log_prob(x)
                log_probs.append(lp)
            log_probs = tf.stack(log_probs, axis=1)
            log_joint = log_probs + log_mix
            memberships = tf.nn.softmax(log_joint / self.temperature, axis=1)
            proc_yields = tf.reduce_sum(memberships * tf.expand_dims(w, axis=1), axis=0)
            if proc == "signal1":
                sig1_yields += proc_yields
            elif proc == "signal2":
                sig2_yields += proc_yields
            else:
                bkg_yields += proc_yields

        S1 = sig1_yields
        B1 = bkg_yields + sig2_yields
        Z1_bins = asymptotic_significance(S1, B1)
        Z1_overall = tf.sqrt(tf.reduce_sum(tf.square(Z1_bins)))

        S2 = sig2_yields
        B2 = bkg_yields + sig1_yields
        Z2_bins = asymptotic_significance(S2, B2)
        Z2_overall = tf.sqrt(tf.reduce_sum(tf.square(Z2_bins)))

        loss = - tf.sqrt(Z1_overall * Z2_overall)
        return loss, tf.reduce_sum(bkg_yields)

# ------------------------------------------------------------------------------
# Main: Run baseline fixed binning and differentiable optimization for 5D.
# ------------------------------------------------------------------------------
def main():
   # Generate 5D toy data with two signals and three backgrounds.
    data = generate_toy_data_multiclass_5D(
        n_signal1=100000,
        n_signal2=100000,
        n_bkg1=200000,
        n_bkg2=100000,
        n_bkg3=100000,
        mean_signal1=[1.5, 0, -0.5, -0.5, -0.5],
        mean_signal2=[0, 1.5, -0.5, -0.5, -0.5],
        mean_bkg1=[-0.5, -0.5, 1.0, -1.0, -1.0],
        mean_bkg2=[-0.5, -0.5, -1.0, 1.0, -1.0],
        mean_bkg3=[0.5, 0.5, 0, 0, 1.0],
        xs_signal1=0.5,
        xs_signal2=0.1,  # signal2 is 5x smaller
        xs_bkg1=100, xs_bkg2=50, xs_bkg3=10,
        lumi=100,
        cov=np.eye(5)*2 + 0.3*(np.ones((5,5))-np.eye(5)*2),
        seed=42
    )

    # For each event, compute the argmax over the full 5D NN output.
    baseline_assignments = {}
    for proc, df in data.items():
        arr = np.stack(df["NN_output"].values)  # shape: (n_events, 5)
        assignments = np.argmax(arr, axis=1)     # shape: (n_events,)
        baseline_assignments[proc] = assignments

    # Define the binning options.
    baseline_binning_options = [2, 5, 10]
    baseline_signif_signal1 = {}
    baseline_signif_signal2 = {}

    # Create an output directory for baseline plots.
    path_plots = "examples/5D_2signals_softmax_example/Plots/Baseline/"
    os.makedirs(path_plots, exist_ok=True)

    # Loop over different bin numbers.
    for nbins in baseline_binning_options:
        # --- For signal1 baseline ---
        # Use events with argmax==0.
        # Background for signal1: events from all processes (including signal2 and backgrounds) with argmax==0.
        bkg_hists_sig1 = []
        bkg_labels_sig1 = []
        for proc, df in data.items():
            mask = baseline_assignments[proc] == 0
            if np.sum(mask) == 0:
                continue
            vals = np.stack(df["NN_output"].values)[:, 0][mask]
            weights = df["weight"].values[mask]
            if proc == "signal1":
                h_signal1 = create_hist(vals, weights=weights, bins=nbins, low=0.2, high=1.0, name=proc)
            else:
                h = create_hist(vals, weights=weights, bins=nbins, low=0.2, high=1.0, name=proc)
                bkg_hists_sig1.append(h)
                bkg_labels_sig1.append(proc)
        # Compute significance for signal1 channel.
        Z1_baseline = compute_significance(h_signal1, bkg_hists_sig1)
        baseline_signif_signal1[nbins] = Z1_baseline

        # --- For signal2 baseline ---
        # Use events with argmax==1.
        bkg_hists_sig2 = []
        bkg_labels_sig2 = []
        for proc, df in data.items():
            mask = baseline_assignments[proc] == 1
            if np.sum(mask) == 0:
                continue
            vals = np.stack(df["NN_output"].values)[:, 1][mask]
            weights = df["weight"].values[mask]
            if proc == "signal2":
                h_signal2 = create_hist(vals, weights=weights, bins=nbins, low=0.2, high=1.0, name=proc)
            else:
                h = create_hist(vals, weights=weights, bins=nbins, low=0.2, high=1.0, name=proc)
                bkg_hists_sig2.append(h)
                bkg_labels_sig2.append(proc)
        Z2 = compute_significance(h_signal2, bkg_hists_sig2)
        baseline_signif_signal2[nbins] = Z2

        baseline_configs = [
            ("signal1", h_signal1, bkg_hists_sig1, bkg_labels_sig1, "NN output (signal1 node)", r"Signal1 $\times 10$"),
            ("signal2", h_signal2, bkg_hists_sig2, bkg_labels_sig2, "NN output (signal2 node)", r"Signal2 $\times 10$")
        ]

        # Loop over each baseline configuration (signal1 and signal2).
        for config in baseline_configs:
            (sig_name, h_signal, bkg_hists, bkg_labels, x_label, sig_label) = config

            # Loop over scale types: linear and log.
            for scale in [False, True]:
                scale_tag = "lin" if not scale else "log"
                output_filename = os.path.join(path_plots, f"baseline_histograms_{sig_name}_{nbins}bins_{scale_tag}.pdf")
                plot_stacked_histograms(
                    stacked_hists=bkg_hists,
                    process_labels=bkg_labels,
                    signal_hists=[10 * h_signal],
                    signal_labels=[sig_label],
                    output_filename=output_filename,
                    axis_labels=(x_label, "Events"),
                    normalize=False,
                    log=scale
                )
                print(f"Saved {sig_name} baseline histogram ({scale_tag}) for {nbins} bins as {output_filename}")

    # ----- Optimization: learn the Gaussian mixture clustering for 5D -----
    path_plots_opt = "examples/5D_2signals_softmax_example/Plots/gato/"
    os.makedirs(path_plots_opt, exist_ok=True)
    # Use the full 5D output.
    Z1_gato = {}
    Z2_gato = {}
    gato_binning_options = [3, 25]
    tensor_data = convert_data_to_tensors(data)
    for n_cats in gato_binning_options:

        # ------------------------------------------------------------------------------
        # Training step wrapped in @tf.function.
        # ------------------------------------------------------------------------------
        @tf.function
        def train_step(model, data, optimizer, lam=0.0):
            with tf.GradientTape() as tape:
                loss, B = model.call(data)
                total_loss = loss
                reg = tf.constant(0.0, dtype=tf.float32)
                if lam != 0:
                    reg = low_bkg_penalty(B, threshold=10, steepness=10)
                    total_loss += lam * reg
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, reg

        model = DiffCatModelExample5D(n_cats=n_cats, dim=5, temperature=1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        lam = 0.0

        loss_history = []
        reg_history = []
        param_history = []
        epochs = 250

        for epoch in range(epochs):
            total_loss, loss, reg = train_step(model, tensor_data, optimizer, lam=lam)
            loss_history.append(loss.numpy())
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch}] total_loss = {loss.numpy():.3f}")

            reg_history.append(reg.numpy())
            param_history.append(model.get_effective_parameters())
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch}] total_loss={total_loss.numpy():.3f}, base_loss={loss.numpy():.3f}")
                print(model.get_effective_parameters())

            if epoch > 0 and epoch % 20 == 0:
                model.temperature /= 2

        # Retrieve effective parameters.
        eff_params = model.get_effective_parameters()
        print("Final learned parameters:", eff_params)

        # Now assign bins using the learned mixture.
        from diffcat_optimizer.plotting_utils import assign_bins_and_order, fill_histogram_from_assignments
        bin_assignments, order, sig_values, inv_mapping = assign_bins_and_order(model, data)

        h_signal1_opt = fill_histogram_from_assignments(bin_assignments["signal1"], data["signal1"]["weight"], n_cats)
        h_signal2_opt = fill_histogram_from_assignments(bin_assignments["signal2"], data["signal2"]["weight"], n_cats)
        h_bkg1_opt = fill_histogram_from_assignments(bin_assignments["bkg1"], data["bkg1"]["weight"], n_cats)
        h_bkg2_opt = fill_histogram_from_assignments(bin_assignments["bkg2"], data["bkg2"]["weight"], n_cats)
        h_bkg3_opt = fill_histogram_from_assignments(bin_assignments["bkg3"], data["bkg3"]["weight"], n_cats)

        # Compute optimized significance (here using your compute_significance routine).
        opt_bkg_hists = [h_bkg1_opt, h_bkg2_opt, h_bkg3_opt]
        Z1_opt = compute_significance(h_signal1_opt, opt_bkg_hists+[h_signal2_opt])
        Z2_opt = compute_significance(h_signal2_opt, opt_bkg_hists+[h_signal1_opt])
        print(f"Optimized binning significances Z1, Z2: {Z1_opt:.3f}, {Z2_opt:.3f}")

        Z1_gato[n_cats] = Z1_opt
        Z2_gato[n_cats] = Z2_opt

        opt_plot_filename = os.path.join(path_plots_opt, f"NN_output_distribution_optimized_{n_cats}bins.pdf")
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg1", "Bkg2", "Bkg3"],
            signal_hists=[h_signal1_opt*10, h_signal2_opt*50],
            signal_labels=[r"Signal 1 $\times 10$", r"Signal 2 $\times 50$"],
            output_filename=opt_plot_filename,
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=False
        )

        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg1", "Bkg2", "Bkg3"],
            signal_hists=[h_signal1_opt*10, h_signal2_opt*50],
            signal_labels=[r"Signal 1 $\times 10$", r"Signal 2 $\times 50$"],
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=True
        )

        plot_history(loss_history, os.path.join(path_plots_opt, f"history_loss_{n_cats}bins.pdf"), 
                    y_label="Negative geometric mean significance", x_label="Epoch", boundaries=False, title="Loss History")

        # Plot learned Gaussian ellipses in 2D projections.
        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=0,
            dim_y=1,
            output_filename=os.path.join(path_plots_opt, f"GaussianBlobs_{n_cats}Bins_dims01.pdf"),
            inv_mapping=inv_mapping
        )
        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=0,
            dim_y=2,
            output_filename=os.path.join(path_plots_opt, f"GaussianBlobs_{n_cats}Bins_dims02.pdf"),
            inv_mapping=inv_mapping
        )
        plot_learned_gaussians(
            data=data,
            model=model,
            dim_x=1,
            dim_y=2,
            output_filename=os.path.join(path_plots_opt, f"GaussianBlobs_{n_cats}Bins_dims12.pdf"),
            inv_mapping=inv_mapping
        )


    path = "examples/5D_2signals_softmax_example/Plots/"
    # Plot baseline significance comparison.
    fig, ax = plt.subplots(figsize=(8, 6))
    sig1_values_baseline = [baseline_signif_signal1[nb] for nb in baseline_binning_options]
    sig2_values_baseline = [baseline_signif_signal2[nb] for nb in baseline_binning_options]

    sig1_values_gato = [Z1_gato[nb] for nb in gato_binning_options]
    sig2_values_gato = [Z2_gato[nb] for nb in gato_binning_options]

    ax.plot(2*np.array(baseline_binning_options)+1, sig1_values_baseline, marker='o', linestyle='-', label="Equidistant binning signal 1")
    ax.plot(2*np.array(baseline_binning_options)+1, sig2_values_baseline, marker='s', linestyle='--', label="Equidistant binning signal 2")
    ax.plot(gato_binning_options, sig1_values_gato, marker='o', linestyle='-', label="GATO binning signal 1")
    ax.plot(gato_binning_options, sig2_values_gato, marker='s', linestyle='--', label="GATO binning signal 2")
    ax.set_xlabel("Number of bins", fontsize=16)
    ax.set_ylabel("Significance", fontsize=16)
    ax.legend(fontsize=14)
    plt.tight_layout()
    comp_filename = os.path.join(path, "significance_comparison.pdf")
    comp_filename = os.path.join(path_plots_opt, "significance_comparison.pdf")
    plt.savefig(comp_filename)
    plt.close(fig)
    print(f"significance comparison plot saved as {comp_filename}")

if __name__ == "__main__":
    main()
