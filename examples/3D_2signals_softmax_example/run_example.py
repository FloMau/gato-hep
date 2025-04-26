import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import hist
from scipy.stats import multivariate_normal
# Append the repo root to sys.path so that we can import our core modules.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from diffcat_optimizer.differentiable_categories import asymptotic_significance, DiffCatModelMultiDimensional, low_bkg_penalty
from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history, plot_learned_gaussians, assign_bins_and_order, fill_histogram_from_assignments, visualize_bins_2d

# ----------------------------------------------------------------------------
# 3D Toy Data Generator for 3-class classifier
# Background consists of 5 individual Gaussian processes
# ----------------------------------------------------------------------------
def generate_toy_data_3class_3D(
    n_signal1=100000, n_signal2=100000,
    n_bkg1=100000, n_bkg2=80000, n_bkg3=50000, n_bkg4=20000, n_bkg5=10000,
    xs_signal1=0.5, xs_signal2=0.1,
    xs_bkg1=100, xs_bkg2=80, xs_bkg3=50, xs_bkg4=20, xs_bkg5=10,
    lumi=100.0, noise_scale=0.2, seed=None
):
    """
    Generate 3D Gaussian data for 2 signals and 5 backgrounds.
    Compute 3-class softmax output on noisy 3D logits.
    Returns dict of DataFrames with 'NN_output' (3-vector) and 'weight'.
    """
    if seed is not None:
        np.random.seed(seed)

    processes = ["signal1","signal2","bkg1","bkg2","bkg3","bkg4","bkg5"]
    means = {
        "signal1": np.array([ 1.5, -1.0, -1.0]),
        "signal2": np.array([-1.0,  1.5, -1.0]),
        "bkg1":    np.array([-0.5, -0.5,  1.0]),
        "bkg2":    np.array([0.5, -0.5,  0.8]),
        "bkg3":    np.array([0.5, 0.5,  -0.6]),
        "bkg4":    np.array([-0.5, 1.0,  -0.4]),
        "bkg5":    np.array([-0.5, 0.5,  -0.2])
    }
    cov = np.eye(3)*1.0 + 0.2*(np.ones((3,3))-np.eye(3))
    counts = {
        "signal1": n_signal1, "signal2": n_signal2,
        "bkg1": n_bkg1, "bkg2": n_bkg2, "bkg3": n_bkg3,
        "bkg4": n_bkg4, "bkg5": n_bkg5
    }
    xs = {
        "signal1": xs_signal1, "signal2": xs_signal2,
        "bkg1": xs_bkg1, "bkg2": xs_bkg2, "bkg3": xs_bkg3,
        "bkg4": xs_bkg4, "bkg5": xs_bkg5
    }
    raw = {p: np.random.multivariate_normal(means[p], cov, size=counts[p]) for p in processes}
    for p in processes:
        raw[p] *= np.random.normal(loc=1.0, scale=noise_scale, size=raw[p].shape)
    nn_out = {p: tf.nn.softmax(tf.constant(raw[p],dtype=tf.float32),axis=1).numpy() for p in processes}
    data = {}
    for p in processes:
        weight = xs[p] * lumi / counts[p]
        data[p] = pd.DataFrame({"NN_output": list(nn_out[p]), "weight": weight})
    return data

def generate_toy_data_3class_3D_Nitish(
    n_signal1=100000, n_signal2=100000,
    n_bkg1=100000, n_bkg2=80000, n_bkg3=50000, n_bkg4=20000, n_bkg5=10000,
    xs_signal1=0.5, xs_signal2=0.1,
    xs_bkg1=100, xs_bkg2=80, xs_bkg3=50, xs_bkg4=20, xs_bkg5=10,
    lumi=100.0, noise_scale=0.2, seed=None
):
    """
    Generate 3D Gaussian data for 2 signal and 5 background classes.
    For each point, compute likelihood-ratio-based 3-class scores:
        [score_signal1, score_signal2, score_background]
    Returns dict of DataFrames with columns: 'NN_output' (3-vector) and 'weight'.
    """
    if seed is not None:
        np.random.seed(seed)

    processes = ["signal1", "signal2", "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"]

    means = {
        "signal1": np.array([1.5, -1.0, -1.0]),
        "signal2": np.array([-1.0, 1.5, -1.0]),
        "bkg1":    np.array([-0.5, -0.5, 1.0]),
        "bkg2":    np.array([0.5, -0.5, 0.8]),
        "bkg3":    np.array([0.5, 0.5, -0.6]),
        "bkg4":    np.array([-0.5, 1.0, -0.4]),
        "bkg5":    np.array([-0.5, 0.5, -0.2])
    }

    # Slightly correlated 3D Gaussian
    cov = np.eye(3)*1.0 + 0.2*(np.ones((3,3)) - np.eye(3))

    counts = {
        "signal1": n_signal1, "signal2": n_signal2,
        "bkg1": n_bkg1, "bkg2": n_bkg2, "bkg3": n_bkg3,
        "bkg4": n_bkg4, "bkg5": n_bkg5
    }

    xs = {
        "signal1": xs_signal1, "signal2": xs_signal2,
        "bkg1": xs_bkg1, "bkg2": xs_bkg2, "bkg3": xs_bkg3,
        "bkg4": xs_bkg4, "bkg5": xs_bkg5
    }

    # 1. Sample raw 3D data
    raw = {
        p: np.random.multivariate_normal(mean=means[p], cov=cov, size=counts[p])
        for p in processes
    }

    # 2. Add multiplicative noise
    for p in processes:
        noise = np.random.normal(loc=1.0, scale=noise_scale, size=raw[p].shape)
        raw[p] *= noise

    # 3. Build PDFs
    pdfs = {
        p: multivariate_normal(mean=means[p], cov=cov)
        for p in processes
    }

    # 4. Combined background PDF with proper cross-section weighting
    bkg_processes = [p for p in processes if p.startswith("bkg")]
    total_bkg_xs = sum(xs[p] for p in bkg_processes)

    def combined_bkg_pdf(X):
        return sum(
            (xs[p] / total_bkg_xs) * pdfs[p].pdf(X)
            for p in bkg_processes
        )

    # 5. Compute likelihood-ratio-based scores
    data = {}
    for proc in processes:
        X = raw[proc]
        weight = xs[proc] * lumi / counts[proc]

        p1 = pdfs["signal1"].pdf(X)
        p2 = pdfs["signal2"].pdf(X)
        pb = combined_bkg_pdf(X)

        total = p1 + p2 + pb + 1e-12  # avoid divide-by-zero

        score1 = p1 / total
        score2 = p2 / total
        score_bkg = pb / total

        nn_output = np.stack([score1, score2, score_bkg], axis=1)
        nn_output = [row for row in nn_output]

        data[proc] = pd.DataFrame({
            "NN_output": nn_output,
            "weight": weight
        })

    return data

# Helpers

def create_hist(vals, weights=None, bins=50, low=0.0, high=1.0, name="x"):
    h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    h.fill(vals, weight=weights)
    return h


def convert_data_to_tensors(data):
    tensor_data = {}
    for proc, df in data.items():
        tensor_data[proc] = {
            "NN_output": tf.constant(np.stack(df['NN_output'].values), dtype=tf.float32),
            "weight": tf.constant(df['weight'].values, dtype=tf.float32)
        }
    return tensor_data


def compute_significance(h_sig, h_bkgs):
    B = sum(h.values() for h in h_bkgs)
    S = h_sig.values()
    Z = asymptotic_significance(tf.constant(S), tf.constant(B))
    return np.sqrt(np.sum(Z.numpy()**2))

# Model definition
class DiffCatModelExample3D(DiffCatModelMultiDimensional):
    """
    Two channels: signal1 vs (signal2+bkg), signal2 vs (signal1+bkg).
    Loss = -sqrt(Z1 * Z2).
    """
    def call(self, data_dict):
        log_mix = tf.nn.log_softmax(self.mixture_logits)
        scale_tril = self.get_scale_tril()
        means = self.means
        sig1_y = tf.zeros(self.n_cats)
        sig2_y = tf.zeros(self.n_cats)
        bkg_y  = tf.zeros(self.n_cats)
        for proc, tensors in data_dict.items():
            x = tensors['NN_output']  # (N,3)
            w = tensors['weight']     # (N,)
            log_probs = []
            for i in range(self.n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=tf.nn.softmax(means[i]),
                    scale_tril=scale_tril[i]
                )
                log_probs.append(dist.log_prob(x))
            log_probs = tf.stack(log_probs, axis=1)
            log_joint = log_probs + log_mix
            memberships = tf.nn.softmax(log_joint / self.temperature, axis=1)
            yields = tf.reduce_sum(memberships * tf.expand_dims(w,1), axis=0)
            if proc == 'signal1':
                sig1_y += yields
            elif proc == 'signal2':
                sig2_y += yields
            else:
                bkg_y += yields
        Z1 = tf.sqrt(tf.reduce_sum(tf.square(asymptotic_significance(sig1_y, bkg_y + sig2_y))))
        Z2 = tf.sqrt(tf.reduce_sum(tf.square(asymptotic_significance(sig2_y, bkg_y + sig1_y))))
        return -tf.sqrt(Z1 * Z2), tf.reduce_sum(bkg_y)

# Main execution
if __name__ == '__main__':

    path_plots = './examples/3D_2signals_softmax_example/Plots/'
    os.makedirs(path_plots, exist_ok=True)
    # Generate data & convert
    # data = generate_toy_data_3class_3D(seed=42)
    data = generate_toy_data_3class_3D_Nitish(seed=42)
    tensor_data = convert_data_to_tensors(data)

    # Baseline significance with simple binning
    baseline_results = {'signal1':{}, 'signal2':{}}

    # first: plot without argmax reqiurement and many bins:
    for dim in range(3):
        _hists = {}
        for proc, df in data.items():
            print(proc)
            vals = np.stack(df['NN_output'].values)[:,dim]
            _hists[proc] = create_hist(vals, df['weight'].values, bins=50, low=0.0, high=1.0)
            print(_hists[proc])
        for use_log in [True, False]:
            log_suffix = "_log" if use_log else ""
            plot_stacked_histograms(
                stacked_hists=[_hists[p] for p in data.keys() if not p.startswith("signal")],
                process_labels=[p for p in data.keys() if not p.startswith("signal")],
                signal_hists=[100*_hists["signal1"], 500*_hists["signal2"]],
                signal_labels=['Signal1 x100', 'Signal2 x500'],
                log=use_log,
                output_filename=os.path.join(path_plots, f"data_dim_{dim}{log_suffix}.pdf"),
                axis_labels=("NN discriminant node {dim}", "Events"),
            )

    for nbins in [2, 5, 10, 20]:
        baseline_configs = []
        # Signal1 channel
        h_sig1 = None; bkg_h1 = []; bkg_labels1 = []
        for proc, df in data.items():
            vals1 = np.stack(df['NN_output'].values)[:,0]
            mask1 = np.argmax(np.stack(df['NN_output'].values), axis=1) == 0
            if proc == 'signal1':
                h_sig1 = create_hist(vals1[mask1], df['weight'].values[mask1], bins=nbins, low=0.33, high=1.0)
            else:
                bkg_h1.append(create_hist(vals1[mask1], df['weight'].values[mask1], bins=nbins, low=0.33, high=1.0))
                bkg_labels1.append(proc)
        Z1 = compute_significance(h_sig1, bkg_h1)
        baseline_results['signal1'][nbins] = Z1

        # Signal2 channel
        h_sig2 = None; bkg_h2 = []; bkg_labels2 = []
        for proc, df in data.items():
            vals2 = np.stack(df['NN_output'].values)[:,1]
            mask2 = np.argmax(np.stack(df['NN_output'].values), axis=1) == 1
            if proc == 'signal2':
                h_sig2 = create_hist(vals2[mask2], df['weight'].values[mask2], bins=nbins, low=0.33, high=1.0)
            else:
                bkg_h2.append(create_hist(vals2[mask2], df['weight'].values[mask2], bins=nbins, low=0.33, high=1.0))
                bkg_labels2.append(proc)
        Z2 = compute_significance(h_sig2, bkg_h2)
        baseline_results['signal2'][nbins] = Z2

        # Plot baseline histograms
        for sig_name, h_sig, bkgs, labels, xlab, sig_label in [
            ('signal1', h_sig1, bkg_h1, bkg_labels1, 'NN output (signal1)', 'Signal1 x10'),
            ('signal2', h_sig2, bkg_h2, bkg_labels2, 'NN output (signal2)', 'Signal2 x10')
        ]:
            for logscale in [False, True]:
                tag = 'log' if logscale else 'lin'
                fname = os.path.join(path_plots, f'baseline_{sig_name}_{nbins}bins_{tag}.pdf')
                plot_stacked_histograms(
                    stacked_hists=bkgs,
                    process_labels=labels,
                    signal_hists=[10*h_sig],
                    signal_labels=[sig_label],
                    output_filename=fname,
                    axis_labels=(xlab, 'Events'),
                    normalize=False,
                    log=logscale,
                )

    # Optimization via DiffCatModelExample3D
    path_gato_plots = path_plots + "gato/"
    os.makedirs(path_gato_plots, exist_ok=True)

    path_gato_plots_bins_2d = path_gato_plots + "2D_bin_visualizations/"
    os.makedirs(path_gato_plots_bins_2d, exist_ok=True)

    optimized_results = {'signal1':{}, 'signal2':{}}
    for n_cats in [3, 10, 20]:
        @tf.function
        def train_step(model, data_dict, optimizer, lam=0.0):
            with tf.GradientTape() as tape:
                loss, B = model.call(data_dict)
                total = loss + (lam * low_bkg_penalty(B) if lam else 0)
            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total, loss

        model = DiffCatModelExample3D(n_cats=n_cats, dim=3, temperature=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        epochs = 150
        loss_history = []
        for epoch in range(epochs):
            total_loss, base_loss = train_step(model, tensor_data, optimizer)
            loss_history.append(base_loss.numpy())
            if epoch % 10 == 0:
                print(f'[Epoch {epoch}] base_loss={base_loss.numpy():.3f}')

        # Assign bins and fill histograms
        assignments, order, _, inv_map = assign_bins_and_order(model, data)
        for proc in data.keys():
            data[proc]["bin_index"] = assignments[proc]
        filled = {}
        for proc in data.keys():
            filled[proc] = fill_histogram_from_assignments(
                assignments[proc], data[proc]['weight'], n_cats
            )
        # --------- compute optimized significances ----------
        # build explicit background list once
        opt_bkg_hists = [filled['bkg1'], filled['bkg2'], filled['bkg3'], filled['bkg4'], filled['bkg5']]

        # Signal1 channel: other signal + all bkg are background
        Z1_opt = compute_significance(
            filled['signal1'],
            opt_bkg_hists + [filled['signal2']]
        )
        optimized_results['signal1'][n_cats] = Z1_opt

        # Signal2 channel
        Z2_opt = compute_significance(
            filled['signal2'],
            opt_bkg_hists + [filled['signal1']]
        )
        optimized_results['signal2'][n_cats] = Z2_opt


        # --------- plot optimized histograms ----------
        opt_plot_filename = os.path.join(path_gato_plots, f"NN_output_distribution_optimized_{n_cats}bins.pdf")

        # linear plot
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg. 1", "Bkg. 2", "Bkg. 3", "Bkg. 4", "Bkg. 5"],
            signal_hists=[filled['signal1']*10, filled['signal2']*50],
            signal_labels=[r"Signal 1 $\times 10$", r"Signal 2 $\times 50$"],
            output_filename=opt_plot_filename,
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=False
        )
        print(f"Saved optimized linear plot {opt_plot_filename}")

        # log plot
        plot_stacked_histograms(
            stacked_hists=opt_bkg_hists,
            process_labels=["Bkg. 1", "Bkg. 2", "Bkg. 3", "Bkg. 4", "Bkg. 5"],
            signal_hists=[filled['signal1']*10, filled['signal2']*50],
            signal_labels=[r"Signal 1 $\times 10$", r"Signal 2 $\times 50$"],
            output_filename=opt_plot_filename.replace(".pdf", "_log.pdf"),
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=True
        )
        print(f"Saved optimized log plot {opt_plot_filename.replace('.pdf','_log.pdf')}")

        # 1) Plot loss history
        plot_history(
            loss_history,
            os.path.join(path_gato_plots, f"history_loss_{n_cats}bins.pdf"),
            y_label="Negative geometric mean significance",
            x_label="Epoch",
            boundaries=False,
            title="Loss History"
        )

        # 2) Plot learned Gaussian ellipses in all 2D projections
        for (dx, dy) in [(0,1), (0,2), (1,2)]:
            plot_learned_gaussians(
                data=data,
                model=model,
                dim_x=dx,
                dim_y=dy,
                output_filename=os.path.join(
                    path_gato_plots, f"GaussianBlobs_{n_cats}Bins_dims{dx}{dy}.pdf"
                ),
                inv_mapping=inv_map
            )

        path_plots_bins_2d = os.path.join(path_plots, )
        visualize_bins_2d(
            data_dict=data,
            var_label="NN_output",
            n_bins=n_cats,
            path_plot=os.path.join(path_gato_plots_bins_2d, f"{n_cats}_bins.pdf")
        )

    # Comparison plot
    fig, ax = plt.subplots()
    for sig in ['signal1','signal2']:
        base = baseline_results[sig]
        opt  = optimized_results[sig]
        ax.plot(2*np.array(list(base.keys()))+1, np.array(list(base.values())), marker='o', label=f'Baseline {sig}')
        ax.plot(np.array(list(opt.keys())),  np.array(list(opt.values())),  marker='s', linestyle='--', label=f'Optimized {sig}')
    ax.set_xlabel('Number of bins')
    ax.set_ylabel('Significance')
    ax.legend()
    comp_file = os.path.join(path_gato_plots, 'significance_comparison.pdf')
    plt.savefig(comp_file)
    print(f'Saved comparison plot to {comp_file}')