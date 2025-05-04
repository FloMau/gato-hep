import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
# Append the repo root to sys.path so that we can import our core modules.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from diffcat_optimizer.differentiable_categories import asymptotic_significance, gato_gmm_model, low_bkg_penalty, high_bkg_uncertainty_penalty, compute_significance_from_hists
from diffcat_optimizer.plotting_utils import plot_stacked_histograms, plot_history, plot_learned_gaussians, assign_bins_and_order, fill_histogram_from_assignments, plot_bin_boundaries_simplex, plot_yield_vs_uncertainty, plot_significance_comparison
from diffcat_optimizer.utils import df_dict_to_tensors, create_hist
from diffcat_optimizer.data_generation import generate_toy_data_3class_3D


def convert_data_to_tensors(data):
    tensor_data = {}
    for proc, df in data.items():
        tensor_data[proc] = {
            "NN_output": tf.constant(np.stack(df['NN_output'].values), dtype=tf.float32),
            "weight": tf.constant(df['weight'].values, dtype=tf.float32)
        }
    return tensor_data

# Model definition
class gato_3D(gato_gmm_model):
    def call(self, data_dict):
        log_mix = tf.nn.log_softmax(self.mixture_logits)
        scale_tril = self.get_scale_tril()
        means = self.means
        sig1_y = tf.zeros(self.n_cats, dtype=tf.float32)
        sig2_y = tf.zeros(self.n_cats, dtype=tf.float32)
        bkg_y  = tf.zeros(self.n_cats, dtype=tf.float32)
        bkg_w2 = tf.zeros(self.n_cats, dtype=tf.float32)

        for proc, t in data_dict.items():
            x = t["NN_output"] # (N,3)
            w = t["weight"]
            w2 = w ** 2
            # log‑pdf for each component
            log_probs = []
            for i in range(self.n_cats):
                dist = tfd.MultivariateNormalTriL(
                    loc=tf.nn.softmax(means[i]),
                    scale_tril=scale_tril[i]
                )
                log_probs.append(dist.log_prob(x))
            log_probs = tf.stack(log_probs, axis=1) # (N,n_cats)
            memberships = tf.nn.softmax((log_probs + log_mix) / self.temperature, axis=1)

            yields = tf.reduce_sum(memberships * w[:,None],  axis=0)
            sumw2  = tf.reduce_sum(memberships * w2[:,None], axis=0)

            if proc == "signal1":
                sig1_y += yields
            elif proc == "signal2":
                sig2_y += yields
            else:
                bkg_y  += yields
                bkg_w2 += sumw2

        Z1 = tf.sqrt(tf.reduce_sum(asymptotic_significance(sig1_y, bkg_y + sig2_y)**2))
        Z2 = tf.sqrt(tf.reduce_sum(asymptotic_significance(sig2_y, bkg_y + sig1_y)**2))
        return -tf.sqrt(Z1 * Z2), bkg_y, bkg_w2


# Main execution
def main():

    path_plots = './examples/3D_2signals_softmax_example/Plots/'
    os.makedirs(path_plots, exist_ok=True)
    # Generate data & convert
    data = generate_toy_data_3class_3D(
        seed=42,
        noise_scale=0.5,
        n_signal1=int(100000/1), n_signal2=int(100000/1), n_bkg1=int(100000/1), n_bkg2=int(80000/1), n_bkg3=int(50000/1), n_bkg4=int(20000/1), n_bkg5=int(10000/1)
    )
    tensor_data = convert_data_to_tensors(data)

    # Baseline significance with simple binning
    baseline_results = {'signal1':{}, 'signal2':{}}

    # first: plot without argmax reqiurement and many bins:
    for dim in range(3):
        _hists = {}
        for proc, df in data.items():
            vals = np.stack(df['NN_output'].values)[:,dim]
            _hists[proc] = create_hist(vals, df['weight'].values, bins=50, low=0.0, high=1.0)
        for use_log in [True, False]:
            log_suffix = "_log" if use_log else ""
            plot_stacked_histograms(
                stacked_hists=[_hists[p] for p in data.keys() if not p.startswith("signal")],
                process_labels=[p for p in data.keys() if not p.startswith("signal")],
                signal_hists=[100*_hists["signal1"], 500*_hists["signal2"]],
                signal_labels=['Signal1 x100', 'Signal2 x500'],
                log=use_log,
                output_filename=os.path.join(path_plots, f"data_dim_{dim}{log_suffix}.pdf"),
                axis_labels=(f"NN discriminant node {dim}", "Events"),
            )

    for nbins in [2, 5, 10, 20]:
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
        Z1 = compute_significance_from_hists(h_sig1, bkg_h1)
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
        Z2 = compute_significance_from_hists(h_sig2, bkg_h2)
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

    # Optimization via gato_3D
    path_gato_plots = path_plots + "gato/"
    os.makedirs(path_gato_plots, exist_ok=True)

    path_gato_plots_bins_2d = path_gato_plots + "2D_bin_visualizations/"
    os.makedirs(path_gato_plots_bins_2d, exist_ok=True)

    epochs = 300
    lam_yield = 0
    lam_unc = 0

    optimized_results = {'signal1':{}, 'signal2':{}}
    for n_cats in [3, 5, 10]:
        @tf.function
        def train_step(model, tensor_data, optimizer, lam_yield=0.0, lam_unc=0.0):
            with tf.GradientTape() as tape:
                loss, B, B_sumw2 = model.call(tensor_data)

                pen_yield = low_bkg_penalty(B, threshold=10.0, steepness=10.0)
                pen_unc = high_bkg_uncertainty_penalty(B_sumw2, B, rel_threshold=0.20)

                total_loss = loss + lam_yield * pen_yield + lam_unc * pen_unc
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss, loss, pen_yield, pen_unc

        model = gato_3D(n_cats=n_cats, dim=3, temperature=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        loss_history, penalty_yield_history, penalty_unc_history = [], [], []

        for epoch in range(epochs):
            total_loss, loss, pen_yield, pen_unc = train_step(model, tensor_data, optimizer, lam_yield=lam_yield, lam_unc=lam_unc)

            loss_history.append(loss.numpy())
            penalty_yield_history.append(pen_yield.numpy())
            penalty_unc_history.append(pen_unc.numpy())

            if epoch % 10 == 0:
                print(f'[Epoch {epoch}] base_loss={loss.numpy():.3f}')

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
        Z1_opt = compute_significance_from_hists(
            filled['signal1'],
            opt_bkg_hists + [filled['signal2']]
        )
        optimized_results['signal1'][n_cats] = Z1_opt

        # Signal2 channel
        Z2_opt = compute_significance_from_hists(
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

        plot_history(loss_history,
                    os.path.join(path_gato_plots, f"loss_{n_cats}.pdf"),
                    y_label=r"Geometric mean $(Z_1, Z_2)$",
                    x_label="Epoch",
        )

        plot_history(penalty_yield_history,
                    os.path.join(path_gato_plots, f"penalty_yield_{n_cats}.pdf"),
                    y_label="Low-bkg. penalty",
                    x_label="Epoch",
        )

        plot_history(penalty_unc_history,
                    os.path.join(path_gato_plots, f"penalty_unc_{n_cats}.pdf"),
                    y_label="High-unc. penalty",
                    x_label="Epoch",
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

        plot_bin_boundaries_simplex(
            model,
            path_plot=os.path.join(path_gato_plots_bins_2d, f"{n_cats}_bins.pdf"),
        )

        # get hard‑assignment stats (works for multi‑D)
        B_sorted, rel_unc_sorted, _ = model.compute_hard_bkg_stats(tensor_data)
        plot_yield_vs_uncertainty(
            B_sorted,
            rel_unc_sorted,
            output_filename=os.path.join(
                path_gato_plots, f"yield_vs_unc_{n_cats}bins.pdf"
            ),
        )

    plot_significance_comparison(
        baseline_results={
            key: {2*n+1: baseline_results[key][n] for n in baseline_results[key].keys()} for key in baseline_results.keys()
        },
        optimized_results=optimized_results,
        output_filename=os.path.join(path_gato_plots, "significance_comparison.pdf"),
    )


if __name__ == '__main__':

    main()