import os
import argparse
import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# Import necessary modules from the gato package
from gato.data_generation import generate_toy_data_3class_3D
from gato.losses import high_bkg_uncertainty_penalty, low_bkg_penalty
from gato.models import gato_gmm_model
from gato.plotting_utils import (
    assign_bins_and_order,
    fill_histogram_from_assignments,
    plot_bin_boundaries_simplex,
    plot_history,
    plot_learned_gaussians,
    plot_significance_comparison,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
)
from gato.utils import (
    asymptotic_significance,
    compute_significance_from_hists,
    create_hist,
)

tfd = tfp.distributions


# helper: slice to 2-D & convert to tensors
def convert_data_to_tensors_2d(data_dict):
    """
    Keep only P_sig1 (dim0) and P_sig2 (dim1).  Shape → (N,2)
    """
    tensor_data = {}
    for proc, df in data_dict.items():
        full_softmax = np.stack(df["NN_output"].values)      # (N,3)
        two_d = full_softmax[:, :2]                   # drop background coord
        tensor_data[proc] = {
            "NN_output": tf.constant(two_d, dtype=tf.float32),
            "weight"   : tf.constant(df["weight"].values, tf.float32)
        }
    return tensor_data


def convert_data_to_tensors(data):
    tensor_data = {}
    for proc, df in data.items():
        nn = np.stack(df["NN_output"].values)[:, :2]
        w = df["weight"].values
        tensor_data[proc] = {
            "NN_output": tf.constant(nn, dtype=tf.float32),
            "weight"   : tf.constant(w , dtype=tf.float32),
        }
    return tensor_data


#  2-D GMM gato model
class GATO_2D(gato_gmm_model):
    def __init__(self, n_cats, temperature=0.3, name="GATO_2D"):
        super().__init__(n_cats=n_cats,
                         dim=2,
                         temperature=temperature,
                         name=name)

    def call(self, data_dict):
        log_mix = tf.nn.log_softmax(self.mixture_logits)
        scale_tril = self.get_scale_tril()
        means_raw = self.means        # shape (n_cats, 2)
        # Append a zero logit for each component:
        zeros = tf.zeros(
            (tf.shape(means_raw)[0], 1), dtype=means_raw.dtype
        )  # (n_cats,1)
        full_logits = tf.concat([means_raw, zeros], axis=1)   # shape (n_cats,3)
        probs3 = tf.nn.softmax(full_logits, axis=1)           # shape (n_cats,3)
        locs = probs3[:, :self.dim]                           # shape (n_cats,2)

        sig1_y = tf.zeros(self.n_cats, tf.float32)
        sig2_y = tf.zeros(self.n_cats, tf.float32)
        bkg_y = tf.zeros(self.n_cats, tf.float32)
        bkg_w2 = tf.zeros(self.n_cats, tf.float32)

        for proc, t in data_dict.items():
            x = t["NN_output"]            # (N,2)
            w = t["weight"]
            w2 = w**2

            # per-component log-pdfs
            lp = []
            for k in range(self.n_cats):
                lp.append(
                    tfd.MultivariateNormalTriL(
                        loc=locs[k],
                        scale_tril=scale_tril[k]
                    ).log_prob(x)
                )
            lp = tf.stack(lp, axis=1)     # (N,k)
            gamma = tf.nn.softmax((lp + log_mix) / self.temperature, axis=1)

            y = tf.reduce_sum(gamma * w[:,None],  axis=0)
            y_w2 = tf.reduce_sum(gamma * w2[:,None], axis=0)

            if proc == "signal1":
                sig1_y += y
            elif proc == "signal2":
                sig2_y += y
            else:
                bkg_y += y
                bkg_w2 += y_w2

        Z1 = tf.sqrt(tf.reduce_sum(asymptotic_significance(sig1_y, bkg_y + sig2_y)**2))
        Z2 = tf.sqrt(tf.reduce_sum(asymptotic_significance(sig2_y, bkg_y + sig1_y)**2))
        loss = -tf.sqrt(Z1 * Z2)
        return loss, bkg_y, bkg_w2


# Main function to run the example
def main():
    parser = argparse.ArgumentParser(description="2-D soft-max GATO optimisation")
    parser.add_argument("--epochs",      type=int,   default=250)
    parser.add_argument("--gato-bins",   nargs="+",  type=int, default=[3,5,10])
    parser.add_argument("--lam-yield",   type=float, default=0.)
    parser.add_argument("--lam-unc",     type=float, default=0.)
    parser.add_argument("--thr-yield",   type=float, default=5.)
    parser.add_argument("--thr-unc",     type=float, default=0.20)
    parser.add_argument("--n-bkg",       type=int,   default=1_000_000)
    parser.add_argument("--out",         type=str,   default="Plots")
    args = parser.parse_args()

    path_plots = f'./examples/three_class_softmax_example/{args.out}/'
    os.makedirs(path_plots, exist_ok=True)

    # ---------- toy data
    data = generate_toy_data_3class_3D(
        seed=42,
        noise_scale=0.3,
        n_signal1=100_000, n_signal2=100_000,
        n_bkg=args.n_bkg
    )
    tensor_data = convert_data_to_tensors(data)

    # ---------- dataset plots (only 2 dims kept)
    # plot to show dataset
    for dim in range(3):              # show 0, 1 and 2
        _h = {}
        for proc, df in data.items():
            vals = np.stack(df['NN_output'].values)[:, dim]   # full 3-D array here
            _h[proc] = create_hist(
                vals, df['weight'].values, bins=50, low=0.0, high=1.0
            )
        for proc, df in data.items():
            vals = np.stack(df["NN_output"].values)[:,dim]
            _h[proc] = create_hist(vals, df["weight"].values, bins=50, low=0., high=1.)
        for log in (False, True):
            suf = "_log" if log else ""
            plot_stacked_histograms(
                stacked_hists=[_h[p] for p in data if not p.startswith("signal")],
                process_labels=[p for p in data if not p.startswith("signal")],
                signal_hists=[100*_h["signal1"], 500*_h["signal2"]],
                signal_labels=['Signal1 x 100', 'Signal2 x 500'],
                log=log,
                output_filename=os.path.join(path_plots, f"data_dim{dim}{suf}.pdf"),
                axis_labels=(f"soft-max dim {dim}", "Events"),
            )

    # ---------- baseline (same as before)
    baseline_results = {'signal1':{}, 'signal2':{}}
    for nb in [2,5,10]:
        # …. identical code, just care about dim==0 and dim==1 …

        # ---- channel 1
        h_sig1 = None
        bkg1 = []
        lbl1 = []
        for p,df in data.items():
            v = np.stack(df["NN_output"].values)[:,0]
            m = np.argmax(np.stack(df["NN_output"].values), 1) == 0
            if p == 'signal1':
                h_sig1 = create_hist(
                    v[m], df["weight"].values[m], bins=nb, low=0.33, high=1.
                )
            else:
                bkg1.append(
                    create_hist(
                        v[m], df["weight"].values[m], bins=nb, low=0.33, high=1.
                        )
                )
                lbl1.append(p)
        baseline_results['signal1'][nb] = compute_significance_from_hists(h_sig1,bkg1)

        # ---- channel 2
        h_sig2 = None
        bkg2 = []
        lbl2 = []
        for p,df in data.items():
            v = np.stack(df["NN_output"].values)[:,1]
            m = np.argmax(np.stack(df["NN_output"].values),1) == 1
            if p == 'signal2':
                h_sig2 = create_hist(
                    v[m], df["weight"].values[m], bins=nb, low=0.33, high=1.
                )
            else:
                bkg2.append(
                    create_hist(
                        v[m], df["weight"].values[m], bins=nb, low=0.33, high=1.
                    )
                )
                lbl2.append(p)
        baseline_results['signal2'][nb] = compute_significance_from_hists(h_sig2,bkg2)

    # ---------- GATO optimisation
    gato_results = {'signal1':{}, 'signal2':{}}
    path_gato = os.path.join(path_plots, "gato")
    os.makedirs(path_gato, exist_ok=True)

    for n_cats in args.gato_bins:

        @tf.function
        def train_step(model, tdata, opt, lamY, lamU, thrY, thrU):
            with tf.GradientTape() as tape:
                loss, B, Bw2 = model.call(tdata)
                penY = low_bkg_penalty(B,     threshold=thrY)
                penU = high_bkg_uncertainty_penalty(Bw2, B, rel_threshold=thrU)
                total = loss + lamY*penY + lamU*penU
            g = tape.gradient(total, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
            return loss

        model = GATO_2D(n_cats=n_cats, temperature=0.5)
        optimizer = tf.keras.optimizers.Adam(0.02)

        loss_history = []
        for ep in range(args.epochs):
            loss = train_step(
                model, tensor_data, optimizer,
                args.lam_yield, args.lam_unc,
                args.thr_yield, args.thr_unc
            )
            if ep % 10 == 0:
                print(f"[{ep:03d}] loss = {loss.numpy():.3f}")
            loss_history.append(loss.numpy())

        # ---- bin assignment
        data_2d = {}
        for proc, df in data.items():
            df2 = df.copy()
            df2["NN_output"] = [v[:2] for v in df["NN_output"].values]  # slice each row
            data_2d[proc] = df2

        assign, order, _, inv = assign_bins_and_order(model, data_2d, reduce=True)

        # 2) make per-process histograms
        filled = {p: fill_histogram_from_assignments(
            assign[p], data_2d[p]["weight"], n_cats
        ) for p in data_2d}

        opt_bkgs = [filled[f"bkg{i}"] for i in range(1,6)]
        Z1 = compute_significance_from_hists(
            filled["signal1"], opt_bkgs+[filled["signal2"]]
        )
        Z2 = compute_significance_from_hists(
            filled["signal2"], opt_bkgs+[filled["signal1"]]
        )
        gato_results['signal1'][n_cats] = Z1
        gato_results['signal2'][n_cats] = Z2

        # ---- quick plots
        plot_learned_gaussians(
            data=data, model=model, dim_x=0, dim_y=1,
            output_filename=os.path.join(path_gato, f"Gaussians_{n_cats}bins.pdf"),
            inv_mapping=inv,
            reduce=True,
        )

        plot_bin_boundaries_simplex(
            model,
            order,
            path_plot=os.path.join(path_gato, f"Bin_boundaries_{n_cats}_bins.pdf"),
            reduce=True
        )

        # loss curve
        plot_history(
            np.array(loss_history),
            os.path.join(path_gato, f"loss_{n_cats}.pdf"),
            y_label=r"Geometric mean $(Z_1,Z_2)$", x_label="Epoch"
        )

        # 1) Stacked histogram of optimized bins:
        # Collect background processes
        bg_procs = [p for p in data if not p.startswith("signal")]
        opt_bkgs = [filled[p] for p in bg_procs]
        sig1_scale = 100
        sig2_scale = 500

        # Plot linear-scale stacked histogram:
        plot_stacked_histograms(
            stacked_hists=opt_bkgs,
            process_labels=bg_procs,
            signal_hists=[
                sig1_scale * filled["signal1"], sig2_scale * filled["signal2"]
            ],
            signal_labels=[
                f"Signal1 x{sig1_scale}", f"Signal2 x{sig2_scale}"
            ],
            output_filename=os.path.join(
                path_gato, f"optimized_dist_{n_cats}bins_linear.pdf"
            ),
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=False
        )
        print(
            f"Saved optimized linear histogram: optimized_dist_{n_cats}bins_linear.pdf"
        )

        # Plot log-scale stacked histogram:
        plot_stacked_histograms(
            stacked_hists=opt_bkgs,
            process_labels=bg_procs,
            signal_hists=[
                sig1_scale * filled["signal1"], sig2_scale * filled["signal2"]
            ],
            signal_labels=[
                f"Signal1 x{sig1_scale}", f"Signal2 x{sig2_scale}"
            ],
            output_filename=os.path.join(
                path_gato, f"optimized_dist_{n_cats}bins_log.pdf"
            ),
            axis_labels=("Bin index", "Events"),
            normalize=False,
            log=True
        )
        print(
            f"Saved optimized log histogram: optimized_dist_{n_cats}bins_log.pdf"
        )

        # 2) Yield vs. relative uncertainty plot for background bins:
        B_sorted, rel_unc_sorted, _ = model.compute_hard_bkg_stats(tensor_data)
        # reorder according to the sorted bin order:
        B_ord = B_sorted[order]
        unc_ord = rel_unc_sorted[order]

        # Linear
        plot_yield_vs_uncertainty(
            B_ord,
            unc_ord,
            output_filename=os.path.join(
                path_gato, f"yield_vs_unc_{n_cats}bins_linear.pdf"
            ),
            log=False
        )
        print(f"Saved yield vs. unc (linear): yield_vs_unc_{n_cats}bins_linear.pdf")

        # Log
        plot_yield_vs_uncertainty(
            B_ord,
            unc_ord,
            log=True,
            output_filename=os.path.join(
                path_gato, f"yield_vs_unc_{n_cats}bins_log.pdf"
            )
        )
        print(f"Saved yield vs. unc (log): yield_vs_unc_{n_cats}bins_log.pdf")

    # ---------- summary comparison
    plot_significance_comparison(
        baseline_results={
            k:{
                2*n+1:baseline_results[k][n] for n in baseline_results[k]
            } for k in baseline_results
        },
        optimized_results=gato_results,
        output_filename=os.path.join(path_gato, "significance_comparison.pdf"),
    )


if __name__ == "__main__":
    main()
