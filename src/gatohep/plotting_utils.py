import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Ellipse

plt.style.use(hep.style.ROOT)
tfd = tfp.distributions


def plot_stacked_histograms(
    stacked_hists,
    process_labels,
    output_filename="./plot.pdf",
    axis_labels=("x-axis", "Events"),
    signal_hists=None,
    signal_labels=None,
    normalize=False,
    log=False,
    log_min=None,
    return_figure=False,
    ax=None,
):
    """
    Plot stacked histograms for backgrounds and overlay signal histograms.

    Parameters
    ----------
    stacked_hists : list of hist.Hist
        List of histograms for background processes.
    process_labels : list of str
        List of labels for background processes.
    output_filename : str, optional
        File name to save the figure. Default is "./plot.pdf".
    axis_labels : tuple of str, optional
        Labels for the x-axis and y-axis. Default is ("x-axis", "Events").
    signal_hists : list of hist.Hist, optional
        List of histograms for signal processes. Default is None.
    signal_labels : list of str, optional
        List of labels for signal histograms. Default is None.
    normalize : bool, optional
        If True, normalize the histograms. Default is False.
    log : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is False.
    log_min : float, optional
        Minimum value for the y-axis in log scale. Default is None.
    return_figure : bool, optional
        If True, return the figure and axis instead of saving. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Default is None.

    Returns
    -------
    None or tuple
        If `return_figure` is True, returns (fig, ax). Otherwise, saves the plot.
    """
    # Normalization if requested.
    if normalize:
        stack_integral = sum([_hist.sum().value for _hist in stacked_hists])
        stacked_hists = [_hist / stack_integral for _hist in stacked_hists]
        if signal_hists:
            for i, sig in enumerate(signal_hists):
                integral_ = sig.sum().value
                if integral_ > 0:
                    signal_hists[i] = sig / integral_

    # Prepare binning from the first histogram.
    bin_edges = stacked_hists[0].to_numpy()[1]

    # Gather values and uncertainties for each background histogram.
    mc_values_list = [_hist.values() for _hist in stacked_hists]
    mc_errors_list = [np.sqrt(_hist.variances()) for _hist in stacked_hists]

    # Setup figure and axis.
    if ax is None:
        fig, ax_main = plt.subplots(figsize=(10, 9))
    else:
        fig = None
        ax_main = ax

    # Plot stacked backgrounds.
    hep.histplot(
        mc_values_list,
        label=process_labels,
        bins=bin_edges,
        stack=True,
        histtype="fill",
        edgecolor="black",
        linewidth=1,
        yerr=mc_errors_list,
        ax=ax_main,
        alpha=0.8,
    )

    # Add an uncertainty band for the total MC (background) if desired.
    mc_total = np.sum(mc_values_list, axis=0)
    mc_total_var = np.sum([err**2 for err in mc_errors_list], axis=0)
    mc_total_err = np.sqrt(mc_total_var)
    hep.histplot(
        mc_total,
        bins=bin_edges,
        histtype="band",
        yerr=mc_total_err,
        ax=ax_main,
        alpha=0.5,
        label=None,  # No legend entry for the band.
    )

    # Overlay signal histograms if provided.
    if signal_hists:
        for sig_hist, label in zip(signal_hists, signal_labels):
            sig_values = sig_hist.values()
            sig_errors = np.sqrt(sig_hist.variances())
            hep.histplot(
                [sig_values],
                label=[label],
                bins=bin_edges,
                linewidth=3,
                linestyle="--",
                yerr=sig_errors,
                ax=ax_main,
            )

    # Final styling.
    ax_main.set_xlabel(axis_labels[0], fontsize=26)
    ax_main.set_ylabel(axis_labels[1], fontsize=26)
    ax_main.margins(y=0.15)
    if log:
        ax_main.set_yscale("log")
        ax_main.set_ylim(ax_main.get_ylim()[0], 30 * ax_main.get_ylim()[1])
        if log_min is not None:
            ax_main.set_ylim(log_min, ax_main.get_ylim()[1])
    else:
        ax_main.set_ylim(0, 1.25 * ax_main.get_ylim()[1])
        ax_main.tick_params(labelsize=22)
    ax_main.tick_params(labelsize=24)

    handles, labels = ax_main.get_legend_handles_labels()
    ncols = 2 if len(labels) < 6 else 3
    ax_main.legend(
        loc="upper right", fontsize=18, ncols=ncols, labelspacing=0.4, columnspacing=1.5
    )

    # Save or return the figure.
    if not return_figure:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close(fig)
    else:
        return fig, ax_main
    return None


def plot_history(
    history_data,
    output_filename,
    y_label="Value",
    x_label="Epoch",
    boundaries=False,
    title=None,
    log_scale=False,
):
    """
    Plot the training history of a model.

    Parameters
    ----------
    history_data : list or ndarray
        History data to plot.
    output_filename : str
        File name to save the plot.
    y_label : str, optional
        Label for the y-axis. Default is "Value".
    x_label : str, optional
        Label for the x-axis. Default is "Epoch".
    boundaries : bool, optional
        If True, plot boundaries instead of scalar values. Default is False.
    title : str, optional
        Title of the plot. Default is None.
    log_scale : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is False.

    Returns
    -------
    None
    """
    epochs = np.arange(len(history_data))
    fig, ax = plt.subplots(figsize=(8, 6))

    if not boundaries:  # scalar history
        ax.plot(epochs, history_data, marker="o")
    else:  # matrix (epochs, n_tracks)
        values = np.asarray(history_data, dtype=float)
        n_trk = values.shape[1]
        cmap = plt.get_cmap("tab20", n_trk)
        for t in range(n_trk):
            ax.plot(
                epochs,
                values[:, t],
                marker="o",
                markersize=3,
                color=cmap(t),
                label=f"Boundary {t + 1}",
            )

    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    if title:
        ax.set_title(title, fontsize=22)
    ax.legend(
        ncol=2,
        fontsize=18,
        markerscale=0.6,
        labelspacing=0.25,
        handlelength=1.0,
        handletextpad=0.4,
    )
    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.savefig(output_filename)
    plt.close(fig)


def assign_bins_and_order(model, data, reduce=False, eps=1e-6):
    """
    Assign events to bins and compute bin significance.

    Parameters
    ----------
    model : tf.Module
        Trained model with Gaussian components.
    data : dict
        Dictionary of input data with process names as keys.
    reduce : bool, optional
        If True, reduce dimensionality. Default is False.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-6.

    Returns
    -------
    tuple
        A tuple containing bin assignments, bin order, significances,
        and inverse mapping.
    """
    n_cats = model.n_cats

    # Retrieve learned parameters.
    log_mix = tf.nn.log_softmax(model.mixture_logits)  # shape: (n_cats,)
    scale_tril = model.get_scale_tril()  # shape: (n_cats, dim, dim)
    means = model.means  # shape: (n_cats, dim)

    if reduce:
        zeros = tf.zeros((tf.shape(means)[0], 1), dtype=means.dtype)
        full_logits = tf.concat([means, zeros], axis=1)
        probs3 = tf.nn.softmax(full_logits)
        locs = probs3.numpy()[:, :-1]
    else:
        locs = tf.nn.softmax(means)

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
            dist = tfd.MultivariateNormalTriL(loc=locs[i], scale_tril=scale_tril[i])
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
            mask = assignments == i
            yield_sum = weights[mask].sum()
            if proc.startswith("signal"):
                S_yields[i] += yield_sum
            else:
                B_yields[i] += yield_sum

    # Compute a significance measure per bin. (Here we use S/sqrt(B))
    significances = S_yields / (np.sqrt(B_yields) + eps)

    # Order: sort original bin indices in ascending order (lowest significance first)
    order = np.argsort(significances)  # e.g., [orig_bin_low, ..., orig_bin_high]
    new_order_mapping = {orig: new for new, orig in enumerate(order)}
    # And the inverse mapping: new index -> original index.
    inv_mapping = {v: k for k, v in new_order_mapping.items()}

    # Remap bin assignments.
    for proc in bin_assignments:
        orig_assign = bin_assignments[proc]
        bin_assignments[proc] = np.vectorize(lambda i: new_order_mapping[i])(
            orig_assign
        )

    return bin_assignments, order, significances, inv_mapping


def fill_histogram_from_assignments(assignments, weights, nbins, name="BinAssignments"):
    """
    Fill a histogram from event assignments and weights.

    Parameters
    ----------
    assignments : array_like
        Array of bin assignments for each event.
    weights : array_like
        Array of weights for each event.
    nbins : int
        Number of bins in the histogram.
    name : str, optional
        Name of the histogram axis. Default is "BinAssignments".

    Returns
    -------
    hist.Hist
        A histogram object.
    """
    # Create a histogram with nbins bins, ranging from 0 to nbins.
    h = hist.Hist.new.Reg(nbins, 0, nbins, name=name).Weight()
    h.fill(assignments, weight=weights)
    return h


def plot_learned_gaussians(
    data: dict,
    model,
    dim_x: int,
    dim_y: int,
    output_filename: str,
    conf_level: float = 2.30,  # chi2 for 1-sigma in 2 D
    inv_mapping: dict | None = None,
    reduce: bool = False,
):
    """
    Visualise the learned Gaussian components (2-D projection) together
    with a scatter of real data points.

    Parameters
    ----------
    data : dict
        ``{process: DataFrame}`` - each DataFrame must contain a column
        ``"NN_output"`` with arrays of shape (dim,).
    model : gato_gmm_model
        Trained multi-D GMM.
    dim_x, dim_y : int
        Component of the NN output plotted on x / y axis.
    output_filename : str
        Path where the figure is saved.
    conf_level : float, optional
        Chi-square value that defines the ellipse (2.30 -> 1 sigma).
    inv_mapping : dict, optional
        Map new bin index → original component index.  If None,
        identity mapping is used.
    reduce : bool, optional
        If True, map raw mean logits through softmax *per component*
        before projecting.  (Keeps prior behaviour.)
    """
    # 1) pull parameters from helper methods
    comp = model.get_mixture_pdf().components_distribution
    means = comp.loc.numpy()  # (k, dim)
    cov_full = np.matmul(
        comp.scale_tril.numpy(), np.transpose(comp.scale_tril.numpy(), (0, 2, 1))
    )  # (k, dim, dim)
    weights = np.exp(model.get_mixture_weight().numpy())  # (k,)
    n_cats = means.shape[0]

    # mapping new->original index
    if inv_mapping is None:
        inv_mapping = {i: i for i in range(n_cats)}

    # 2) scatter a subset of the data
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = {
        "signal": "tab:red",
        "bkg1": "tab:blue",
        "bkg2": "tab:orange",
        "bkg3": "tab:cyan",
    }
    markers = {"signal": "o", "bkg1": "s", "bkg2": "v", "bkg3": "d"}

    stop = 1000
    for proc, df in data.items():
        arr = np.stack(df["NN_output"].values)
        ax.scatter(
            arr[:stop, dim_x],
            arr[:stop, dim_y],
            s=10,
            alpha=0.3,
            color=colours.get(proc, "gray"),
            marker=markers.get(proc, "o"),
            label=proc,
        )

    # 3) ellipses per component
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()
    base_cols = prop_cycle["color"]
    linestyles = ["solid", "dashed", "dotted", "dashdot"] * 100
    colours = (base_cols * 100)[:n_cats]

    for new_idx in range(n_cats):
        orig = inv_mapping[new_idx]

        # mean projection
        if reduce:
            # treat raw mean logits as 3-vector on simplex (old behaviour)
            mean_raw = means[orig]
            full_logits = tf.concat([mean_raw, [0.0]], axis=0)
            mu = tf.nn.softmax(full_logits).numpy()[:-1]
        else:
            mu = means[orig][[dim_x, dim_y]]

        # 2-D covariance slice
        cov = cov_full[orig][np.ix_([dim_x, dim_y], [dim_x, dim_y])]

        # ellipse parameters
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
        width = 2 * np.sqrt(conf_level * eigval[0])
        height = 2 * np.sqrt(conf_level * eigval[1])

        alpha = max(0.3, weights[orig] / weights.max())
        ellipse = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            edgecolor=colours[new_idx],
            fc="none",
            lw=3,
            linestyle=linestyles[new_idx],
            alpha=alpha,
            label=f"Gaussian {new_idx}",
        )
        ax.add_patch(ellipse)

    ax.set_xlabel(f"Dimension {dim_x}", fontsize=18)
    ax.set_ylabel(f"Dimension {dim_y}", fontsize=18)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.legend(fontsize=12, ncol=3)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)


def get_distinct_colors(n):
    """
    Generate a list of n distinct RGB colors.

    Parameters
    ----------
    n : int
        Number of distinct colors to generate.

    Returns
    -------
    list of tuple
        List of RGB color tuples.
    """
    hsv = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))
    return [tuple(rgb[:3]) for rgb in hsv]


def plot_bin_boundaries_2D(
    model,
    bin_order,
    path_plot,
    *,
    resolution: int = 1000,
    reduce: bool = False,  # kept for API compatibility, ignored for dim==2
):
    """
    Plot hard-bin regions of a *2-D* GMM on the 2-simplex face
    (x >= 0, y >= 0, x + y ≤ 1).

    Parameters
    ----------
    model : gato_gmm_model   (must have dim == 2)
    bin_order : list[int]
        Desired plotting order of the components.
    path_plot : str
        Destination file (PDF, PNG, …).
    resolution : int, optional
        Grid resolution per axis.  Default 500.
    reduce : bool, optional
        Ignored (for backward compatibility with older callers).
    """
    if model.dim != 2:
        raise ValueError("This helper expects a 2D model.")

    os.makedirs(os.path.dirname(path_plot), exist_ok=True)

    # 1)  Build triangular grid inside the simplex
    xs = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    mask = X + Y <= 1.0  # boolean mask
    pts = np.stack([X[mask], Y[mask]], axis=-1)  # (P, 2)

    # 2)  Ask the model which bin each point belongs to
    bin_ids = model.get_bin(tf.constant(pts, dtype=tf.float32)).numpy()  # (P,)

    # Map original bin indices -> desired plotting order
    #   original index  = bin_ids[i]
    #   plotting index  = bin_order.index(original)
    inv_map = np.empty_like(bin_order)
    for new_idx, orig_idx in enumerate(bin_order):
        inv_map[orig_idx] = new_idx
    bin_ids_plot = inv_map[bin_ids]  # (P,)

    assign = np.full(X.shape, np.nan)
    assign[mask] = bin_ids_plot

    # build color list from current cycle + tab colors
    base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tab = ["tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:brown", "black"]
    needed = model.n_cats - (len(base) + len(tab))
    extra = [] if needed < 1 else get_distinct_colors(needed)
    colors = (base + tab + extra)[: model.n_cats]

    cmap = ListedColormap(colors)
    bounds = np.arange(len(bin_order) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(bin_order))

    # 4)  Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, assign, levels=bounds, cmap=cmap, norm=norm, alpha=0.6)
    ax.contour(X, Y, assign, levels=bounds, colors="k", linewidths=0.8)

    # label the middle of each coloured region
    for b in range(len(bin_order)):
        xi = X[assign == b]
        yi = Y[assign == b]
        if xi.size:
            ax.text(
                xi.mean(),
                yi.mean(),
                str(b),
                color=colors[b],
                fontsize=15,
                ha="center",
                va="center",
            )

    proxies = [
        plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in range(len(bin_order))
    ]
    ax.legend(
        proxies,
        [f"Bin {b}" for b in range(len(bin_order))],
        fontsize=12,
        ncol=2,
        loc="upper right",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Discriminant dim. 0", fontsize=24)
    ax.set_ylabel("Discriminant dim. 1", fontsize=24)

    plt.tight_layout()
    plt.savefig(path_plot)
    plt.close(fig)


def plot_yield_vs_uncertainty(
    B_sorted,
    rel_unc_sorted,
    output_filename,
    log=False,
    x_label="Bin index",
    y_label_left="Background yield",
    y_label_right="Rel. stat. unc.",
    fig_size=(8, 6),
    bar_kwargs_left=None,
    bar_kwargs_right=None,
):
    """
    Plot background yield and relative uncertainty as dual-axis bar plots.

    Parameters
    ----------
    B_sorted : array_like
        Sorted background yields.
    rel_unc_sorted : array_like
        Sorted relative uncertainties.
    output_filename : str
        File name to save the plot.
    log : bool, optional
        If True, use a logarithmic scale for the y-axis. Default is False.
    x_label : str, optional
        Label for the x-axis. Default is "Bin index".
    y_label_left : str, optional
        Label for the left y-axis. Default is "Background yield".
    y_label_right : str, optional
        Label for the right y-axis. Default is "Rel. stat. unc.".
    fig_size : tuple, optional
        Size of the figure. Default is (8, 6).
    bar_kwargs_left : dict, optional
        Additional arguments for the left bar plot. Default is None.
    bar_kwargs_right : dict, optional
        Additional arguments for the right bar plot. Default is None.

    Returns
    -------
    None
    """
    B_sorted = np.asarray(B_sorted)
    rel_unc_sorted = np.asarray(rel_unc_sorted)
    bins = np.arange(len(B_sorted))
    width = 0.4  # bar width
    fontsize = 22

    # default styles
    left_style = {"alpha": 0.6, "color": "C0", "width": width}
    right_style = {"alpha": 0.6, "color": "C1", "width": width}

    if bar_kwargs_left:
        left_style.update(bar_kwargs_left)
    if bar_kwargs_right:
        right_style.update(bar_kwargs_right)

    fig, ax1 = plt.subplots(figsize=fig_size)

    # background yield, shifted left
    ax1.bar(bins - width / 2, B_sorted, **left_style)
    ax1.set_ylabel(y_label_left, color=left_style["color"], fontsize=fontsize)
    ax1.tick_params(axis="y", colors=left_style["color"])
    ax1.spines["left"].set_color(left_style["color"])

    if log:
        ax1.set_yscale("log")

    # relative uncertainty, shifted right
    ax2 = ax1.twinx()
    ax2.bar(bins + width / 2, rel_unc_sorted, **right_style)
    ax2.set_ylabel(y_label_right, color=right_style["color"], fontsize=fontsize)
    ax2.tick_params(axis="y", colors=right_style["color"])
    ax2.spines["right"].set_color(right_style["color"])

    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax1.set_xticks(bins)
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)


def plot_significance_comparison(
    baseline_results, optimized_results, output_filename, fig_size=(8, 6)
):
    """
    Plot baseline vs. optimized significance for multiple signals.

    Parameters
    ----------
    baseline_results : dict
        Dictionary mapping signal names to baseline significance results.
    optimized_results : dict
        Dictionary mapping signal names to optimized significance results.
    output_filename : str
        File name to save the plot.
    fig_size : tuple, optional
        Size of the figure. Default is (8, 6).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # pick distinct markers for baseline vs. optimized:
    base_style = {"marker": "o", "linestyle": "-"}
    opt_style = {"marker": "s", "linestyle": "--"}

    for sig in baseline_results:
        # get sorted bins & values
        b_bins = np.array(sorted(baseline_results[sig].keys()))
        b_Z = np.array([baseline_results[sig][nb] for nb in b_bins])

        o_bins = np.array(sorted(optimized_results[sig].keys()))
        o_Z = np.array([optimized_results[sig][nb] for nb in o_bins])

        ax.plot(b_bins, b_Z, label=f"Equidistant binning {sig}", **base_style)
        ax.plot(o_bins, o_Z, label=f"GATO binning {sig}", **opt_style)

    ax.set_xlabel("Number of bins", fontsize=22)
    ax.set_ylabel("Significance", fontsize=22)
    ax.legend(fontsize=14)
    ax.set_xlim(0, max(ax.get_xlim()[1], max(b_bins.max(), o_bins.max()) * 1.05))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    plt.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)


def plot_gmm_1d(model, output_filename, x_range=(0.0, 1.0), n_points=10_000):
    """
    Plot each weighted component of a 1D GMM.

    Parameters
    ----------
    model : gato_gmm_model   (dim == 1)
    output_filename : str
    x_range : tuple(float, float)
    n_points : int
    """

    gmm = model.get_mixture_pdf()
    comp = gmm.components_distribution  # MultivariateNormalTriL(k)
    weights = tf.exp(model.get_mixture_weight())  # shape (k,)

    x = np.linspace(*x_range, n_points, dtype=np.float32)  # (N,)
    x_tf = tf.constant(x[:, None, None])  # (N, 1, 1)
    x_tf = tf.broadcast_to(x_tf, (n_points, model.n_cats, 1))  # (N, k, 1)

    pdf_matrix = comp.prob(x_tf[:, None, :]).numpy()  # (N, k)
    pdf_matrix = comp.prob(x_tf).numpy()  # (N, k)
    weighted_pdfs = pdf_matrix * weights.numpy()  # broadcast multiply

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(colors) < model.n_cats:  # extend colour list if needed
        colors += [f"C{i}" for i in range(len(colors), model.n_cats)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(model.n_cats):
        ax.plot(x, weighted_pdfs[:, k], lw=3, color=colors[k], label=f"Comp. {k}")

    ax.set_xlim(*x_range)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("Weighted PDF", fontsize=22)
    ax.legend(fontsize=14, ncol=2)
    plt.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)


def make_gif(frame_files, out_name, interval=800):
    fig = plt.figure(figsize=(6, 4))
    plt.axis("off")

    ims = []
    for fname in frame_files:
        img = plt.imread(fname)
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims,
        interval=interval,
        blit=True,
        repeat_delay=1000
    )
    # This requires that pillow is available (it's a dependency of matplotlib)
    ani.save(out_name, writer="pillow")
    plt.close(fig)
