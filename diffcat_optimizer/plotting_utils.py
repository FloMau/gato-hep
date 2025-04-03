import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep  # assuming you use mplhep for histplot
plt.style.use(hep.style.ROOT)
import tensorflow as tf


def plot_stacked_histograms(
    stacked_hists,           # list of hist.hist objects for backgrounds
    process_labels,          # list of labels for backgrounds
    output_filename="./plot.pdf",
    axis_labels=("x-axis", "Events"),
    signal_hists=None,       # optional list of hist.hist objects for signals
    signal_labels=None,      # optional labels for signal histograms
    normalize=False,
    log=False,
    log_min=None,
    include_flow=False,
    colors=None,
    return_figure=False,
    ax=None,
):
    """
    Plots stacked histograms for backgrounds and overlays signal histograms.
    This is a simplified version that drops ratio panels, data hist, and CMS labels.

    Parameters:
      - stacked_hists: list of hist.hist objects (backgrounds).
      - process_labels: list of strings for background process names.
      - output_filename: file name to save the figure.
      - axis_labels: tuple with (x-axis label, y-axis label).
      - signal_hists: list of hist.hist objects for signals (optional).
      - signal_labels: list of labels for signal histograms (optional).
      - normalize: if True, normalize the histograms.
      - log: if True, use log scale on the y-axis.
      - log_min: if provided, set the y-axis lower limit.
      - include_flow: if True, include overflow/underflow (functionality not implemented here).
      - colors: list of colors for the backgrounds.
      - return_figure: if True, return (fig, ax) instead of saving.
      - ax: if provided, plot on the given axes.
    """
    
    # Optionally include overflow/underflow here if needed (not implemented in this version)
    # if include_flow:
    #     stacked_hists = [include_overflow_underflow(h) for h in stacked_hists]
    #     if signal_hists:
    #         signal_hists = [include_overflow_underflow(h) for h in signal_hists]
    
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
    # We assume that each hist has one axis and use its bin edges.
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
        # color=colors,
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
    ax_main.legend(loc="upper right", fontsize=18, ncols=ncols, labelspacing=0.4, columnspacing=1.5)
    
    # Save or return the figure.
    if not return_figure:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close(fig)
    else:
        return fig, ax_main

def plot_history(
    history_data,
    output_filename,
    y_label="Value",
    x_label="Epoch",
    boundaries=False,
    title=None,
    log_scale=False
):
    """
    Plots a 1D array or a list-of-lists over epochs.
    
    Parameters
    ----------
    history_data : list
        - If boundaries=False, a simple list of scalar floats (e.g. loss each epoch).
        - If boundaries=True, a list of lists. Each history_data[i] is a list
          of boundary positions at epoch i (for a 1D categorization).
    output_filename : str
        Where to save the resulting PDF plot.
    y_label : str
        Label for the y-axis.
    x_label : str
        Label for the x-axis.
    boundaries : bool
        If False, we assume a scalar (1D) history => plot one line of y vs epoch.
        If True, we assume each item in history_data[i] is a list of boundary positions => 
        we draw multiple lines, one for each boundary index.
    title : str
        (Optional) title for the plot.
    log_scale : bool
        Whether to set the y-axis to log scale.
    """
    if not history_data:
        print("[plot_history] No data to plot.")
        return
    
    epochs = np.arange(len(history_data))
    fig, ax = plt.subplots(figsize=(8, 6))

    if not boundaries:
        # history_data is assumed to be a list of floats
        ax.plot(epochs, history_data, marker='o', label=y_label)
    else:
        # history_data is a list of lists: boundary positions at each epoch
        max_nb = max(len(b) for b in history_data)  # how many boundaries in total
        for boundary_idx in range(max_nb):
            boundary_vals = []
            for i in range(len(history_data)):
                b_list = history_data[i]
                if boundary_idx < len(b_list):
                    boundary_vals.append(b_list[boundary_idx])
                else:
                    boundary_vals.append(np.nan)
            ax.plot(epochs, boundary_vals, marker='o', label=f"Boundary {boundary_idx+1}", markersize=2 if max_nb>10 else 4)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    ax.legend(
        ncol=2,
        fontsize=8,           # reduce the text size
        markerscale=0.5,      # make the legend markers smaller
        labelspacing=0.2,     # reduce vertical space between labels
        handlelength=1,       # shorten the line length for legend markers
        handletextpad=0.4     # reduce space between marker and text
    )

    if log_scale:
        ax.set_yscale('log')
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.savefig(output_filename)
    plt.close(fig)

from matplotlib.patches import Ellipse

def plot_learned_gaussians(data, model, dim_x, dim_y, output_filename, conf_level=2.30, order=None):
    """
    Plot the learned Gaussian components of the model (projected to two specified dimensions)
    together with the data (scatter plot).
    
    Parameters:
      data: dict mapping process name -> pandas DataFrame with column "NN_output"
            where each entry is an array-like of shape (dim,).
      model: trained DiffCatModelMultiDimensional (or similar) with method get_effective_parameters()
      dim_x, dim_y: integers specifying which dimensions to plot (e.g., 0 and 1)
      output_filename: file name to save the plot.
      conf_level: chi-square threshold for the ellipse contour; for 2D, 2.30 ~ 68% (1 sigma)
      
    The function will:
      - Retrieve the learned parameters (mixture weights, means, scale_tril).
      - Compute the covariance for each Gaussian as cov = L L^T.
      - Project each Gaussian to the (dim_x, dim_y) plane.
      - Compute the 1σ ellipse (using the eigen-decomposition of the projected covariance).
      - Scatter-plot the data (using all events, projected to (dim_x, dim_y)).
      - Overlay the ellipse contours.
    """
    # Retrieve effective parameters from the model.
    eff_params = model.get_effective_parameters()
    means = np.array(eff_params["means"])    # shape: (n_cats, dim)
    scale_tril = np.array(eff_params["scale_tril"])  # shape: (n_cats, dim, dim)
    # Number of components.
    n_cats = means.shape[0]

    if order is None:
        order = [_i for _i in range(n_cats)]
    
    # Compute covariances for each component.
    covariances = np.array([np.dot(L, L.T) for L in scale_tril])
    
    # Prepare figure.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter-plot the data.
    # We plot all processes, using a different marker/color per process.
    colors = {"signal": "tab:red", "bkg1": "tab:blue", "bkg2": "tab:orange", "bkg3": "tab:cyan"}
    markers = {"signal": "o", "bkg1": "s", "bkg2": "v", "bkg3": "d"}
    stop = 1000
    for proc, df in data.items():
        # Assume each entry in df["NN_output"] is an array-like of length >= max(dim_x, dim_y)+1.
        arr = np.stack(df["NN_output"].values)  # shape: (n_events, dim)
        x_vals = arr[:, dim_x]
        y_vals = arr[:, dim_y]
        ax.scatter(x_vals[:stop], y_vals[:stop], s=10, alpha=0.5, label=proc, color=colors.get(proc, "gray"), marker=markers.get(proc, "o"))

    # Plot the 1σ ellipses for each learned Gaussian.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["solid", "dashed", "dotted"]
    n_colors = len(colors)
    colors *= 10

    for i in range(n_cats):
        # Project mean and covariance to the desired dimensions.
        mu = tf.nn.softmax(means[i]).numpy()[[dim_x, dim_y]]  # shape: (2,)

        cov_proj = covariances[i][np.ix_([dim_x, dim_y], [dim_x, dim_y])]
        # Compute eigenvalues and eigenvectors.
        eigenvals, eigenvecs = np.linalg.eigh(cov_proj)
        # The angle of the ellipse is the angle of the largest eigenvector.
        angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        # Width and height are given by 2*sqrt(conf_level*eigenvalue).
        width = 2 * np.sqrt(conf_level * eigenvals[1])
        height = 2 * np.sqrt(conf_level * eigenvals[0])
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, linestyle=linestyles[i//n_colors],
                          edgecolor=colors[i], fc='none', lw=3, label=f'Gaussian {order[i]}')
        ax.add_patch(ellipse)
    
    ax.set_xlabel(f"Dimension {dim_x}", fontsize=18)
    ax.set_ylabel(f"Dimension {dim_y}", fontsize=18)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
