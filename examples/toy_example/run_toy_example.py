import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_toy_data import generate_toy_data
import sys, os
# Append the parent directory (i.e. the repo root) to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from diffcat_optimizer.plotting_utils import plot_stacked_histograms


# Import the hist package (assumed installed, e.g., via pip install hist)
import hist

def create_hist(data, weights=None, bins=50, low=0.0, high=1.0, name="NN_output"):
    """
    Creates a hist.hist object from a 1D numpy array 'data'
    with specified binning in the range [low, high].
    """
    h = hist.Hist.new.Reg(bins, low, high, name=name).Weight()
    # Fill the histogram with data. Here, all weights are 1.
    if weights is not None:
        h.fill(data, weight=weights)
    else:
        h.fill(data)
    return h

def main():
    # Generate toy data.
    data = generate_toy_data(
        n_signal=10000,
        n_bkg1=10000, n_bkg2=10000, n_bkg3=10000,
        xs_signal=0.5,    # 500 fb = 0.5 pb
        xs_bkg1=100, xs_bkg2=50, xs_bkg3=15,
        lumi=100,         # in /fb
        seed=42
    )

    # Create histograms using the hist package.
    # Assuming NN output values are in [0,1].
    bins = 25
    low = 0.0
    high = 1.0

    h_signal = create_hist(data["signal"]["NN_output"], weights=data["signal"]["weight"], bins=bins, low=low, high=high, name="Signal")
    h_bkg1 = create_hist(data["bkg1"]["NN_output"], weights=data["bkg1"]["weight"], bins=bins, low=low, high=high, name="Bkg1")
    h_bkg2 = create_hist(data["bkg2"]["NN_output"], weights=data["bkg2"]["weight"], bins=bins, low=low, high=high, name="Bkg2")
    h_bkg3 = create_hist(data["bkg3"]["NN_output"], weights=data["bkg3"]["weight"], bins=bins, low=low, high=high, name="Bkg3")

    # Prepare lists for stacked backgrounds and signal.
    stacked_hists = [h_bkg1, h_bkg2, h_bkg3]
    process_labels = ["Background 1", "Background 2", "Background 3"]
    signal_hists = [h_signal*100]
    signal_labels = ["Signal x 100"]

    # Call the plotting function.
    plot_stacked_histograms(
        stacked_hists=stacked_hists,
        process_labels=process_labels,
        signal_hists=signal_hists,
        signal_labels=signal_labels,
        output_filename="examples/toy_example/NN_output_distribution.pdf",
        axis_labels=("Toy NN output", "Toy events"),
        # bins=bins,
        normalize=False,
        log=False
    )
    print("Plot saved as examples/toy_example/NN_output_distribution.pdf")

if __name__ == "__main__":
    main()

