1D GMM example
==============

Optimize bin boundaries for a single discriminant using a Gaussian mixture model. The script builds the toy dataset, trains multiple category counts, and compares GATO-derived significances to equidistant baselines.

Run:

.. code-block:: console

    python examples/1D_example/run_gmm_example.py --gato-bins 5 10 20 --epochs 300

Key outputs
-----------
- Stacked histograms for both equidistant and optimized binning schemes.
- Loss, boundary and penalty histories saved under ``examples/1D_example/Plots*/``.
- ``checkpoints/<N>_bins`` directories storing model weights for later inspection.

Source code
-----------

.. literalinclude:: ../../../examples/1D_example/run_gmm_example.py
    :language: python
    :linenos:
    :caption: Source of the 1D GMM toy example
