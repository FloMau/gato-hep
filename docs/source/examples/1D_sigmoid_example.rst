1D example based on sigmoids
============================

Use monotonic sigmoids to approximate bin boundaries in a single discriminant. This variant exposes steepness annealing alongside the usual yield/uncertainty penalties.

Run:

.. code-block:: console

    python examples/1D_example/run_sigmoid_example.py --gato-bins 5 10 20 --epochs 300

To regenerate plots and tables from trained checkpoints without re-running the optimization, use:

.. code-block:: console

    python examples/analyse_sigmoid_models.py --checkpoint-root examples/1D_example/PlotsSigmoidModel/checkpoints

Outputs mirror the GMM example: diagnostic PDFs, boundary histories, and saved models for each category count.

Source code
-----------

.. literalinclude:: ../../../examples/1D_example/run_sigmoid_example.py
    :language: python
    :linenos:
    :caption: Source of the 1D sigmoid toy example
