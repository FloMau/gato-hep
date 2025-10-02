Three-class softmax example
===========================

Optimize categories directly on a 3-class softmax output (visualized in two dimensions). The script trains several Gaussian mixtures, compares to argmax baselines, and produces animations of the learned components.

Run:

.. code-block:: console

    python examples/three_class_softmax_example/run_example.py --gato-bins 5 10 20 --epochs 500

Notable artifacts
-----------------
- ``frames_<N>`` folders with boundary evolution frames (assembled into GIFs).
- Stacked histograms contrasting background compositions with scaled signal templates.
- Yield vs. uncertainty bar charts for each bin.

The :doc:`../quickstart` page provides a trimmed version of this workflow that fits a single configuration and demonstrates saving/restoring models.

Source code
-----------

.. literalinclude:: ../../../examples/three_class_softmax_example/run_example.py
    :language: python
    :linenos:
    :caption: Source of the 3-class softmax example
