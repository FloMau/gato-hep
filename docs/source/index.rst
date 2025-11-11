gato-hep
========

We present `gato-hep`: the Gradient-based cATegorization Optimizer for High Energy Physics analyses.
`gato-hep` learns boundaries in N-dimensional discriminants that maximize signal significance for binned likelihood fits, using a differentiable approximation of signal significance and gradient descent techniques for optimization with TensorFlow.

- 📘 Documentation: https://gato-hep.readthedocs.io/
- 📦 PyPI: https://pypi.org/project/gato-hep/
- 🧪 Examples: see the `examples/` directory in this repository

Key Features
------------
- Optimize categorizations in multi-dimensional spaces using Gaussian Mixture Models (GMM) or 1D sigmoid-based models
- Set the range of the discriminant dimensions as needed for your analysis
- Penalize low-yield or high-uncertainty categories to keep optimizations analysis-friendly
- Built-in annealing schedules for temperature / steepness (setting the level of approximation for differentiability), and learning rate to stabilize training
- Ready-to-run toy workflows that mirror real HEP analysis patterns
Quick links
-----------
- GitHub: https://github.com/FloMau/gato-hep
- PyPI: https://pypi.org/project/gato-hep/
- Examples: see the ``examples/`` directory in the repository

For setup details see :doc:`installation`, and jump to the :doc:`api/index` reference when integrating gato-hep into your analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   api/index

Indices and tables
==================
* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
