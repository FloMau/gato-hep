gato-hep
========

gato-hep is a gradient-based cATegorization Optimizer for High Energy Physics analyses. It learns boundaries in N-dimensional discriminants that maximize signal significance for binned likelihood fits, using differentiable significance models and TensorFlow-based optimization.

Quick links
-----------
- Documentation: https://gato-hep.readthedocs.io/
- PyPI: https://pypi.org/project/gato-hep/
- Examples: see the ``examples/`` directory in the repository

Highlights
----------
- Optimize categorizations in multi-dimensional spaces using Gaussian Mixture Models (GMM) or 1D sigmoid-based models.
- Set per-dimension ranges to match your discriminants.
- Penalize low-yield or high-uncertainty categories to keep optimizations analysis-friendly.
- Built-in annealing schedules for temperature / steepness and learning rate to stabilize training and approach hard class assignments.
- Ready-to-run toy workflows that mirror real HEP analysis patterns.

Need a rapid tour? Start with :doc:`quickstart`. For setup details see :doc:`installation`, and jump to the :doc:`api/index` reference when integrating gato-hep into your analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   installation
   examples
   api/index

Indices and tables
==================
* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
