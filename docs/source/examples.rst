Examples
========

The repository has runnable demonstrations that mirror typical HEP optimization tasks.
Each script generates toy datasets, trains a gato-hep model, and writes diagnostic plots/tables under the corresponding ``examples/.../Plots*/`` directory.

Specifically:
----------
- **1D sigmoid/GMM** - compare different approaches to sculpting categories in a single discriminant and inspect penalty terms, bias, and yield-vs-uncertainty plots.
- **Three-class softmax** - operate directly on a three-class score. Can be similarly used on multiple 1D discriminants by stacking their scores.

.. toctree::
   :maxdepth: 1

   examples/1D_gmm_example
   examples/1D_sigmoid_example
   examples/three_class_softmax_example
