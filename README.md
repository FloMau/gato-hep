gato-hep: Gradient-based Categorization Optimizer for High Energy Physics
==============================================

[![Documentation Status](https://readthedocs.org/projects/gato-hep/badge/?version=latest)](https://gato-hep.readthedocs.io/en/latest/)


A toolkit for binning / categorisation optimisation with respect to signal 
significance for HEP analyses, using gradient-descent methods.
gatohep relies on TensorFlow with TensorFlow-Probability.

The categorisation can be performed directly in a multidimensional discriminant 
space, e.g. from a mutliclassifier with softmax activation.
The bins are defined by learnable multidimensional Gaussians as a Gaussian Mixture Model (GMM), or, well working in 1D, using bin boundaries approximated by steep sigmoid functions of learnable position.

See the full documentation at <https://gato-hep.readthedocs.io/>.

--------------------------------------------------------------------
Quick install (editable mode)
--------------------------------------------------------------------
```bash
git clone https://github.com/FloMau/gato-hep.git
cd gato-hep
python3 -m venv gato_env       # or use conda
source gato_env/bin/activate
pip install -e .
```

Dependencies are declared in *pyproject.toml*. 
Note: The only tricky part is to find matching versions of tensorflow, tensorflow-probability and ml-dtypes. The requirements mentioned here should work, however, other combinations may work as well.

--------------------------------------------------------------------
## Running the toy examples
--------------------------------------------------------------------
### 1D toy (signal vs. multi-background)
```python
python examples/1D_example/run_toy_example.py
```

### 3-class soft-max (2 D slice of 3 D)
```python
python examples/three_class_softmax_example/run_example.py
```

Each script writes plots & a significance comparison table.

--------------------------------------------------------------------
## Apply gato-hep to your own data
--------------------------------------------------------------------
``` python
# standard GMM model for ND optimisation
from gatohep.models import gato_gmm_model
# more to be included here later on

# see ./examples for a full workflow!
```

--------------------------------------------------------------------
## Directory layout
--------------------------------------------------------------------
```
gato-hep/                       project root
│
├─ pyproject.toml           metadata + dependencies
├─ src/gatohep/                installable Python package
│   │
│   ├─ __init__.py
│   ├─ models.py            Trainable model class
│   └─ losses.py            custom loss / penalty terms
│   ├─ utils.py             misc helpers
│   ├─ plotting_utils.py    helper plots (stacked hists, bin boundaries, ...)
│   ├─ data_generation.py   toy data generators (1D / 3-class softmax)
│
└─ examples/                runnable demos
    ├─ 1D_example/run_example.py
    └─ three_class_softmax_example/run_example.py
```

--------------------------------------------------------------------
## Contributing
--------------------------------------------------------------------
1. `git checkout -b feature/xyz`
2. Code under *src/gatohep/*, add tests under *tests/*.
3. Update version in *pyproject.toml*.
4. `black` / `isort` / `pytest`, then open a PR.

