Installation
============

PyPI release
------------

The latest tagged release is available on PyPI and can be installed with pip. We recommend using an isolated environment (``venv``, ``micromamba`` or ``conda``):

.. code-block:: bash

   python -m venv .venv        # optional but encouraged
   source .venv/bin/activate
   pip install gato-hep

The base install targets CPU execution and pulls the matching TensorFlow/TensorFlow-Probability versions automatically. Optional extras:

.. code-block:: bash

   pip install "gato-hep[gpu]"   # CUDA-enabled TensorFlow wheels
   pip install "gato-hep[dev]"   # linting + testing helpers for development

When using the ``gpu`` extra make sure the host already provides compatible NVIDIA drivers and CUDA libraries.

Editable install from source
----------------------------

To track main or contribute patches, install the repository in editable mode:

.. code-block:: bash

   git clone https://github.com/FloMau/gato-hep.git
   cd gato-hep
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

You can append ``[gpu]`` to the extras list if you need the CUDA stack in the same environment (``pip install -e ".[dev,gpu]"``).

Post-install check
------------------

Confirm that the package imports correctly and report the version:

.. code-block:: bash

   python -c "import gatohep; print(f'gato-hep version {gatohep.__version__} installed.')"

Environment compatibility
-------------------------

- Python ``>=3.10``.
- TensorFlow ``2.17`` -- ``2.19`` and TensorFlow-Probability ``>=0.24`` (installed automatically through the dependency metadata).
- ``ml_dtypes >= 0.4.1``.

If pip cannot find compatible TensorFlow wheels (e.g. on Apple Silicon), install the platform-specific ``tensorflow-macos`` / ``tensorflow-metal`` packages first and then re-run ``pip install gato-hep``. See the official TensorFlow release notes for detailed platform guidance.
