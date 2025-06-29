Installation
============

Getting Started
---------------

GATO is available on PyPI and can be installed via pip. We recommend using a virtual environment:

.. code-block:: bash

   # Create (optional) and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Install the latest stable release from PyPI
   pip install gato

Development Version
-------------------

To try the very latest changes or contribute to the codebase, you can for example use the editable installation:

   .. code-block:: bash

      git clone https://github.com/FloMau/gato.git
      cd gato
      pip install -e .

Post-install Check
------------------

After installation, verify that GATO is available and check the version:

.. code-block:: bash

   python -c "import gato; print(f'GATO version {gato.__version__} installed.')"
