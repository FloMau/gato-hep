Bump hunt example
==========================

This workflow closely follows the three-class softmax example, but here we do the inference on an unrelated event variable ("mass"), mirroring a bump hunt workflow as e.g. in Higgs-to-γγ analyses.
Hence, events in a category are not all in one bin (as it was in the other workflows), but rather spread out over the full mass range and only partly contribute to the significance obtained in the small signal window.

* Generate two resonant signals and five continuum backgrounds (building on the three-class softmax example).
* Assign a diphoton mass to every event (Gaussian for signal, exponentials for the continuum).
* Technically, we could perform the categorization optimization purely on the events in a small signal window around 125 GeV, but practically, often we suffer from low background statistics.
* Therefore, we use the full power of the continuum background simulation by including all events in the gradient calculations, but reweighting the yield to match the expectation in the signal window (125 ± σ) during training. For this, we fit the background with exponentials in each category.

Run it with for instance:

.. code-block:: bash

   python examples/bumphunt_example/run_example.py \
       --epochs 400 \
       --gato-bins 5 8 \
       --out PlotsBumpHunt

Outputs land under ``examples/bumphunt_example/<out>/`` and contain:

* Inclusive diphoton-mass spectra before categorisation (linear/log).
* Per-category diphoton spectra for all signals/backgrounds.
* Loss, penalty, and bias histories with temperature annotations.
* Boundary snapshots + GIFs showing the 2-D category evolution.
* Yield vs. statistical-uncertainty bar plots per category.
* Saved checkpoints for each trained configuration.

Source
------

.. literalinclude:: ../../../examples/bumphunt_example/run_example.py
   :language: python
   :caption: Diphoton bump-hunt optimisation script
   :linenos:
