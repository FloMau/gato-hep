# GATO: Gradient-based cATegorization Optimizer

This repository contains a framework for optimizing analysis categorisation boundaries in a differentiable way using TensorFlow.
It is designed to be both general and configurable, and can be adapted for specific HEP analyses.
## Project Structure

- `diffcat_optimizer/`: Contains the core Python modules.
- `examples/`: Examples how to use the package
- `requirements.txt`: Lists the required Python packages.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/FloMau/gato.git

## Run examples
### 1D signal-vs.background toy example
This example represents a generic HEP analysis with a signal process and multiple backgrounds with a one-dimensional discriminant.
We obtain the discriminant based on random numbers following exponential PDFs, which represents a discriminant that could be obtained from multivariate classifiers.
It will compare the significance obtained from equidistant binning with the GATO-optimized binning for different numbers of bins.

A minimum of `N` events can be required in the GATO binning. Here, `N=10` background events are used as default.
This is implemented with a differentiable penalty added to the loss.

Run it via: 
```
python examples/toy_example/run_toy_example.py
'''

### Further examples
To be included soon
