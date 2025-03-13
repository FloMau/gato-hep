# GATO: Gradient-based cATegoryzation Optimizer

This repository contains a framework for optimizing analysis categories in a differentiable way using TensorFlow. It is designed to be both general and configurable, and can be adapted for specific HEP analyses such as the ttH vs. tH optimization.

## Project Structure

- `diffcat_optimizer/`: Contains the core Python modules.
- `examples/`: Example scripts for toy data and a specialized ttH-tH analysis.
- `requirements.txt`: Lists the required Python packages.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/FloMau/gato.git

## Run examples
### Toy example
Idea: generate a toy NN output distribution for three background processes and a signal process. 
So far, it only generates the data and plots them.
Optimisation of binning for signal sensitivity will be included soon.

Run it via: 
```bash
python examples/toy_example/run_toy_example.py


### Further examples
To be included soon
