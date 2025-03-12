# analysis_tth_th/run_tth_th.py
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from model_tth_th import TTHTHCategoryModel
from some_data_loading_module import load_ttH_tH_data  # your specialized data loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lam", type=float, default=0.0)
    args = parser.parse_args()

    # 1) Load your specialized data
    data_dict = load_ttH_tH_data()

    # 2) Define the config for variables
    #    1) first variable => ttH vs. tH region
    #    2) second variable => sub-cats in ttH region
    #    3) third variable => sub-cats in tH region
    variables_config = [
      {"name": "ttH_vs_tH_NN",   "n_cats": 2},  # or 3 if you have 2 boundaries
      {"name": "sig_vs_bkg_NN_ttH_had", "n_cats": 2},
      {"name": "sig_vs_bkg_NN_tH_had",  "n_cats": 2},
    ]

    model = TTHTHCategoryModel(
        variables_config=variables_config,
        mass_min=100.0, 
        mass_max=180.0,
        lam=args.lam
    )

    # 3) Define an optimizer (like Adam)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    # 4) Training loop
    for epoch in range(args.epochs):
        with tf.GradientTape() as tape:
            loss_val = model.call(data_dict)  # The call() returns a scalar
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}, loss={loss_val.numpy():.3f}")

    # 5) Print final boundaries
    final_boundaries = model.get_effective_boundaries()
    print("Final effective boundaries:", final_boundaries)

    # 6) (Optional) Recompute significance or yields, etc., in a final pass
    final_loss = model.call(data_dict)
    print("Final significance ~", -final_loss.numpy())  # if penalty is small


if __name__ == "__main__":
    main()
