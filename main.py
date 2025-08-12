# main.py
import torch
import numpy as np
import importlib
import argparse

# Import from the renamed 'pinn_lib' package
from pinn_lib.data import DataGenerator
from pinn_lib.models import PINN
from pinn_lib.training import Trainer
from pinn_lib.utils import plot_solution

def main(config_name):
    """Main execution function."""
    # Dynamically import the specified configuration module
    try:
        config = importlib.import_module(f"configs.{config_name}")
    except ImportError:
        print(f"Error: Configuration file 'configs/{config_name}.py' not found.")
        return

    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"device: {config.DEVICE}")
    print(f"configuration: {config_name}")

    # --- Step 1: Generate Data ---
    print("Step 1: Generating data...")
    data_gen = DataGenerator(config)
    training_data = data_gen.generate_data()
    print("Data generation complete.")

    # --- Step 2: Create Model ---
    print("\nStep 2: Creating PINN model...")
    pinn_model = PINN(config.LAYERS, config.NU)
    print("Model created.")
    print(f"Model Architecture: {config.LAYERS}")

    # --- Step 3: Train Model ---
    print("\nStep 3: Training model...")
    trainer = Trainer(pinn_model, config, training_data)
    trainer.train()
    print("Training complete.")

    # --- Step 4: Visualize Results ---
    print("\nStep 4: Visualizing the solution...")
    plot_solution(pinn_model, config)
    print("Visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN model with a specific configuration.")
    parser.add_argument('--config', type=str, default='burgers_equation',
                        help='Name of the configuration file to use (without .py extension).')
    args = parser.parse_args()
    
    main(args.config)
