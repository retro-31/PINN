# Modular Physics-Informed Neural Networks (PINNs)

This repository provides a modular and extensible PyTorch implementation of Physics-Informed Neural Networks (PINNs). The code is structured with subdirectories to allow for easy addition of new PDEs, models, and experiments.

The initial example solves the 1D viscous Burgers' equation.

## Repository Structure

The repository is organized into a core `pinn_lib` package and a `configs` directory:

- `main.py`: The main script to execute the training and prediction workflow.
- `requirements.txt`: Lists the necessary Python packages.
- `README.md`: This file.
- **`configs/`**: A directory for experiment-specific configuration files.
- `configs/burgers_equation.py`: Configuration for the Burgers' equation problem.
- **`pinn_lib/`**: The main Python package containing the core logic.
- `__init__.py`: Makes `pinn_lib` a package.
- `data.py`: Handles data generation.
- `models.py`: Defines the PINN architecture.
- `training.py`: Manages the model training loop.
- `utils.py`: Contains utility functions, like plotting.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/retro-31/PINN.git
    cd PINN
    ```

2. **Create and activate the conda environment (recommended):**

    ```bash
    conda create -n pinn python=3.10
    conda activate pinn
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the PINN model, execute the main script from the project root directory:

```bash
# This will use the default 'burgers_equation' config
python main.py

# To specify a different config
# python main.py --config your_config_name
```
