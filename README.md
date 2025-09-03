# Physics-Informed Neural Networks (PINNs) Framework

This repository implements a modular framework for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs). The framework focuses on mechanical engineering applications including heat transfer, fluid dynamics, and structural analysis.

## Overview

PINNs are neural networks trained to satisfy both the governing equations of a physical system and its boundary conditions. This approach allows solving PDEs without traditional numerical methods by leveraging the universal approximation capability of neural networks.

## Features

- Modular architecture for different types of PDEs
- Support for both time-dependent and steady-state problems
- 1D and 2D spatial domains
- Scalar and vector field solutions
- Multiple boundary condition enforcement techniques
- Visualization tools for each problem type

## Problem Types

The framework includes the following pre-configured problems:

1. **Burgers Equation (`burgers`)**:  
   - Time-dependent viscous Burgers equation  
   - Fundamental fluid dynamics application

2. **1D Steady State (`1d_steady`)**:  
   - Heat conduction in a steel rod with internal heat generation  
   - Fixed temperature at both ends (Dirichlet boundary conditions)

3. **2D Scalar Field (`2d_scalar`)**:  
   - 2D heat transfer in an aluminum plate  
   - Applications include CPU heat spreaders and electronic cooling

4. **2D Vector Field (`2d_vector`)**:  
   - Aerodynamic flow around a circular cylinder  
   - Potential flow model

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- SciPy
- pyDOE
- Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Commands

1. **List all available problems**:

   ```bash
   python3 main.py --list
   ```

2. **Run a specific problem**:

   ```bash
   python3 main.py --problem PROBLEM_NAME
   ```
   
   Where `PROBLEM_NAME` is one of: `burgers`, `1d_steady`, `2d_scalar`, `2d_vector`

3. **Run all problems sequentially**:

   ```bash
   python3 main.py --problem all
   ```

4. **Skip visualization plots**:

   ```bash
   python3 main.py --problem PROBLEM_NAME --no-plots
   ```

### Examples

1. Run the Burgers equation problem:

   ```bash
   python3 main.py --problem burgers
   ```

2. Run the 1D heat conduction problem:

   ```bash
   python3 main.py --problem 1d_steady
   ```

## Code Structure

- **`main.py`**: Entry point for running simulations
- **`configs/`**: Configuration files for each problem type
  - `burgers_equation.py`: Time-dependent Burgers equation
  - `steady_state_1d.py`: 1D heat conduction
  - `steady_state_2d_scalar.py`: 2D heat transfer
  - `steady_state_2d_vector.py`: Flow around cylinder
- **`pinn_lib/`**: Core library implementing the PINN framework
  - `data.py`: Data generation for training
  - `models.py`: Neural network architecture and PDE residuals
  - `training.py`: Training loop with optimization methods
  - `utils.py`: Visualization and utility functions

## Technical Details

### Boundary Condition Enforcement

The framework implements three approaches for enforcing boundary conditions:

1. **Penalty Method**: Adds boundary conditions as weighted terms in the loss function (default)
2. **Lagrangian Method**: Uses Lagrange multipliers to enforce constraints
3. **Hybrid Approach**: Blends penalty and Lagrangian methods using an alpha parameter

### Training Process

The training process consists of two phases:

1. **Adam Optimization**: Used for initial training due to its robustness
2. **L-BFGS Optimization**: Applied for fine-tuning due to its faster convergence

### PDE Residuals

PDE residuals are computed using automatic differentiation:

- **Time-dependent PDEs**: Computes temporal and spatial derivatives
- **Steady-state PDEs**: Computes Laplacian and other spatial derivatives
- **Vector field problems**: Uses potential flow formulation

## Configuration Parameters

Each problem has its own configuration file with parameters such as:

- **Domain parameters**: Spatial and temporal ranges
- **PDE parameters**: Coefficients, source terms
- **Network architecture**: Layer sizes
- **Training parameters**: Epochs, learning rate
- **Boundary condition parameters**: Types, values
- **Physical parameters**: Material properties, etc.

## Customization

To create a new problem:

1. Create a new configuration file in the `configs/` directory
2. Define all necessary parameters (see existing files for examples)
3. Extend the `PROBLEM_CONFIGS` dictionary in `main.py` if needed

## Computational Performance

Training time depends on:

- Problem complexity
- Neural network size
- Number of training points
- Hardware (CPU/GPU)

For reference, on a modern CPU:

- 1D problems typically train in seconds to minutes
- 2D problems may take several minutes

GPU acceleration is automatically used if available.

## References

This implementation is based on the PINN methodology introduced in:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
