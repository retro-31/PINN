# Configuration for 1D steady-state problems using Lagrange Multipliers
# Mechanical Engineering: Heat Conduction in a Rod with Heat Generation
# This config demonstrates the Lagrange multiplier method for boundary condition enforcement

import numpy as np
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Boundary Condition Method ---
BC_METHOD = "lagrange"  # Use Lagrange multipliers instead of penalty method
NUM_LAGRANGE_MULTIPLIERS = 5  # Number of Lagrange multiplier parameters

# --- Problem Type ---
PROBLEM_TYPE = "steady_state_1d"
FIELD_TYPE = "scalar"  # scalar or vector
OUTPUT_DIM = 1  # 1 for scalar, 2 for 2D vector, 3 for 3D vector

# --- PDE Configuration ---
# Heat conduction equation with internal heat generation:
# -k * d²T/dx² = q_gen
# Where: k = thermal conductivity, q_gen = heat generation rate
# 
# Dimensionless form: d²T/dx² = -Q*sin(πx)
# Physical interpretation: Rod with sinusoidal heat generation (e.g., electrical heating)
# Analytical solution: T(x) = (Q/π²)sin(πx) + C₁x + C₂
# With BCs T(0)=T₀=0, T(L)=T₀=0: T(x) = (Q/π²)sin(πx)

# Heat generation rate (dimensionless)
Q = 1.0  # Heat generation intensity
SOURCE_FUNCTION = lambda x: -Q * torch.sin(np.pi * x)

# --- Domain Configuration ---
X_RANGE = [0.0, 1.0]  # Rod length (dimensionless: 0 to L)

# --- Boundary Conditions ---
# For Lagrange multiplier method, these are used to define the boundary constraints
BC_LEFT = 0.0    # T(0) = 0°C (left end)
BC_RIGHT = 0.0   # T(1) = 0°C (right end)

# --- Training Data Configuration ---
N_BC = 2        # Number of boundary points (fixed for 1D: left and right ends)
N_F = 2000      # Number of collocation points for PDE residual

# --- Network Architecture ---
LAYERS = [1, 30, 30, 30, 1]  # Slightly larger network for Lagrange method

# --- Training Configuration ---
ADAM_EPOCHS = 8000     # Increased epochs for Lagrange multiplier convergence
LBFGS_EPOCHS = 2000    # Additional L-BFGS epochs
LEARNING_RATE = 1e-3   # Learning rate for Adam optimizer

# --- Loss Function Weights ---
BC_WEIGHT = 0.1         # Reduced weight since BCs are enforced implicitly
PDE_WEIGHT = 1.0        # Standard PDE weight
LAMBDA_REGULARIZATION = 1e-4  # Regularization for Lagrange multipliers

# --- Reproducibility ---
SEED = 42

# --- Physical Parameters (for interpretation) ---
# Steel rod properties (for engineering context)
THERMAL_CONDUCTIVITY = 50.0  # W/m·K (carbon steel)
ROD_LENGTH = 0.1            # m (10 cm rod)
HEAT_GENERATION_RATE = 1e6  # W/m³ (1 MW/m³)

# --- Analytical Solution (for validation) ---
def ANALYTICAL_SOLUTION(x):
    """
    Analytical solution for validation.
    T(x) = (Q/π²)sin(πx)
    """
    return (Q / (np.pi**2)) * torch.sin(np.pi * x)

# --- Engineering Context ---
DESCRIPTION = """
Lagrange Multiplier Method Demo: 1D Heat Conduction

This configuration demonstrates the Lagrange multiplier approach for enforcing
boundary conditions implicitly rather than using penalty terms in the loss function.

Physical Problem:
- Steel rod with internal heat generation (e.g., electrical heating)
- Fixed temperatures at both ends (heat sinks)
- Sinusoidal heat generation pattern

Lagrange Method Benefits:
- Exact satisfaction of boundary conditions
- Reduced dependence on penalty weights
- More stable convergence for some problems
- Better handling of complex boundary geometries

Mathematical Formulation:
- PDE: d²T/dx² = -Q*sin(πx)
- BCs: T(0) = T(1) = 0 (enforced via Lagrange multipliers)
- Network output automatically satisfies BCs through basis functions
"""

print("Configuration: 1D Steady-State Heat Conduction (Lagrange Multiplier Method)")
print(f"Boundary Condition Method: {BC_METHOD.upper()}")
print(f"Number of Lagrange Multipliers: {NUM_LAGRANGE_MULTIPLIERS}")
print(f"Network Architecture: {LAYERS}")
print(f"Training: {ADAM_EPOCHS} Adam + {LBFGS_EPOCHS} L-BFGS epochs")
