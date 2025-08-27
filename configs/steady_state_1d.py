# Configuration for 1D steady-state problems
# Mechanical Engineering: Heat Conduction in a Rod with Heat Generation

import numpy as np
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Fixed temperature at both ends (e.g., both ends in contact with heat sink)
BC_TYPE = "dirichlet"
BC_VALUES = {"left": 0.0, "right": 0.0}  # T(0) = T(L) = 0°C

# --- Analytical Solution (for comparison) ---
ANALYTICAL_SOLUTION = lambda x: (Q/np.pi**2) * torch.sin(np.pi * x)

# --- Physical Parameters (for reference) ---
# Thermal conductivity: k = 50 W/m·K (typical for steel)
# Rod length: L = 0.1 m
# Heat generation: q_gen = 1000 W/m³

# --- Data Configuration ---
N_BC = 2  # Number of boundary points (both ends)
N_F = 1000  # Number of collocation points for PDE residual

# --- Model Configuration ---
# Neural network architecture: [input_dim, hidden_layers..., output_dim]
LAYERS = [1, 20, 20, 20, 1]

# --- Training Configuration ---
ADAM_EPOCHS = 5000
LBFGS_EPOCHS = 1000
LEARNING_RATE = 1e-3

# --- Random Seed ---
SEED = 1234
