# Configuration for 2D steady-state scalar field problems
# Mechanical Engineering: Heat Transfer in a 2D Plate with Thermal Loads

import numpy as np
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Problem Type ---
PROBLEM_TYPE = "steady_state_2d"
FIELD_TYPE = "scalar"
OUTPUT_DIM = 1

# --- PDE Configuration ---
# 2D Heat conduction equation with distributed heat source:
# ∇²T = -q_gen/k
# 
# Dimensionless form: ∇²T = -Q*sin(πx)sin(πy)
# Physical interpretation: 
# - Square plate with distributed heat generation (e.g., electronic components)
# - Could represent thermal loading in a CPU heat spreader
# - Or solar heating on a flat plate collector
# 
# Analytical solution: T(x,y) = (Q/2π²)sin(πx)sin(πy)

# Heat generation intensity (dimensionless)
Q = 2.0
SOURCE_FUNCTION = lambda x, y: -Q * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# --- Domain Configuration ---
X_RANGE = [0.0, 1.0]  # Plate width (dimensionless)
Y_RANGE = [0.0, 1.0]  # Plate height (dimensionless)

# --- Boundary Conditions ---
# All edges maintained at constant temperature (heat sink boundaries)
# Represents plate edges in contact with cooling system
BC_TYPE = "dirichlet"
BC_FUNCTION = lambda x, y: torch.zeros_like(x)  # T = 0°C on all boundaries

# --- Physical Parameters (for reference) ---
# Thermal conductivity: k = 200 W/m·K (aluminum plate)
# Plate dimensions: 0.1m × 0.1m × 0.01m (thickness)
# Heat generation: q_gen = 10,000 W/m³ (electronic heat dissipation)
# Boundary temperature: T_boundary = 20°C (ambient cooling)

# --- Analytical Solution (for comparison) ---
ANALYTICAL_SOLUTION = lambda x, y: (Q/(2*np.pi**2)) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# --- Data Configuration ---
N_BC = 400  # Number of boundary points (100 per side)
N_F = 10000  # Number of collocation points

# --- Model Configuration ---
LAYERS = [2, 50, 50, 50, 50, 1]

# --- Training Configuration ---
ADAM_EPOCHS = 10000
LBFGS_EPOCHS = 2000
LEARNING_RATE = 1e-3

# --- Random Seed ---
SEED = 1234
