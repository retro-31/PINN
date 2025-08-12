# Configuration for the 1D Burgers' Equation problem.

import numpy as np
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PDE Configuration ---
# Equation: 1D Burgers' Equation
# u_t + u * u_x - nu * u_xx = 0
NU = 0.01 / np.pi

# --- Domain Configuration ---
X_RANGE = [-1.0, 1.0]
T_RANGE = [0.0, 1.0]

# --- Data Configuration ---
# Number of training points
N_U = 100  # Number of initial/boundary data points
N_F = 10000 # Number of collocation points for PDE residual

# --- Model Configuration ---
# Neural network architecture: [input_dim, hidden_1, ..., hidden_n, output_dim]
LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# --- Training Configuration ---
ADAM_EPOCHS = 10000
LEARNING_RATE = 1e-3

# --- Random Seed for Reproducibility ---
SEED = 1234
