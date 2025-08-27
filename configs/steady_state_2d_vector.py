# Configuration for 2D steady-state vector field problems
# Mechanical Engineering: Flow Around Circular Cylinder (Aerodynamics)

import numpy as np
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Problem Type ---
PROBLEM_TYPE = "steady_state_2d"
FIELD_TYPE = "vector"
OUTPUT_DIM = 2  # [u_x, u_y] velocity components

# --- PDE Configuration ---
# Potential flow around circular cylinder:
# ∇²φ = 0  (Laplace equation for velocity potential)
# u = ∇φ   (velocity field from potential gradient)
# 
# Physical interpretation:
# - Inviscid flow around circular obstacle (cylinder/pipe)
# - Foundation for aerodynamic analysis of bluff bodies
# - Relevant for wind loading on structures, flow meters, heat exchangers
# - Simplification of Navier-Stokes for high Reynolds number flow

# --- Domain Configuration ---
X_RANGE = [-2.0, 2.0]  # Flow domain width
Y_RANGE = [-2.0, 2.0]  # Flow domain height

# --- Boundary Conditions ---
# No-slip condition on cylinder surface: u = 0
# Uniform flow at far-field boundaries: u_∞ = (U_∞, 0)
BC_TYPE = "mixed"  # Dirichlet on cylinder + far-field conditions

# Cylinder geometry
CYLINDER_CENTER = [0.0, 0.0]  # Cylinder center
CYLINDER_RADIUS = 0.5         # Cylinder radius (D = 1.0)

# Flow conditions
FREE_STREAM_VELOCITY = 1.0    # U_∞ = 1.0 m/s (dimensionless)
REYNOLDS_NUMBER = 1e6         # High Re for potential flow approximation

# --- Physical Parameters (for reference) ---
# Fluid: Air at standard conditions
# Density: ρ = 1.225 kg/m³
# Viscosity: μ = 1.81e-5 Pa·s
# Cylinder diameter: D = 1.0 m
# Free stream velocity: U_∞ = 1.0 m/s
# Applications: Wind loading, flow meters, heat exchanger tubes

# --- Data Configuration ---
N_BC = 1000  # Boundary points (cylinder + far-field)
N_F = 15000  # Collocation points (excludes cylinder interior)

# --- Model Configuration ---
# Network outputs velocity potential φ, then compute u = ∇φ
LAYERS = [2, 60, 60, 60, 60, 60, 1]  # Output is potential φ

# --- Training Configuration ---
ADAM_EPOCHS = 15000
LBFGS_EPOCHS = 3000
LEARNING_RATE = 1e-3

# --- Random Seed ---
SEED = 1234
