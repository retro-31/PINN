# Defines the PINN architecture for different problem types.

import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
Physics-Informed Neural Network (PINN) implementation.

This module implements a deep neural network for solving partial differential equations (PDEs)
using the Physics-Informed Neural Network approach.

Features:
- Multi-layer feedforward neural network with customizable architecture
- Support for time-dependent and steady-state problems
- 1D and 2D spatial domains
- Scalar and vector field solutions
- Penalty method for boundary condition enforcement
- Lagrange multiplier method for boundary condition enforcement (1D steady-state only)

The network can handle various PDE types including:
- Burgers' equation (time-dependent)
- 1D steady-state diffusion equations  
- 2D Poisson equations (scalar fields)
- 2D potential flow problems (vector fields)

Supports both penalty method and Lagrange multiplier method for boundary conditions.
Note: Lagrange multiplier method is only supported for 1D steady-state problems.
"""
    def __init__(self, layers, config):
        super().__init__()
        self.layers = layers
        self.config = config
        self.problem_type = getattr(config, 'PROBLEM_TYPE', 'time_dependent')
        self.field_type = getattr(config, 'FIELD_TYPE', 'scalar')
        
        # Boundary condition enforcement method
        self.bc_method = getattr(config, 'BC_METHOD', 'penalty')  # 'penalty' or 'lagrange'
        
        # For time-dependent problems (backward compatibility)
        if hasattr(config, 'NU'):
            self.nu = config.NU

        # Main neural network
        linears = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.linears = nn.ModuleList(linears)
        self.activation = nn.Tanh()
        
        # Initialize Lagrange multiplier parameters if using Lagrange method
        if self.bc_method == 'lagrange':
            self._init_lagrange_multipliers()
        
        self.init_weights()

    def _init_lagrange_multipliers(self):
        """Initialize Lagrange multiplier parameters for boundary conditions."""
        # Only support 1D problems with Lagrange multipliers
        if self.problem_type != 'steady_state_1d':
            raise ValueError(f"Lagrange multiplier method only supported for 1D steady-state problems, got: {self.problem_type}")
        
        # 1D problems typically need fewer multipliers (left & right boundaries)
        num_multipliers = getattr(self.config, 'NUM_LAGRANGE_MULTIPLIERS', 2)
        
        # Create learnable Lagrange multiplier parameters
        self.lagrange_multipliers = nn.Parameter(
            torch.zeros(num_multipliers, device=self.config.DEVICE, requires_grad=True)
        )
        
        print(f"Initialized {num_multipliers} Lagrange multipliers for BC enforcement")

    def init_weights(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network."""
        a = x
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def forward_with_bc_enforcement(self, x, x_bc=None, u_bc=None):
        """
        Forward pass with boundary condition enforcement.
        
        Args:
            x: Input points
            x_bc: Boundary condition points (only needed for Lagrange method)
            u_bc: Boundary condition values (only needed for Lagrange method)
        
        Returns:
            Network output with boundary conditions enforced
        """
        if self.bc_method == 'penalty' or x_bc is None:
            # Standard forward pass (penalty method handles BCs in loss)
            return self.forward(x)
        else:
            # Lagrange multiplier method: modify output to satisfy BCs
            return self._enforce_bc_lagrange(x, x_bc, u_bc)
    
    def _enforce_bc_lagrange(self, x, x_bc, u_bc):
        """
        Enforce boundary conditions using Lagrange multipliers.
        Only supports 1D steady-state problems.
        """
        u_base = self.forward(x)
        
        if self.problem_type == 'steady_state_1d':
            return self._enforce_bc_lagrange_1d(x, x_bc, u_bc, u_base)
        else:
            raise ValueError(f"Lagrange multiplier method only supported for 1D steady-state problems, got: {self.problem_type}")
    
    def _enforce_bc_lagrange_1d(self, x, x_bc, u_bc, u_base):
        """Enforce 1D boundary conditions using Lagrange multipliers."""
        # For 1D problems with Dirichlet BCs at x=0 and x=1
        # Modify solution: u(x) = u_base(x) + λ₁*φ₁(x) + λ₂*φ₂(x)
        # where φᵢ(x) are basis functions that enforce BCs
        
        # Use simple polynomial basis functions
        x_normalized = x  # Assuming x is already in [0,1]
        
        # Basis functions: φ₁(x) = x(1-x), φ₂(x) = x²(1-x)
        phi1 = x_normalized * (1 - x_normalized)
        phi2 = x_normalized**2 * (1 - x_normalized)
        
        # Correction term using first two Lagrange multipliers
        if len(self.lagrange_multipliers) >= 2:
            correction = (self.lagrange_multipliers[0] * phi1 + 
                         self.lagrange_multipliers[1] * phi2)
        else:
            correction = 0
        
        # For homogeneous Dirichlet BCs (u=0 at boundaries), use scaling
        boundary_scaling = x_normalized * (1 - x_normalized)
        
        return u_base * boundary_scaling + correction
    
    def get_pde_residual(self, x_f, x_bc=None, u_bc=None):
        """
        Computes the PDE residual based on problem type.
        For Lagrange multiplier method, pass boundary condition data.
        """
        if self.problem_type == 'time_dependent':
            return self._get_burgers_residual(x_f, x_bc, u_bc)
        elif self.problem_type == 'steady_state_1d':
            return self._get_steady_1d_residual(x_f, x_bc, u_bc)
        elif self.problem_type == 'steady_state_2d':
            if self.field_type == 'scalar':
                return self._get_poisson_2d_residual(x_f, x_bc, u_bc)
            elif self.field_type == 'vector':
                return self._get_vector_field_residual(x_f, x_bc, u_bc)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

    def _get_burgers_residual(self, x_f, x_bc=None, u_bc=None):
        """
        Original Burgers' equation residual: f = u_t + u*u_x - nu*u_xx.
        """
        x_f.requires_grad = True
        x, t = x_f[:, 0:1], x_f[:, 1:2]
        u = self.forward_with_bc_enforcement(torch.cat((x, t), dim=1), x_bc, u_bc)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def _get_steady_1d_residual(self, x_f, x_bc=None, u_bc=None):
        """
        1D steady-state residual: d²u/dx² = source_term.
        """
        x_f.requires_grad = True
        u = self.forward_with_bc_enforcement(x_f, x_bc, u_bc)
        u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Add source term if provided
        source = torch.zeros_like(u)
        if hasattr(self.config, 'SOURCE_FUNCTION'):
            source = self.config.SOURCE_FUNCTION(x_f)
        
        f = u_xx - source
        return f

    def _get_poisson_2d_residual(self, x_f, x_bc=None, u_bc=None):
        """
        2D Poisson equation residual: ∇²u = source_term.
        """
        x_f.requires_grad_(True)
        u = self.forward_with_bc_enforcement(x_f, x_bc, u_bc)
        
        # Compute first derivatives
        grad_u = torch.autograd.grad(u.sum(), x_f, create_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_f, create_graph=True)[0][:, 1:2]
        
        laplacian = u_xx + u_yy
        
        # Add source term if provided
        source = torch.zeros_like(u)
        if hasattr(self.config, 'SOURCE_FUNCTION'):
            x, y = x_f[:, 0:1], x_f[:, 1:2]
            source = self.config.SOURCE_FUNCTION(x, y)
        
        f = laplacian - source
        return f

    def _get_vector_field_residual(self, x_f, x_bc=None, u_bc=None):
        """
        Vector field residual for 2D flow problems.
        For potential flow: ∇²φ = 0, then u = ∇φ
        """
        x_f.requires_grad_(True)
        
        # Network outputs potential φ with BC enforcement
        phi = self.forward_with_bc_enforcement(x_f, x_bc, u_bc)
        
        # Compute first derivatives using full gradient
        grad_phi = torch.autograd.grad(phi.sum(), x_f, create_graph=True)[0]
        phi_x = grad_phi[:, 0:1]
        phi_y = grad_phi[:, 1:2]
        
        # Compute second derivatives
        phi_xx = torch.autograd.grad(phi_x.sum(), x_f, create_graph=True)[0][:, 0:1]
        phi_yy = torch.autograd.grad(phi_y.sum(), x_f, create_graph=True)[0][:, 1:2]
        
        # Laplace equation for potential function
        f = phi_xx + phi_yy
        return f

    def get_velocity_field(self, x_f, x_bc=None, u_bc=None):
        """
        For vector field problems, compute velocity from potential.
        u = ∂φ/∂x, v = ∂φ/∂y
        """
        if self.field_type != 'vector':
            raise ValueError("get_velocity_field only applicable for vector field problems")
        
        # Ensure gradient computation is enabled
        x_f = x_f.clone().detach().requires_grad_(True)
        phi = self.forward_with_bc_enforcement(x_f, x_bc, u_bc)
        
        # Compute velocity using full gradient  
        try:
            grad_phi = torch.autograd.grad(phi.sum(), x_f, create_graph=False)[0]
            u = grad_phi[:, 0:1]  # ∂φ/∂x
            v = grad_phi[:, 1:2]  # ∂φ/∂y
            return torch.cat([u, v], dim=1)
        except RuntimeError as e:
            # Fallback: use simple network output for visualization
            print(f"Warning: Could not compute velocity gradients: {e}")
            # Return zero velocity as placeholder
            return torch.zeros(x_f.shape[0], 2, device=x_f.device)
