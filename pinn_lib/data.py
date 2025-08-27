# Module for generating training data.

import torch
from pyDOE import lhs
import numpy as np

class DataGenerator:
    """
    Generates training data for the PINN based on configuration.
    """
    def __init__(self, config):
        self.config = config
        self.problem_type = getattr(config, 'PROBLEM_TYPE', 'time_dependent')
        
        if self.problem_type == 'time_dependent':
            # Original Burgers equation setup
            self.x_min, self.x_max = config.X_RANGE
            self.t_min, self.t_max = config.T_RANGE
            self.N_u = config.N_U
            self.N_f = config.N_F
        elif 'steady_state' in self.problem_type:
            self.x_min, self.x_max = config.X_RANGE
            if hasattr(config, 'Y_RANGE'):
                self.y_min, self.y_max = config.Y_RANGE
            self.N_bc = getattr(config, 'N_BC', 100)
            self.N_f = config.N_F

    def generate_data(self):
        """
        Generates and returns the training data tensors based on problem type.
        """
        if self.problem_type == 'time_dependent':
            return self._generate_time_dependent_data()
        elif self.problem_type == 'steady_state_1d':
            return self._generate_steady_state_1d_data()
        elif self.problem_type == 'steady_state_2d':
            return self._generate_steady_state_2d_data()
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

    def _generate_time_dependent_data(self):
        """Original Burgers' equation data generation."""
        # Initial Condition Data
        t_initial = torch.zeros((self.N_u // 2, 1)).float()
        x_initial = self.x_min + (self.x_max - self.x_min) * torch.rand(self.N_u // 2, 1)
        X_initial = torch.cat((x_initial, t_initial), dim=1)
        u_initial = -torch.sin(np.pi * X_initial[:, 0:1])

        # Boundary Condition Data
        t_boundary = self.t_min + (self.t_max - self.t_min) * torch.rand(self.N_u // 2, 1)
        x_boundary_left = self.x_min * torch.ones_like(t_boundary)
        x_boundary_right = self.x_max * torch.ones_like(t_boundary)
        u_boundary = torch.zeros_like(t_boundary)

        X_boundary_left = torch.cat((x_boundary_left, t_boundary), dim=1)
        X_boundary_right = torch.cat((x_boundary_right, t_boundary), dim=1)

        X_u_train = torch.cat([X_initial, X_boundary_left, X_boundary_right], dim=0)
        u_train = torch.cat([u_initial, u_boundary, u_boundary], dim=0)

        # Collocation Points for PDE Loss
        X_f_train_lhs = self.t_min + (self.t_max - self.t_min) * lhs(2, self.N_f)
        X_f_train = torch.from_numpy(X_f_train_lhs).float()
        X_f_train = X_f_train[:, [1, 0]]
        X_f_train[:, 0] = self.x_min + (self.x_max - self.x_min) * X_f_train[:, 0]

        return X_u_train, u_train, X_f_train

    def _generate_steady_state_1d_data(self):
        """Generate data for 1D steady-state problems."""
        # Boundary points
        X_bc = torch.tensor([[self.x_min], [self.x_max]], dtype=torch.float32)
        
        # Boundary values
        if hasattr(self.config, 'BC_VALUES'):
            u_bc = torch.tensor([[self.config.BC_VALUES['left']], 
                                [self.config.BC_VALUES['right']]], dtype=torch.float32)
        else:
            u_bc = torch.zeros_like(X_bc)

        # Collocation points
        X_f = self.x_min + (self.x_max - self.x_min) * lhs(1, self.N_f)
        X_f = torch.from_numpy(X_f).float()

        return X_bc, u_bc, X_f

    def _generate_steady_state_2d_data(self):
        """Generate data for 2D steady-state problems."""
        # Boundary points
        N_bc_per_side = self.N_bc // 4
        
        # Bottom boundary (y = y_min)
        x_bottom = torch.linspace(self.x_min, self.x_max, N_bc_per_side).unsqueeze(1)
        y_bottom = self.y_min * torch.ones_like(x_bottom)
        
        # Top boundary (y = y_max)
        x_top = torch.linspace(self.x_min, self.x_max, N_bc_per_side).unsqueeze(1)
        y_top = self.y_max * torch.ones_like(x_top)
        
        # Left boundary (x = x_min)
        y_left = torch.linspace(self.y_min, self.y_max, N_bc_per_side).unsqueeze(1)
        x_left = self.x_min * torch.ones_like(y_left)
        
        # Right boundary (x = x_max)
        y_right = torch.linspace(self.y_min, self.y_max, N_bc_per_side).unsqueeze(1)
        x_right = self.x_max * torch.ones_like(y_right)
        
        # Combine all boundary points
        X_bc = torch.cat([
            torch.cat([x_bottom, y_bottom], dim=1),
            torch.cat([x_top, y_top], dim=1),
            torch.cat([x_left, y_left], dim=1),
            torch.cat([x_right, y_right], dim=1)
        ], dim=0)
        
        # Boundary values
        if hasattr(self.config, 'BC_FUNCTION'):
            u_bc = self.config.BC_FUNCTION(X_bc[:, 0:1], X_bc[:, 1:2])
        else:
            # For vector fields with potential flow, output is scalar potential
            if hasattr(self.config, 'FIELD_TYPE') and self.config.FIELD_TYPE == 'vector':
                u_bc = torch.zeros(X_bc.shape[0], 1)  # Potential φ boundary conditions
            else:
                u_bc = torch.zeros(X_bc.shape[0], 1)

        # Handle vector field case
        if self.config.FIELD_TYPE == 'vector':
            # For vector fields, we might need special boundary handling
            if hasattr(self.config, 'CYLINDER_RADIUS'):
                # Add cylinder boundary points for flow problems
                X_bc, u_bc = self._add_cylinder_boundary(X_bc, u_bc)

        # Collocation points
        X_f_temp = lhs(2, self.N_f)
        X_f = torch.from_numpy(X_f_temp).float()
        X_f[:, 0] = self.x_min + (self.x_max - self.x_min) * X_f[:, 0]
        X_f[:, 1] = self.y_min + (self.y_max - self.y_min) * X_f[:, 1]

        # Remove collocation points inside cylinder if present
        if hasattr(self.config, 'CYLINDER_RADIUS'):
            X_f = self._remove_interior_points(X_f)

        return X_bc, u_bc, X_f

    def _add_cylinder_boundary(self, X_bc, u_bc):
        """Add boundary points on cylinder surface for flow problems."""
        if hasattr(self.config, 'CYLINDER_RADIUS'):
            r = self.config.CYLINDER_RADIUS
            center = getattr(self.config, 'CYLINDER_CENTER', [0.0, 0.0])
            N_cyl = 200  # More points for better cylinder resolution
            theta = torch.linspace(0, 2*np.pi, N_cyl).unsqueeze(1)
            x_cyl = center[0] + r * torch.cos(theta)
            y_cyl = center[1] + r * torch.sin(theta)
            X_cyl = torch.cat([x_cyl, y_cyl], dim=1)
            
            # No-slip boundary condition on cylinder (φ = constant on cylinder)
            # For potential flow around cylinder, the potential is constant on the surface
            u_cyl = torch.zeros(N_cyl, 1)  # Potential boundary condition
            
            X_bc = torch.cat([X_bc, X_cyl], dim=0)
            u_bc = torch.cat([u_bc, u_cyl], dim=0)
        
        return X_bc, u_bc

    def _remove_interior_points(self, X_f):
        """Remove collocation points inside cylinder."""
        if hasattr(self.config, 'CYLINDER_RADIUS'):
            r = self.config.CYLINDER_RADIUS
            center = getattr(self.config, 'CYLINDER_CENTER', [0.0, 0.0])
            # Calculate distances from cylinder center
            distances = torch.sqrt((X_f[:, 0] - center[0])**2 + (X_f[:, 1] - center[1])**2)
            mask = distances > r * 1.05  # Small buffer to avoid boundary issues
            X_f = X_f[mask]
        return X_f
