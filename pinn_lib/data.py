# Module for generating training data.

import torch
from pyDOE import lhs
import numpy as np

class DataGenerator:
    """
    Generates training data for the PINN based on configuration.
    """
    def __init__(self, config):
        self.x_min, self.x_max = config.X_RANGE
        self.t_min, self.t_max = config.T_RANGE
        self.N_u = config.N_U
        self.N_f = config.N_F

    def generate_data(self):
        """
        Generates and returns the training data tensors for Burgers' equation.
        """
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
