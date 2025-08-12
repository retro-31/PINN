# Defines the PINN architecture.

import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network model for Burgers' Equation.
    """
    def __init__(self, layers, nu):
        super().__init__()
        self.layers = layers
        self.nu = nu

        linears = [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.linears = nn.ModuleList(linears)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.linears:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_and_t):
        a = x_and_t
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def get_pde_residual(self, x_f):
        """
        Computes the residual of the Burgers' equation: f = u_t + u*u_x - nu*u_xx.
        """
        x_f.requires_grad = True
        x, t = x_f[:, 0:1], x_f[:, 1:2]
        u = self.forward(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f
