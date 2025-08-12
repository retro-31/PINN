# Utility functions for plotting, saving, etc.

import torch
import matplotlib.pyplot as plt

def plot_solution(model, config):
    model.eval()
    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], 256)
    t = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], 101)
    X, T = torch.meshgrid(x, t, indexing='ij')
    X_star = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    with torch.no_grad():
        u_pred = model(X_star.to(config.DEVICE)).cpu()
    
    U_pred = u_pred.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T.numpy(), X.numpy(), U_pred.numpy(), cmap='rainbow', shading='auto')
    plt.colorbar(label="u(x,t)")
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")
    plt.title(f"Predicted Solution u(x,t) of {config.__name__.split('.')[-1].replace('_', ' ').title()}")
    plt.tight_layout()
    plt.show()
