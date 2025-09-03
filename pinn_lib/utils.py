# Utility functions for plotting, saving, etc.

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_solution(model, config):
    """Plot solution based on problem type."""
    model.eval()
    problem_type = getattr(config, 'PROBLEM_TYPE', 'time_dependent')
    
    if problem_type == 'time_dependent':
        plot_time_dependent_solution(model, config)
    elif problem_type == 'steady_state_1d':
        plot_steady_1d_solution(model, config)
    elif problem_type == 'steady_state_2d':
        if getattr(config, 'FIELD_TYPE', 'scalar') == 'scalar':
            plot_steady_2d_scalar_solution(model, config)
        else:
            plot_steady_2d_vector_solution(model, config)

def _get_model_prediction(model, x_test, config):
    """
    Get model prediction.
    """
    # Standard forward pass
    return model(x_test.to(config.DEVICE))

def plot_time_dependent_solution(model, config):
    """Plot time-dependent solution (original)."""
    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], 256)
    t = torch.linspace(config.T_RANGE[0], config.T_RANGE[1], 101)
    X, T = torch.meshgrid(x, t, indexing='ij')
    X_star = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    with torch.no_grad():
        u_pred = _get_model_prediction(model, X_star, config).cpu()
    
    U_pred = u_pred.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T.numpy(), X.numpy(), U_pred.numpy(), cmap='rainbow', shading='auto')
    plt.colorbar(label="u(x,t)")
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")
    plt.title(f"Predicted Solution u(x,t) of {config.__name__.split('.')[-1].replace('_', ' ').title()}")
    plt.tight_layout()
    plt.show()

def plot_steady_1d_solution(model, config):
    """Plot 1D steady-state solution - Heat conduction in rod."""
    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], 1000).unsqueeze(1)
    
    with torch.no_grad():
        T_pred = _get_model_prediction(model, x, config).cpu()
    
    plt.figure(figsize=(12, 6))
    
    # Plot PINN solution
    plt.plot(x.numpy(), T_pred.numpy(), 'b-', linewidth=3, label='PINN Solution', alpha=0.8)
    
    # Plot analytical solution if available
    if hasattr(config, 'ANALYTICAL_SOLUTION'):
        T_exact = config.ANALYTICAL_SOLUTION(x)
        plt.plot(x.numpy(), T_exact.numpy(), 'r--', linewidth=2, label='Analytical Solution')
        
        # Calculate and display error
        error = torch.mean(torch.abs(T_pred - T_exact)).item()
        plt.text(0.05, 0.95, f'Mean Absolute Error: {error:.2e}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.xlabel('Position along rod (x/L)', fontsize=12)
    plt.ylabel('Temperature (T - T₀) [°C]', fontsize=12)
    plt.title('Heat Conduction in Rod with Internal Heat Generation\n' + 
              r'$\frac{d^2T}{dx^2} = -Q\sin(\pi x)$, Fixed ends at T=0°C', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add physical interpretation
    plt.text(0.5, 0.05, 'Physical interpretation: Rod with sinusoidal heat generation\n' +
                        '(e.g., electrical heating, solar absorption)', 
             transform=plt.gca().transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def plot_steady_2d_scalar_solution(model, config):
    """Plot 2D steady-state scalar field - Heat transfer in plate."""
    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], 100)
    y = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_star = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        T_pred = _get_model_prediction(model, X_star, config).cpu()
    
    T_pred = T_pred.reshape(X.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Contour plot
    contour = axes[0].contourf(X.numpy(), Y.numpy(), T_pred.numpy(), levels=20, cmap='hot')
    axes[0].set_xlabel('x/L (Plate Width)', fontsize=12)
    axes[0].set_ylabel('y/L (Plate Height)', fontsize=12)
    axes[0].set_title('Temperature Distribution in 2D Plate\n' + 
                      r'$\nabla^2 T = -Q\sin(\pi x)\sin(\pi y)$', fontsize=14)
    axes[0].set_aspect('equal')
    cbar1 = plt.colorbar(contour, ax=axes[0])
    cbar1.set_label('Temperature (T - T₀) [°C]', fontsize=12)
    
    # Add heat source indication
    axes[0].text(0.5, 0.5, 'Heat\nGeneration\nZone', ha='center', va='center', 
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    axes[1] = fig.add_subplot(122, projection='3d')
    surf = axes[1].plot_surface(X.numpy(), Y.numpy(), T_pred.numpy(), cmap='hot', alpha=0.9)
    axes[1].set_xlabel('x/L', fontsize=12)
    axes[1].set_ylabel('y/L', fontsize=12)
    axes[1].set_zlabel('Temperature [°C]', fontsize=12)
    axes[1].set_title('3D Temperature Field', fontsize=14)
    cbar2 = plt.colorbar(surf, ax=axes[1], shrink=0.5)
    cbar2.set_label('Temperature [°C]', fontsize=10)
    
    # Add application note
    fig.suptitle('Heat Transfer in Electronic Component/Heat Spreader', fontsize=16, y=1.02)
    plt.figtext(0.5, 0.02, 'Applications: CPU heat spreaders, electronic cooling, solar collectors', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()

def plot_steady_2d_vector_solution(model, config):
    """Plot 2D steady-state vector field - Flow around cylinder."""
    # Create grid for plotting
    nx, ny = 40, 40
    x = torch.linspace(config.X_RANGE[0], config.X_RANGE[1], nx)
    y = torch.linspace(config.Y_RANGE[0], config.Y_RANGE[1], ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_star = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Filter out points inside cylinder if present
    if hasattr(config, 'CYLINDER_RADIUS'):
        r = config.CYLINDER_RADIUS
        center = getattr(config, 'CYLINDER_CENTER', [0.0, 0.0])
        distances = torch.sqrt((X_star[:, 0] - center[0])**2 + (X_star[:, 1] - center[1])**2)
        mask = distances > r
        X_plot = X_star[mask]
    else:
        X_plot = X_star
        mask = torch.ones(X_star.shape[0], dtype=torch.bool)
    
    # Compute velocity field
    if hasattr(model, 'X_bc_train') and hasattr(model, 'u_bc_train'):
        # Use gradient-enabled computation
        X_plot_grad = X_plot.to(config.DEVICE).requires_grad_(True)
        
        # Use the basic network forward pass
        phi = model.forward(X_plot_grad)
        
        # Compute velocity components
        phi_grad = torch.autograd.grad(phi.sum(), X_plot_grad, create_graph=False)[0]
        u_vel = phi_grad[:, 0:1]  # ∂φ/∂x
        v_vel = phi_grad[:, 1:2]  # ∂φ/∂y
        velocity = torch.cat([u_vel, v_vel], dim=1).cpu().detach()
    else:
        # Fallback for other methods
        with torch.no_grad():
            velocity = model.get_velocity_field(X_plot.to(config.DEVICE)).cpu()
    
    # Create velocity grids
    U_vel = torch.zeros(X.shape)
    V_vel = torch.zeros(Y.shape)
    
    vel_idx = 0
    for i in range(nx):
        for j in range(ny):
            flat_idx = i * ny + j
            if mask[flat_idx]:
                U_vel[i, j] = velocity[vel_idx, 0]
                V_vel[i, j] = velocity[vel_idx, 1]
                vel_idx += 1
            else:
                U_vel[i, j] = float('nan')
                V_vel[i, j] = float('nan')
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Vector field plot
    # Show domain coverage
    valid_mask = ~torch.isnan(U_vel)
    speed = torch.sqrt(U_vel**2 + V_vel**2)
    
    # Simple scatter plot showing domain and speed
    X_plot = X[valid_mask].numpy()
    Y_plot = Y[valid_mask].numpy()
    speed_plot = speed[valid_mask].numpy()
    
    # Create scatter plot with color-coded speed
    scatter = axes[0].scatter(X_plot, Y_plot, c=speed_plot, cmap='viridis', s=2, alpha=0.8)
    plt.colorbar(scatter, ax=axes[0], label='Velocity Magnitude [m/s]')
    
    # Add cylinder boundary if present
    if hasattr(config, 'CYLINDER_RADIUS'):
        center = getattr(config, 'CYLINDER_CENTER', [0.0, 0.0])
        circle = patches.Circle((center[0], center[1]), config.CYLINDER_RADIUS, fill=True, color='darkgray', 
                               edgecolor='black', linewidth=2, alpha=0.9)
        axes[0].add_patch(circle)
        axes[0].text(center[0], center[1], 'Cylinder\n(No-slip)', ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
    
    axes[0].set_xlabel('x/D (Cylinder Diameters)', fontsize=12)
    axes[0].set_ylabel('y/D (Cylinder Diameters)', fontsize=12)
    axes[0].set_title('Velocity Field Around Circular Cylinder\n' + 
                      r'Potential Flow: $\nabla^2 \phi = 0$, $\vec{u} = \nabla \phi$', fontsize=14)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Streamlines plot
    # Show a summary of the flow
    axes[1].text(0.5, 0.5, 'Penalty Method\n\n' +
                           f'BC Enforcement via Loss Function\n' +
                           f'Cylinder at center with\n' +
                           f'radius = {config.CYLINDER_RADIUS:.1f}\n' +
                           f'Far-field flow conditions\n' +
                           f'enforced via penalty',
                 transform=axes[1].transAxes, ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    axes[1].set_xlim(config.X_RANGE)
    axes[1].set_ylim(config.Y_RANGE)
    
    axes[1].set_xlabel('x/D (Cylinder Diameters)', fontsize=12)
    axes[1].set_ylabel('y/D (Cylinder Diameters)', fontsize=12)
    axes[1].set_title('Penalty Method Results\n(Boundary Condition Enforcement via Loss)', fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Add application note
    fig.suptitle('Aerodynamic Flow Analysis: Cylinder in Cross-Flow', fontsize=16, y=0.98)
    plt.figtext(0.5, 0.02, 'Applications: Wind loading on structures, flow meters, heat exchanger tubes, vortex shedding analysis', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
