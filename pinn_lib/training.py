# Manages the model training process for different problem types.
# Training module for Physics-Informed Neural Networks (PINNs).
# Supports both penalty method and Lagrange multiplier method for boundary conditions.
# Note: Lagrange multiplier method is only supported for 1D steady-state problems.

import torch
import numpy as np
import time

class Trainer:
    def __init__(self, model, config, data):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        self.problem_type = getattr(config, 'PROBLEM_TYPE', 'time_dependent')
        self.bc_method = getattr(config, 'BC_METHOD', 'penalty')  # 'penalty' or 'lagrange'
        
        # Unpack data based on problem type
        if self.problem_type == 'time_dependent':
            self.X_u_train, self.u_train, self.X_f_train = [d.to(config.DEVICE) for d in data]
        else:  # steady-state problems
            self.X_bc_train, self.u_bc_train, self.X_f_train = [d.to(config.DEVICE) for d in data]
        
        self.epochs = config.ADAM_EPOCHS
        self.lbfgs_epochs = getattr(config, 'LBFGS_EPOCHS', 1000)
        
        # Setup optimizers
        self._setup_optimizers()
        self.iter = 0
        
        # Store boundary condition data on model for visualization (only for steady-state problems)
        if self.problem_type != 'time_dependent':
            self.model.X_bc_train = self.X_bc_train
            self.model.u_bc_train = self.u_bc_train
        
        # Print boundary condition method
        print(f"Using {self.bc_method.upper()} method for boundary condition enforcement")

    def _setup_optimizers(self):
        """Setup optimizers based on boundary condition method."""
        if self.bc_method == 'lagrange':
            # For Lagrange multipliers, we need to optimize both network params and multipliers
            params = list(self.model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=self.config.LEARNING_RATE)
            self.optimizer_lbfgs = torch.optim.LBFGS(
                params, lr=1.0, max_iter=50000, max_eval=50000,
                history_size=50, tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
            )
        else:
            # Standard penalty method
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0, max_iter=50000, max_eval=50000,
                history_size=50, tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
            )

    def loss_func(self):
        """Compute loss function based on problem type and BC method."""
        self.optimizer.zero_grad()
        
        if self.bc_method == 'penalty':
            return self._penalty_loss()
        else:
            return self._lagrange_loss()

    def _penalty_loss(self):
        """Traditional penalty method loss function."""
        if self.problem_type == 'time_dependent':
            # Original time-dependent loss
            u_pred = self.model(self.X_u_train)
            f_pred = self.model.get_pde_residual(self.X_f_train)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        else:
            # Steady-state loss
            u_pred = self.model(self.X_bc_train)
            f_pred = self.model.get_pde_residual(self.X_f_train)
            loss_u = torch.mean((self.u_bc_train - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        
        # Combine losses with penalty weighting
        bc_weight = getattr(self.config, 'BC_WEIGHT', 1.0)
        pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
        
        loss = bc_weight * loss_u + pde_weight * loss_f
        loss.backward()
        
        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, PDE Loss: {loss_f.item():.4e}")
        self.iter += 1
        return loss

    def _lagrange_loss(self):
        """Lagrange multiplier method loss function. Only supports 1D steady-state problems."""
        if self.problem_type != 'steady_state_1d':
            raise ValueError(f"Lagrange multiplier method only supported for 1D steady-state problems, got: {self.problem_type}")
        
        # 1D steady-state problems with Lagrange multipliers
        u_pred = self.model.forward_with_bc_enforcement(self.X_bc_train, self.X_bc_train, self.u_bc_train)
        f_pred = self.model.get_pde_residual(self.X_f_train, self.X_bc_train, self.u_bc_train)
        
        # For 1D problems, BCs should be well-enforced by Lagrange multipliers
        bc_residual = u_pred - self.u_bc_train
        loss_u = torch.mean(bc_residual ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        # Add regularization term for Lagrange multipliers
        lambda_reg = getattr(self.config, 'LAMBDA_REGULARIZATION', 1e-4)
        if hasattr(self.model, 'lagrange_multipliers'):
            loss_lambda = lambda_reg * torch.mean(self.model.lagrange_multipliers ** 2)
        else:
            loss_lambda = 0
        
        # For Lagrange method, focus primarily on PDE residual since BCs are enforced implicitly
        pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
        bc_weight = getattr(self.config, 'BC_WEIGHT', 0.1)  # Reduced weight since BCs are implicit
        
        loss = bc_weight * loss_u + pde_weight * loss_f + loss_lambda
        loss.backward()
        
        if self.iter % 100 == 0:
            if hasattr(self.model, 'lagrange_multipliers'):
                lambda_norm = torch.norm(self.model.lagrange_multipliers).item()
                print(f"Iter: {self.iter}, Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, "
                      f"PDE Loss: {loss_f.item():.4e}, λ Norm: {lambda_norm:.4e}")
            else:
                print(f"Iter: {self.iter}, Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, PDE Loss: {loss_f.item():.4e}")
        self.iter += 1
        return loss

    def train(self):
        """Train the model using Adam followed by L-BFGS optimization."""
        print("Starting Adam optimization...")
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            if self.bc_method == 'penalty':
                # Traditional penalty method training
                if self.problem_type == 'time_dependent':
                    u_pred = self.model(self.X_u_train)
                    f_pred = self.model.get_pde_residual(self.X_f_train)
                    loss_u = torch.mean((self.u_train - u_pred) ** 2)
                else:
                    u_pred = self.model(self.X_bc_train)
                    f_pred = self.model.get_pde_residual(self.X_f_train)
                    loss_u = torch.mean((self.u_bc_train - u_pred) ** 2)
                
                loss_f = torch.mean(f_pred ** 2)
                
                # Apply weights
                bc_weight = getattr(self.config, 'BC_WEIGHT', 1.0)
                pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
                loss = bc_weight * loss_u + pde_weight * loss_f
                
            else:
                # Lagrange multiplier method training
                if self.problem_type == 'time_dependent':
                    u_pred = self.model.forward_with_bc_enforcement(self.X_u_train, self.X_u_train, self.u_train)
                    f_pred = self.model.get_pde_residual(self.X_f_train, self.X_u_train, self.u_train)
                    loss_u = torch.mean((self.u_train - u_pred) ** 2)
                else:
                    u_pred = self.model.forward_with_bc_enforcement(self.X_bc_train, self.X_bc_train, self.u_bc_train)
                    f_pred = self.model.get_pde_residual(self.X_f_train, self.X_bc_train, self.u_bc_train)
                    
                    # For homogeneous BCs with Lagrange method, BC loss should be exactly zero
                    if torch.allclose(self.u_bc_train, torch.zeros_like(self.u_bc_train)):
                        loss_u = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        loss_u = torch.mean((self.u_bc_train - u_pred) ** 2)
                
                loss_f = torch.mean(f_pred ** 2)
                
                # Add regularization for Lagrange multipliers
                lambda_reg = getattr(self.config, 'LAMBDA_REGULARIZATION', 1e-4)
                if hasattr(self.model, 'lagrange_multipliers'):
                    loss_lambda = lambda_reg * torch.mean(self.model.lagrange_multipliers ** 2)
                else:
                    loss_lambda = 0
                
                # Apply weights (reduced BC weight for Lagrange method)
                bc_weight = getattr(self.config, 'BC_WEIGHT', 0.1)
                pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
                loss = bc_weight * loss_u + pde_weight * loss_f + loss_lambda
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                if self.bc_method == 'lagrange' and hasattr(self.model, 'lagrange_multipliers'):
                    lambda_norm = torch.norm(self.model.lagrange_multipliers).item()
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, "
                          f"PDE Loss: {loss_f.item():.4e}, λ Norm: {lambda_norm:.4e}")
                else:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, PDE Loss: {loss_f.item():.4e}")
        
        print(f"Adam optimization finished in {time.time() - start_time:.2f} seconds.")

        print("\nStarting L-BFGS optimization...")
        start_time_lbfgs = time.time()
        self.optimizer_lbfgs.step(self.loss_func)
        print(f"L-BFGS optimization finished in {time.time() - start_time_lbfgs:.2f} seconds.")
