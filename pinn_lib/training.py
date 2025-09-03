# Manages the model training process for different problem types.
# Training module for Physics-Informed Neural Networks (PINNs).

import torch
import numpy as np
import time

class Trainer:
    def __init__(self, model, config, data):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        self.problem_type = getattr(config, 'PROBLEM_TYPE', 'time_dependent')
        
        # Blending parameter between penalty and Lagrangian methods
        # 0 = pure penalty, 1 = pure Lagrangian, values between = blend
        self.lagrangian_alpha = getattr(config, 'LAGRANGIAN_ALPHA', 0.0)
        
        # Initialize Lagrangian multipliers if needed (no need to print message here as it's in main.py)
        
        # Unpack data based on problem type
        if self.problem_type == 'time_dependent':
            self.X_u_train, self.u_train, self.X_f_train = [d.to(config.DEVICE) for d in data]
        else:  # steady-state problems
            self.X_bc_train, self.u_bc_train, self.X_f_train = [d.to(config.DEVICE) for d in data]
            
            # Initialize Lagrangian multipliers for non-time-dependent problems if alpha > 0
            if self.lagrangian_alpha > 0 and self.problem_type != 'time_dependent':
                num_bc_points = self.X_bc_train.shape[0]
                self.model.init_lagrange_multipliers(num_bc_points, self.device)
        
        self.epochs = config.ADAM_EPOCHS
        self.lbfgs_epochs = getattr(config, 'LBFGS_EPOCHS', 1000)
        
        # Setup optimizers
        self._setup_optimizers()
        self.iter = 0
        
        # Store boundary condition data on model for visualization (only for steady-state problems)
        if self.problem_type != 'time_dependent':
            self.model.X_bc_train = self.X_bc_train
            self.model.u_bc_train = self.u_bc_train

    def _setup_optimizers(self):
        """Setup optimizers for training."""
        # Standard optimization setup
        if self.lagrangian_alpha > 0 and self.problem_type != 'time_dependent':
            # Create parameter groups for network parameters and Lagrange multipliers
            network_params = [p for p in self.model.parameters() if p is not self.model.lagrange_multipliers]
            
            # Use separate parameter groups with potentially different learning rates
            self.optimizer = torch.optim.Adam([
                {'params': network_params},
                {'params': [self.model.lagrange_multipliers], 'lr': self.config.LEARNING_RATE * 0.1}
            ], lr=self.config.LEARNING_RATE)
            
            # L-BFGS optimizer includes all parameters
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), lr=0.5, max_iter=50000, max_eval=50000,
                history_size=50, tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
            )
        else:
            # Standard setup for penalty method
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0, max_iter=50000, max_eval=50000,
                history_size=50, tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
            )

    def loss_func(self):
        """Compute loss function based on problem type and method."""
        self.optimizer.zero_grad()
        
        if self.problem_type == 'time_dependent' or self.lagrangian_alpha == 0:
            return self._penalty_loss()
        else:
            return self._hybrid_loss()
        
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
        
        # Apply weights to different loss components
        bc_weight = getattr(self.config, 'BC_WEIGHT', 1.0)
        pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
        
        loss = bc_weight * loss_u + pde_weight * loss_f
        loss.backward()
        
        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {loss.item():.4e}, BC Loss: {loss_u.item():.4e}, PDE Loss: {loss_f.item():.4e}")
        self.iter += 1
        return loss
        
    def _hybrid_loss(self):
        """Hybrid loss function blending penalty and Lagrangian methods."""
        # Compute PDE residual (same for both methods)
        f_pred = self.model.get_pde_residual(self.X_f_train)
        loss_f = torch.mean(f_pred ** 2)
        
        # Compute boundary predictions
        u_pred = self.model(self.X_bc_train)
        
        # Compute boundary violations
        bc_violations = self.u_bc_train - u_pred
        
        # Penalty term (weighted by 1-alpha)
        penalty_loss = torch.mean(bc_violations ** 2)
        
        # Lagrangian term (weighted by alpha)
        if hasattr(self.model, 'lagrange_multipliers') and self.model.lagrange_multipliers is not None:
            # Add a larger stabilization term to prevent numerical issues
            # This is similar to the augmented Lagrangian method
            epsilon = 1e-2
            lagrangian_term = torch.sum(self.model.lagrange_multipliers * bc_violations) + \
                              epsilon * torch.mean(bc_violations ** 2)
        else:
            lagrangian_term = 0.0
            
        # Get weights from config
        bc_weight = getattr(self.config, 'BC_WEIGHT', 1.0)
        pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
        
        # Combined loss with blending
        loss = pde_weight * loss_f + \
               bc_weight * ((1 - self.lagrangian_alpha) * penalty_loss + 
                           self.lagrangian_alpha * lagrangian_term)
                           
        loss.backward()
        
        # Compute mean BC error for logging (always use squared error for reporting)
        bc_error = torch.mean(bc_violations ** 2)
        
        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {loss.item():.4e}, BC Error: {bc_error.item():.4e}, "
                  f"PDE Loss: {loss_f.item():.4e}")
        self.iter += 1
        return loss

    def train(self):
        """Train the model using Adam followed by L-BFGS optimization."""
        print("Starting Adam optimization...")
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            # Choose loss function based on method
            if self.problem_type == 'time_dependent' or self.lagrangian_alpha == 0:
                # Standard penalty method for time-dependent problems
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
                
                bc_error = loss_u  # For reporting
                
            else:
                # Hybrid method for steady-state problems
                f_pred = self.model.get_pde_residual(self.X_f_train)
                loss_f = torch.mean(f_pred ** 2)
                
                u_pred = self.model(self.X_bc_train)
                bc_violations = self.u_bc_train - u_pred
                
                # Penalty component
                penalty_loss = torch.mean(bc_violations ** 2)
                
                # Lagrangian component
                lagrangian_term = torch.sum(self.model.lagrange_multipliers * bc_violations)
                
                # Get weights from config
                bc_weight = getattr(self.config, 'BC_WEIGHT', 1.0)
                pde_weight = getattr(self.config, 'PDE_WEIGHT', 1.0)
                
                # Combined loss with blending
                loss = pde_weight * loss_f + \
                       bc_weight * ((1 - self.lagrangian_alpha) * penalty_loss + 
                                   self.lagrangian_alpha * lagrangian_term)
                
                # For reporting
                bc_error = torch.mean(bc_violations ** 2)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update Lagrangian multipliers (if using Lagrangian method)
            if self.lagrangian_alpha > 0 and self.problem_type != 'time_dependent' and (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    # Recompute boundary violations after neural network update
                    u_pred = self.model(self.X_bc_train)
                    bc_violations = self.u_bc_train - u_pred
                    
                    # Update multipliers: λ = λ + step_size * constraint
                    # This is a gradient ascent step for the dual problem
                    step_size = getattr(self.config, 'LAGRANGIAN_STEP_SIZE', 0.01)
                    self.model.lagrange_multipliers.data += step_size * bc_violations
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4e}, BC Error: {bc_error.item():.4e}, "
                      f"PDE Loss: {loss_f.item():.4e}")
        
        # Report Adam training time
        adam_time = time.time() - start_time
        print(f"Adam optimization finished in {adam_time:.2f} seconds.")
        
        # L-BFGS optimization (if specified)
        if self.lbfgs_epochs > 0:
            print("\nStarting L-BFGS optimization...")
            start_time = time.time()
            
            # L-BFGS closure function
            def closure():
                self.optimizer_lbfgs.zero_grad()
                loss = self.loss_func()
                return loss
            
            # Run L-BFGS
            self.optimizer_lbfgs.step(closure)
            
            # Report L-BFGS training time
            lbfgs_time = time.time() - start_time
            print(f"L-BFGS optimization finished in {lbfgs_time:.2f} seconds.")
        
        # Return the trained model
        return self.model
