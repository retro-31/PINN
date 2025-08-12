# Manages the model training process.

import torch
import numpy as np
import time

class Trainer:
    def __init__(self, model, config, data):
        self.model = model.to(config.DEVICE)
        self.X_u_train, self.u_train, self.X_f_train = [d.to(config.DEVICE) for d in data]
        self.epochs = config.ADAM_EPOCHS
        self.device = config.DEVICE
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=50000, max_eval=50000,
            history_size=50, tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
        )
        self.iter = 0

    def loss_func(self):
        self.optimizer.zero_grad()
        u_pred = self.model(self.X_u_train)
        f_pred = self.model.get_pde_residual(self.X_f_train)
        loss_u = torch.mean((self.u_train - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        loss = loss_u + loss_f
        loss.backward()
        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {loss.item():.4e}")
        self.iter += 1
        return loss

    def train(self):
        print("Starting Adam optimization...")
        start_time = time.time()
        self.model.train()
        for epoch in range(self.epochs):
            u_pred = self.model(self.X_u_train)
            f_pred = self.model.get_pde_residual(self.X_f_train)
            loss = torch.mean((self.u_train - u_pred) ** 2) + torch.mean(f_pred ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4e}")
        print(f"Adam optimization finished in {time.time() - start_time:.2f} seconds.")

        print("\nStarting L-BFGS optimization...")
        start_time_lbfgs = time.time()
        self.optimizer_lbfgs.step(self.loss_func)
        print(f"L-BFGS optimization finished in {time.time() - start_time_lbfgs:.2f} seconds.")
