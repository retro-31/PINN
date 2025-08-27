# Physics-Informed Neural Networks (PINNs) for Mechanical Engineering

A comprehensive implementation of Physics-Informed Neural Networks for solving partial differential equations in mechanical engineering applications. This framework provides both penalty method and Lagrange multiplier approaches for boundary condition enforcement.

## 🚀 Features

### Problem Types Supported

- **Time-dependent problems**: Burgers' equation for fluid dynamics
- **1D steady-state problems**: Heat conduction in rods with heat generation
- **2D steady-state scalar**: Heat transfer in plates
- **2D steady-state vector**: Potential flow around cylinders

### Boundary Condition Methods

- **Penalty Method**: Works for all problem types, enforces BCs through loss function
- **Lagrange Multiplier Method**: Exact BC satisfaction for 1D steady-state problems

### Engineering Applications

- Heat conduction analysis in steel rods
- Thermal management in aluminum plates
- Aerodynamic flow analysis around cylinders
- Viscous fluid dynamics (Burgers' equation)

## 📁 Project Structure

```text
PINN/
├── main.py                 # Entry point and problem orchestration
├── pinn_lib/              # Core PINN implementation
│   ├── models.py          # Neural network + physics integration
│   ├── training.py        # Training loops and loss functions
│   ├── data.py           # Training data generation
│   └── utils.py          # Visualization utilities
├── configs/              # Problem-specific configurations
│   ├── burgers_equation.py           # Time-dependent fluid dynamics
│   ├── steady_state_1d.py            # 1D heat conduction (penalty)
│   ├── steady_state_1d_lagrange.py   # 1D heat conduction (Lagrange)
│   ├── steady_state_2d_scalar.py     # 2D heat transfer
│   └── steady_state_2d_vector.py     # 2D flow problems
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/retro-31/PINN.git
cd PINN

# Create conda environment
conda create -n pinn python=3.10
conda activate pinn

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 🎯 Quick Start

### List Available Problems

```bash
python main.py --list
```

### Run Individual Problems

```bash
# Time-dependent fluid dynamics
python main.py --problem burgers

# 1D heat conduction (penalty method)
python main.py --problem 1d_steady

# 1D heat conduction (Lagrange multiplier method)
python main.py --problem 1d_lagrange

# 2D heat transfer in plate
python main.py --problem 2d_scalar

# 2D flow around cylinder
python main.py --problem 2d_vector

# Run all problems sequentially
python main.py --problem all
```

### Run Without Plots

```bash
python main.py --problem 1d_steady --no-plots
```

## 📊 Example Results

### 1D Heat Conduction

**Problem**: Steel rod with sinusoidal heat generation

- **PDE**: `d²T/dx² = -Q·sin(πx)`
- **BCs**: `T(0) = T(1) = 0`
- **Analytical**: `T(x) = (Q/π²)·sin(πx)`

Both penalty and Lagrange methods achieve:

- BC Loss: ~10⁻¹⁰ (exact satisfaction)
- PDE Loss: ~10⁻⁶ (excellent physics compliance)

### 2D Heat Transfer

**Problem**: Aluminum plate with internal heat generation

- **PDE**: `∇²T = -Q·sin(πx)·sin(πy)`
- **BCs**: `T = 0` on all boundaries
- **Result**: Smooth temperature distribution with maximum at center

### 2D Flow Analysis

**Problem**: Potential flow around circular cylinder

- **PDE**: `∇²φ = 0` (Laplace equation)
- **BCs**: No-penetration on cylinder, uniform flow at far-field
- **Result**: Classic flow pattern with stagnation points

## 🧠 Technical Details

### Neural Network Architecture

- **Activation**: Hyperbolic tangent (smooth derivatives for PDE computation)
- **Initialization**: Xavier uniform for stable training
- **Automatic Differentiation**: PyTorch autograd for exact derivatives

### Training Strategy

1. **Adam Optimizer**: Global exploration phase (5000-15000 epochs)
2. **L-BFGS Optimizer**: Local convergence phase (1000-3000 epochs)

### BC Enforcement Methods

#### Penalty Method (Default)

```python
# Loss function includes BC penalty
loss = bc_weight * loss_bc + pde_weight * loss_pde
```

- ✅ Works for all problem types
- ⚠️ Requires penalty weight tuning

#### Lagrange Multiplier Method (1D only)

```python
# Modified network output
u(x) = u_base(x) * x(1-x) + Σ λᵢ φᵢ(x)
```

- ✅ Exact BC satisfaction
- ✅ No weight tuning needed
- ❌ Limited to 1D steady-state problems

## 🔧 Configuration System

Each problem is defined by a configuration file. Example for 1D heat conduction:

```python
# configs/steady_state_1d.py

# Physical parameters
THERMAL_CONDUCTIVITY = 50.0  # W/m·K
HEAT_GENERATION_RATE = 1000.0  # W/m³

# Domain
X_RANGE = [0.0, 1.0]

# Boundary conditions
BC_LEFT = 0.0   # T(0) = 0°C
BC_RIGHT = 0.0  # T(1) = 0°C

# Network architecture
LAYERS = [1, 20, 20, 20, 1]

# Training parameters
ADAM_EPOCHS = 5000
LBFGS_EPOCHS = 1000
LEARNING_RATE = 0.001

# Method selection
BC_METHOD = "penalty"  # or "lagrange"
```

## 📈 Performance Metrics

### Typical Training Times (RTX 3080)

- 1D problems: 15-45 seconds
- 2D scalar: 60-100 seconds  
- 2D vector: 200-400 seconds

### Accuracy Metrics

- **BC Error**: 10⁻⁶ to 10⁻¹⁰ (excellent)
- **PDE Residual**: 10⁻⁵ to 10⁻⁶ (very good)
- **L² Error vs Analytical**: 10⁻³ to 10⁻⁴ (good)

## 🔬 Mathematical Background

### Physics-Informed Neural Networks

PINNs solve PDEs by minimizing a loss function that includes:

1. **PDE Residual**: `∫ |PDE(u_NN)|² dx`
2. **Boundary Conditions**: `∫ |u_NN - u_BC|² ds`
3. **Initial Conditions**: `∫ |u_NN(t=0) - u₀|² dx`

### Automatic Differentiation

```python
# Example: Computing ∂²u/∂x² for heat equation
u = model(x)
u_x = torch.autograd.grad(u, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
```

## 🎨 Visualization Features

- **1D Plots**: Solution vs analytical comparison with error metrics
- **2D Contours**: Temperature/potential distributions with colorbars
- **Error Analysis**: Quantitative accuracy assessment
- **Physical Interpretation**: Engineering context and parameter values

## 🧪 Adding New Problems

1. **Create Configuration File**:

```python
# configs/my_problem.py
PROBLEM_TYPE = "steady_state_1d"
FIELD_TYPE = "scalar"
# ... define parameters
```

1. **Add to Main Script**:

```python
# main.py
PROBLEM_CONFIGS['my_problem'] = 'my_problem'
```

1. **Implement PDE Residual** (if needed):

```python
# pinn_lib/models.py
def _get_my_pde_residual(self, x_f, x_bc=None, u_bc=None):
    # Implement your PDE
    pass
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:

```python
# Reduce batch size in config
N_F = 1000  # Instead of 2000
```

**Poor Convergence**:

```python
# Increase network size or training epochs
LAYERS = [1, 50, 50, 50, 1]
ADAM_EPOCHS = 10000
```

**BC Not Satisfied**:

```python
# Increase BC weight for penalty method
BC_WEIGHT = 10.0
# Or use Lagrange method for 1D problems
BC_METHOD = "lagrange"
```

### Validation Tools

```bash
# Check BC satisfaction
python main.py --problem 1d_lagrange --no-plots | grep "BC Loss"

# Compare methods
python main.py --problem 1d_steady && python main.py --problem 1d_lagrange
```

## 📚 References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

- Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-pde`)
3. Commit changes (`git commit -am 'Add new PDE solver'`)
4. Push to branch (`git push origin feature/new-pde`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- PyTorch team for automatic differentiation framework
- Original PINN authors (Raissi et al.) for the foundational methodology
- Mechanical engineering community for practical problem formulations

---

**For questions and support**: Open an issue or contact the maintainers.

**Citation**: If you use this code in your research, please cite the original PINN paper and this implementation.
