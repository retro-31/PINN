# main.py
"""
Extended Modular Physics-Informed Neural Networks (PINNs) Framework

This is the main execution script for running PINN models on different types of problems:
- Time-dependent PDEs (e.g., Burgers' equation)
- Steady-state 1D problems (e.g., heat conduction in rod)
- 2D scalar fields (e.g., heat transfer in plates)
- 2D vector fields (e.g., aerodynamic flow around cylinders)

All problems focus on mechanical engineering applications including:
- Heat transfer and thermal management
- Fluid mechanics and aerodynamics
- Structural analysis and thermal stress

Usage:
    python main.py --problem burgers      # Time-dependent Burgers equation
    python main.py --problem 1d_steady    # 1D heat conduction in steel rod
    python main.py --problem 2d_scalar    # 2D heat transfer in aluminum plate
    python main.py --problem 2d_vector    # Aerodynamic flow around cylinder
    python main.py --problem all          # Run all examples sequentially
"""

import torch
import numpy as np
import importlib
import argparse
import time
from typing import Dict, List

# Import from the pinn_lib package
from pinn_lib.data import DataGenerator
from pinn_lib.models import PINN
from pinn_lib.training import Trainer
from pinn_lib.utils import plot_solution

# Problem type mapping for user-friendly names
PROBLEM_CONFIGS: Dict[str, str] = {
    'burgers': 'burgers_equation',
    '1d_steady': 'steady_state_1d', 
    '2d_scalar': 'steady_state_2d_scalar',
    '2d_vector': 'steady_state_2d_vector',
}

# Problem descriptions for mechanical engineering context
PROBLEM_DESCRIPTIONS: Dict[str, str] = {
    'burgers': 'Time-dependent viscous Burgers equation - Fundamental fluid dynamics',
    '1d_steady': '1D heat conduction in steel rod (penalty method)',
    '2d_scalar': '2D heat transfer in aluminum plate (penalty method)',
    '2d_vector': 'Aerodynamic flow around circular cylinder (penalty method)',
}

def print_available_problems():
    """Print all available problems with descriptions."""
    print("\n" + "="*80)
    print("AVAILABLE MECHANICAL ENGINEERING PROBLEMS")
    print("="*80)
    for key, description in PROBLEM_DESCRIPTIONS.items():
        print(f"  {key:12} - {description}")
    print("  all          - Run all problems sequentially")
    print("="*80)

def run_single_problem(problem_key: str, show_plots: bool = True) -> bool:
    """
    Run a single PINN problem.
    
    Args:
        problem_key: Key identifying the problem type
        show_plots: Whether to display visualization plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    if problem_key not in PROBLEM_CONFIGS:
        print(f"Error: Unknown problem '{problem_key}'")
        print_available_problems()
        return False
    
    config_name = PROBLEM_CONFIGS[problem_key]
    
    # Dynamically import the specified configuration module
    try:
        config = importlib.import_module(f"configs.{config_name}")
    except ImportError:
        print(f"Error: Configuration file 'configs/{config_name}.py' not found.")
        return False

    print(f"\n{'='*80}")
    print(f"RUNNING: {PROBLEM_DESCRIPTIONS[problem_key]}")
    print(f"Config: {config_name}")
    print(f"{'='*80}")

    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    print(f"Device: {config.DEVICE}")
    print(f"Problem Type: {getattr(config, 'PROBLEM_TYPE', 'time_dependent')}")
    print(f"Field Type: {getattr(config, 'FIELD_TYPE', 'scalar')}")
    
    start_time = time.time()

    try:
        # --- Step 1: Generate Data ---
        print("\nStep 1: Generating training data...")
        data_gen = DataGenerator(config)
        training_data = data_gen.generate_data()
        print(f"  Generated {len(training_data)} data components")

        # --- Step 2: Create Model ---
        print("\nStep 2: Creating PINN model...")
        pinn_model = PINN(config.LAYERS, config)
        print(f"  Model Architecture: {config.LAYERS}")
        
        # Count parameters
        total_params = sum(p.numel() for p in pinn_model.parameters())
        print(f"  Total Parameters: {total_params:,}")

        # --- Step 3: Train Model ---
        print("\nStep 3: Training model...")
        
        # Display boundary condition enforcement method
        lagrangian_alpha = getattr(config, 'LAGRANGIAN_ALPHA', 0.0)
        if lagrangian_alpha > 0:
            print(f"  Using hybrid method: {(1-lagrangian_alpha)*100:.0f}% penalty, {lagrangian_alpha*100:.0f}% Lagrangian")
        else:
            print("  Using penalty method for boundary conditions")
            
        trainer = Trainer(pinn_model, config, training_data)
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")

        # --- Step 4: Visualize Results ---
        if show_plots:
            print("\nStep 4: Generating visualization...")
            plot_solution(pinn_model, config)
            print("  Plots saved and displayed")
        
        total_time = time.time() - start_time
        print(f"\n✅ Problem '{problem_key}' completed successfully in {total_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running problem '{problem_key}': {str(e)}")
        return False

def run_all_problems():
    """Run all available problems sequentially."""
    print("\n" + "="*80)
    print("RUNNING ALL MECHANICAL ENGINEERING PROBLEMS")
    print("="*80)
    
    results = {}
    total_start_time = time.time()
    
    for problem_key in PROBLEM_CONFIGS.keys():
        success = run_single_problem(problem_key, show_plots=False)
        results[problem_key] = success
        
        if not success:
            print(f"\n⚠️  Stopping execution due to failure in '{problem_key}'")
            break
            
        print("\n" + "-"*50)
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    for problem_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {problem_key:12} - {status}")
    
    print(f"\nOverall: {successful}/{total} problems completed successfully")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if successful == total:
        print("\n🎉 All mechanical engineering problems completed successfully!")
    else:
        print(f"\n⚠️  {total - successful} problem(s) failed")

def main():
    """Main execution function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extended PINN Framework for Mechanical Engineering Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --problem burgers       # Run time-dependent Burgers equation
  python main.py --problem 1d_steady     # Run 1D heat conduction (penalty method)
  python main.py --problem 2d_scalar     # Run 2D heat transfer (penalty method)
  python main.py --problem 2d_vector     # Run aerodynamic flow problem (penalty method)
  python main.py --problem all           # Run all problems sequentially
  python main.py --list                  # List all available problems

For backward compatibility:
  python main.py --config burgers_equation    # Use config file directly
        """
    )
    
    parser.add_argument(
        '--problem', type=str, 
        help='Problem type to run (burgers, 1d_steady, 2d_scalar, 2d_vector, all)'
    )
    
    parser.add_argument(
        '--config', type=str,
        help='Configuration file name (for backward compatibility)'
    )
    
    parser.add_argument(
        '--list', action='store_true',
        help='List all available problems'
    )
    
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip visualization plots'
    )
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        print_available_problems()
        return
    
    # Handle backward compatibility with --config
    if args.config:
        print("⚠️  Using legacy --config option. Consider using --problem instead.")
        # Find matching problem key for the config
        problem_key = None
        for key, config_name in PROBLEM_CONFIGS.items():
            if config_name == args.config:
                problem_key = key
                break
        
        if problem_key:
            run_single_problem(problem_key, show_plots=not args.no_plots)
        else:
            # Direct config execution for backward compatibility
            try:
                config = importlib.import_module(f"configs.{args.config}")
                torch.manual_seed(config.SEED)
                np.random.seed(config.SEED)
                
                data_gen = DataGenerator(config)
                training_data = data_gen.generate_data()
                
                pinn_model = PINN(config.LAYERS, config)
                trainer = Trainer(pinn_model, config, training_data)
                trainer.train()
                
                if not args.no_plots:
                    plot_solution(pinn_model, config)
                    
            except ImportError:
                print(f"Error: Configuration file 'configs/{args.config}.py' not found.")
        return
    
    # Handle --problem option
    if args.problem:
        if args.problem == 'all':
            run_all_problems()
        else:
            run_single_problem(args.problem, show_plots=not args.no_plots)
        return
    
    # Default behavior - show help and available problems
    print("No problem specified. Use --problem to select a problem to run.")
    print_available_problems()
    print("\nFor detailed help: python main.py --help")

if __name__ == "__main__":
    main()
