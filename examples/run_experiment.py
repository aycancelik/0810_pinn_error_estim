"""
Unified experiment runner for all problem types.

Supports:
- heat: 1D Heat equation
- wave: 1D Wave equation
- drift_diffusion: 1D Drift-Diffusion equation
- poisson_1d: 1D Poisson equation
- poisson_2d: 2D Poisson equation

Usage:
    python run_experiment.py --problem heat --constraint_mode {hard, soft_ic, soft} --n_iterations 1000 --nx 32 --nt 32
    python run_experiment.py --problem poisson_1d --constraint_mode {hard, soft_ic, soft} --n_iterations 1000 --nx 32
    python run_experiment.py --problem poisson_2d --constraint_mode {hard, soft_ic, soft} --n_iterations 1000 --nx 16 --ny 16

*Please note that the default constraint mode is 'hard'.  
"""

import argparse

import deepxde as dde
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np 

from pinn_error.core.pinn import PINNConfig
from pinn_error.core.error_bounds import PINNErrorBoundEstimator
from pinn_error.utils.experiment_factory import ExperimentFactory
from pinn_error.utils.plotting import (plot_experiment_results,
                                       print_error_metrics)
from pinn_error.utils.setup import set_default_device


# font setup 
try: 
    fm.fontManager.addfont('/Users/krasowski/Library/Fonts/LibertinusSerif-Regular.ttf')
    sns.set_theme(font='Libertinus Serif', style='whitegrid')
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Libertinus Serif'
    mpl.rcParams['mathtext.it'] = 'Libertinus Serif:italic'
    mpl.rcParams['mathtext.bf'] = 'Libertinus Serif:bold'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Libertinus Serif']
except Exception as e:
    print(f"Warning: Could not load custom font. Using default. Error: {e}")

sns.set_context('paper', font_scale=2)

dde.config.set_default_float("float64")

set_default_device("cpu")


def main(args):
    """Run experiment with given arguments."""

    # Pass all problem kwargs - factory will handle filtering
    problem_kwargs = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "t_max": args.t_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "diffusivity": args.diffusivity,
        "velocity_x": args.velocity_x,
        "phase_shift": args.phase_shift,
        "propagation_speed": args.propagation_speed,
        "frequency": args.frequency,
    }

    # Pass all FDM kwargs
    fdm_solver_kwargs = {
        "nx": args.nx,
        "nt": args.nt,
        "ny": args.ny,
    }

    print("=" * 60)
    print(f"RUNNING EXPERIMENT: {args.problem.upper()}")
    print("=" * 60)
    print(f"\nProblem kwargs: {problem_kwargs}")
    print(f"FDM kwargs: {fdm_solver_kwargs}")
    print(f"PINN iterations: {args.n_iterations}")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    if args.problem == "poisson_1d":
        args.layers[0] = 1

    # Create experiment factory
    experiment = ExperimentFactory(
        problem_name=args.problem,
        problem_kwargs=problem_kwargs,
        fdm_solver_kwargs=fdm_solver_kwargs,
        pinn_config=PINNConfig(
            n_iterations=args.n_iterations,
            num_domain=args.num_domain,
            seed=args.seed,
            layers=args.layers,
            constraint_mode=args.constraint_mode,
        ),
        verbose=True,
    )

    # Run experiment
    results = experiment.run_experiment()

    # Print error metrics
    print_error_metrics(results)

    # Plot results
    save_path = f"{args.problem}_results.png"
    plot_experiment_results(results, save_path=save_path, show=not args.no_show)

    if args.problem == 'heat' and not args.skip_bound_estimation:
        # Compute error bound estimate for heat equation
        bound_estimator = PINNErrorBoundEstimator(
            nx=args.nx,
            problem=experiment.problem,
            domain=experiment.problem.domain,
            pinn_model=experiment.pinn_trainer,
            use_trapezoid_norm=True
        )
        bound_results = bound_estimator.compute(t_eval = results['t'])

        print(f"Error bound run time: {bound_estimator.run_time:.2f} seconds")

        results.update(bound_results)
        print(f"Number of subintervals used for integration:")
        for t, n in zip(results['t'], results['n_sp']):
            print(f"t={t:.3f}: n_subintervals={n}")

        e_true = np.zeros_like(results['t'])
        fdm_estimated_error = np.zeros_like(results['t'])
        fdm_error = np.zeros_like(results['t'])

        for i, t in enumerate(results['t']):
            e_true[i] = bound_estimator._compute_spatial_l2_norm(results['e_true'][i])[0]
            fdm_estimated_error[i] = bound_estimator._compute_spatial_l2_norm(results['e_res'][i])[0]
            fdm_error[i] = bound_estimator._compute_spatial_l2_norm(results['u_fdm'][i].flatten() - results['u_pinn'][i].flatten())[0]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(results['t'], e_true, label=f'$e_{{\\mathrm{{true}}}}$', color='#4daf4a', lw=2.0, alpha=0.5)
        ax.plot(results['t'], fdm_estimated_error, label=f'$e_{{\\mathrm{{res}}}}$', color='#377eb8', linestyle='--', lw=1.5)
        ax.plot(results['t'], fdm_error, label=f'$e_{{\\mathrm{{FDM}}}}$', color='#ff7f00', linestyle='-', lw=1.5)
        ax.plot(results['t'], results['epsilon'], label=f'$e_{{\\mathrm{{HU}}}}$', color='#f781bf', linestyle='-.', lw=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('$L_2$ Error')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, alpha=0.3)


        metrics_text = (
            "\\begin{tabular}{lr}"
            "\\multicolumn{2}{l}{Computation time.}\\\\"
            f"PINN training & {results['pinn_time']:.3f} s\\\\"
            f"$e_{{\\mathrm{{res}}}}$ & {results['e_res_time']:.3f} s\\\\"
            f"$e_{{\\mathrm{{FDM}}}}$ & {results['e_fdm_time']:.3f} s\\\\"
            f"$e_{{\\mathrm{{bound}}}}$ & {bound_estimator.run_time:.3f} s\\\\"
            "\\end{tabular}"
        )

        ax.text(0.55, 0.05, metrics_text, verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, bbox=dict(edgecolor='black', alpha=0.8, facecolor='white'))

        plt.savefig("./figures/heat_eq_err_comparison_time.pdf", bbox_inches='tight')
        ax.legend()
        ax.set_title('Error Comparison for 1D Heat Equation')
        plt.show()


    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PINN experiment with FDM error estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem selection
    parser.add_argument(
        "--problem",
        type=str,
        choices=["heat", "wave", "drift_diffusion", "poisson_1d", "poisson_2d"],
        required=True,
        help="The problem to solve.",
    )

    # FDM grid parameters
    parser.add_argument(
        "--nx",
        type=int,
        default=64,
        help="Number of spatial grid points in x direction.",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=64,
        help="Number of temporal grid points (time-dependent problems).",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=64,
        help="Number of spatial grid points in y direction (2D problems).",
    )

    # PINN parameters
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10_000,
        help="Number of training iterations for PINN.",
    )
    parser.add_argument(
        "--num_domain",
        type=int,
        default=10_000,
        help="Number of domain collocation points for PINN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 20, 20, 20, 1],
        help="List of layer sizes for the PINN.",
    )
    parser.add_argument(
        "--constraint_mode",
        type=str,
        choices=["hard", "soft_ic", "soft"],
        default="hard",
        help=(
            "How IC/BC are enforced: 'hard' constrains both IC and BC via the "
            "network's output transform (baseline); 'soft_ic' hard-constrains "
            "BC only and learns IC via a soft loss term (not valid for "
            "time-independent problems like poisson_1d/poisson_2d, which have "
            "no IC); 'soft_full' learns both IC and BC via soft loss terms. "
            "Currently only 'heat' and 'poisson_1d' support soft modes end-to-"
            "end (FDM error estimation for the others still assumes hard "
            "IC/BC)."
        ),
    )

    # Problem-specific parameters
    parser.add_argument(
        "--frequency",
        type=int,
        default=2,
        help="Frequency/mode number for initial condition.",
    )
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=0.05,
        help="Diffusivity for Heat and Drift-Diffusion equations.",
    )
    parser.add_argument(
        "--propagation_speed",
        type=float,
        default=0.5,
        help="Propagation speed for Wave equation.",
    )
    parser.add_argument(
        "--velocity_x",
        type=float,
        default=2.0,
        help="Velocity in x direction for Drift-Diffusion equation.",
    )
    parser.add_argument(
        "--phase_shift",
        type=float,
        default=0.0,
        help="Phase shift for Drift-Diffusion equation IC.",
    )

    # Domain parameters
    parser.add_argument("--x_min", type=float, default=0.0, help="Minimum x value.")
    parser.add_argument("--x_max", type=float, default=1.0, help="Maximum x value.")
    parser.add_argument("--t_max", type=float, default=1.0, help="Maximum t value.")
    parser.add_argument(
        "--y_min", type=float, default=0.0, help="Minimum y value (2D)."
    )
    parser.add_argument(
        "--y_max", type=float, default=1.0, help="Maximum y value (2D)."
    )

    # Output options
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display the plot (only save).",
    )
    parser.add_argument(
        "--skip_bound_estimation",
        action="store_true",
        help="Skip error bound estimation step (only for heat equation).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)