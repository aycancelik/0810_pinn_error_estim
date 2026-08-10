"""
Comprehensive experiment runner for systematic evaluation.

Runs experiments across multiple problem types with varying configurations
and collects error metrics into a DataFrame for analysis.

Usage:
    python run_all_experiments.py --problems heat wave --output results.csv
    python run_all_experiments.py --problems all --output full_results.csv
    python run_all_experiments.py --problems heat --dry-run  # Preview configs
    python run_all_experiments.py --problems heat --force    # Re-run all, ignore existing
"""

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from pinn_error.core.pinn import PINNConfig
from pinn_error.utils.experiment_factory import ExperimentFactory
from pinn_error.utils.setup import set_default_device

import deepxde as dde 

# set default float precision and device
dde.config.set_default_float("float64")
set_default_device("cpu")

@dataclass
class ProblemConfig:
    """Configuration for a specific problem type."""

    name: str
    problem_kwargs: Dict[str, Any]
    nx_values: List[int]
    nt_values: Optional[List[int]]  # None for steady-state
    ny_values: Optional[List[int]]  # None for 1D problems
    n_iterations_values: List[int]
    num_domain_values: List[int]


# Grid sizes as powers of 2
GRID_SIZES = [8, 16, 32, 64, 128, 256]#, 512, 1024]

# random and well-trained
ITERATION_COUNTS = [0, 10000]
# well-trained
NUM_DOMAIN_POINTS = [1, 10, 100, 1000, 10000]

# same defaults as in run_experiment
PROBLEM_DEFAULTS = {
    "diffusivity": 0.05,
    "frequency": 2.0,
    "velocity_x": 2.0,
    "phase_shift": 0.0,
    "propagation_speed": 0.5,
    "initital_concentration": 1.0,
    "x_min": 0.0,
    "x_max": 1.0,
    "t_max": 1.0,  
    "y_min": 0.0,
    "y_max": 1.0,
}

# Problem configurations

PROBLEM_CONFIGS = {
    "heat": ProblemConfig(
        name="heat",
        problem_kwargs={
            "x_min": 0.0,
            "x_max": 1.0,
            "t_max": 1.0,
            **PROBLEM_DEFAULTS,
        },
        nx_values=GRID_SIZES,
        nt_values=GRID_SIZES,
        ny_values=None,
        n_iterations_values=ITERATION_COUNTS,
        num_domain_values=NUM_DOMAIN_POINTS,
    ),
    "wave": ProblemConfig(
        name="wave",
        problem_kwargs={
            "x_min": 0.0,
            "x_max": 1.0,
            "t_max": 1.0,
            **PROBLEM_DEFAULTS,
            
        },
        nx_values=GRID_SIZES,
        nt_values=GRID_SIZES,
        ny_values=None,
        n_iterations_values=ITERATION_COUNTS,
        num_domain_values=NUM_DOMAIN_POINTS,
    ),
    "drift_diffusion": ProblemConfig(
        name="drift_diffusion",
        problem_kwargs={
            "x_min": 0.0,
            "x_max": 1.0,
            "t_max": 1.0,
            **PROBLEM_DEFAULTS,
        },
        nx_values=GRID_SIZES,
        nt_values=GRID_SIZES,
        ny_values=None,
        n_iterations_values=ITERATION_COUNTS,
        num_domain_values=NUM_DOMAIN_POINTS,
    ),
    "poisson_1d": ProblemConfig(
        name="poisson_1d",
        problem_kwargs={
            "x_min": 0.0,
            "x_max": 1.0,
            **PROBLEM_DEFAULTS,
        },
        nx_values=GRID_SIZES,
        nt_values=None,
        ny_values=None,
        n_iterations_values=ITERATION_COUNTS,
        num_domain_values=NUM_DOMAIN_POINTS,
    ),
    "poisson_2d": ProblemConfig(
        name="poisson_2d",
        problem_kwargs={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            **PROBLEM_DEFAULTS,
        },
        nx_values=GRID_SIZES,
        nt_values=None,
        ny_values=GRID_SIZES,
        n_iterations_values=ITERATION_COUNTS,
        num_domain_values=NUM_DOMAIN_POINTS,
    ),
}

### Helper functions for config keys and results management

def create_config_key(config: Dict[str, Any], seed: int) -> str:
    """
    Create a unique key for an experiment configuration.
    """
    def _fmt(v) -> str:
        """Normalize value: integers stay as int strings, floats drop .0 if whole."""
        if v is None:
            return "None"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    fdm = config["fdm_solver_kwargs"]
    key_parts = [
        config["problem_name"],
        f"nx={_fmt(fdm['nx'])}",
        f"nt={_fmt(fdm.get('nt'))}",
        f"ny={_fmt(fdm.get('ny'))}",
        f"n_iter={_fmt(config['pinn_kwargs']['n_iterations'])}",
        f"num_dom={_fmt(config['pinn_kwargs']['num_domain'])}",
        f"seed={_fmt(seed)}",
    ]
    for k, v in sorted(config["problem_kwargs"].items()):
        key_parts.append(f"prob_{k}={_fmt(v)}")
    return "|".join(key_parts)


def create_key_from_row(row: Dict[str, Any], prob_columns: List[str]) -> str:
    """
    Reconstruct a config key from a result row.
    """
    def _fmt(v) -> str:
        if v is None:
            return "None"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    key_parts = [
        str(row["problem"]),
        f"nx={_fmt(row['nx'])}",
        f"nt={_fmt(row.get('nt'))}",
        f"ny={_fmt(row.get('ny'))}",
        f"n_iter={_fmt(row['n_iterations'])}",
        f"num_dom={_fmt(row['num_domain'])}",
        f"seed={_fmt(row['seed'])}",
    ]
    for col in sorted(prob_columns):
        key_parts.append(f"{col}={_fmt(row.get(col))}")
    return "|".join(key_parts)


def load_existing_results(output_path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not os.path.exists(output_path):
        print(f"  No existing results file found at {output_path}")
        return [], set()

    try:
        df = pl.read_csv(output_path)
        if len(df) == 0:
            print(f"  Existing results file is empty")
            return [], set()

        prob_columns = [col for col in df.columns if col.startswith("prob_")]
        results = df.to_dicts()
        completed_keys = set()

        for row in results:
            key = create_key_from_row(row, prob_columns)
            completed_keys.add(key)

        print(f"  Loaded {len(results)} existing results from {output_path}")
        print(f"  Found {len(completed_keys)} unique completed configurations")

        # --- DEBUG: show a sample key from CSV vs from config ---
        if results:
            sample_row = results[0]
            csv_key = create_key_from_row(sample_row, prob_columns)
            print(f"\n  DEBUG sample key from CSV row:\n    {csv_key}")

        return results, completed_keys

    except Exception as e:
        print(f"  Warning: Could not load existing results from {output_path}: {e}")
        return [], set()


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to CSV, deduplicating by config key columns."""
    df = pl.DataFrame(results)
    key_cols = [c for c in ["problem", "nx", "nt", "ny", "n_iterations", "num_domain", "seed"] if c in df.columns]
    df = df.unique(subset=key_cols, keep="last")
    df.write_csv(output_path)
    return df

### Metrics computation

def compute_error_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute summary error metrics from experiment results.

    Returns metrics comparing:
    - Estimated error vs true PINN error
    - PINN solution vs exact solution
    - FDM solution vs exact solution

    """
    e_true = results["e_true"]
    e_res = results["e_res"]
    e_fdm = results["e_fdm"]
    e_fdm_vs_true = results["e_fdm_vs_true"]
    u_true = results["u_true"]

    # Flatten arrays for consistent computation
    e_true_flat = e_true.ravel()
    e_res_flat = e_res.ravel()
    e_fdm_vs_true_flat = e_fdm_vs_true.ravel()
    u_true_flat = u_true.ravel()

    # Estimation quality metrics (how well does our method estimate PINN error?)
    estimation_diff = e_res_flat - e_true_flat

    metrics = {
        "pinn_error_l2": np.sqrt(np.mean(e_true_flat**2)),
        "pinn_error_relative_l2": np.sqrt(np.mean(e_true_flat**2)) / (np.sqrt(np.mean(u_true_flat**2)) + 1e-16),
        "e_res_vs_true_l2": np.sqrt(np.mean(estimation_diff**2)),
        "e_res_vs_true_relative_l2": np.sqrt(np.mean(estimation_diff**2)) / (np.sqrt(np.mean(e_true_flat**2)) + 1e-16),
        "e_fdm_vs_true_l2": np.sqrt(np.mean((e_fdm_vs_true_flat)**2)),
        "e_fdm_vs_true_relative_l2": np.sqrt(np.mean((e_fdm_vs_true_flat)**2)) / (np.sqrt(np.mean(e_true_flat**2)) + 1e-16),
    }

    return metrics


### generate experiments

def generate_experiment_configs(config: ProblemConfig) -> List[Dict[str, Any]]:
    """Generate all experiment configurations for a problem."""
    configs = []

    # For time-dependent problems
    if config.name in ["heat", "drift_diffusion", "wave"]:
        for nx, n_iter, num_dom in itertools.product(
            config.nx_values,
            config.n_iterations_values,
            config.num_domain_values,
        ):
            configs.append(
                {
                    "problem_name": config.name,
                    "problem_kwargs": config.problem_kwargs.copy(),
                    "fdm_solver_kwargs": {
                        "nx": nx,
                        "nt": nx,
                        "ny": 16,
                    },
                    "pinn_kwargs": {"n_iterations": n_iter, "num_domain": num_dom},
                }
            )

    # For 2D steady-state (Poisson 2D)
    elif config.name == "poisson_2d":
        for nx, n_iter, num_dom in itertools.product(
            config.nx_values,
            config.n_iterations_values,
            config.num_domain_values,
        ):
            configs.append(
                {
                    "problem_name": config.name,
                    "problem_kwargs": config.problem_kwargs.copy(),
                    "fdm_solver_kwargs": {"nx": nx, "nt": 32, "ny": nx},
                    "pinn_kwargs": {"n_iterations": n_iter, "num_domain": num_dom},
                }
            )

    # For 1D steady-state (Poisson 1D)
    else:
        for nx, n_iter, num_dom in itertools.product(
            config.nx_values,
            config.n_iterations_values,
            config.num_domain_values,
        ):
            configs.append(
                {
                    "problem_name": config.name,
                    "problem_kwargs": config.problem_kwargs.copy(),
                    "fdm_solver_kwargs": {"nx": nx, "nt": 32, "ny": 16},
                    "pinn_kwargs": {"n_iterations": n_iter, "num_domain": num_dom},
                }
            )

    return configs

### Experiment execution

def run_single_experiment(
    exp_config: Dict[str, Any],
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any] | None:
    """Run a single experiment and return metrics."""

    pinn_config = PINNConfig(
        n_iterations=exp_config["pinn_kwargs"]["n_iterations"],
        num_domain=exp_config["pinn_kwargs"]["num_domain"],
        seed=seed,
        use_cache=True,
    )

    try:
        # create experiment runner and execute
        experiment = ExperimentFactory(
            problem_name=exp_config["problem_name"],
            problem_kwargs=exp_config["problem_kwargs"],
            fdm_solver_kwargs=exp_config["fdm_solver_kwargs"],
            pinn_config=pinn_config,
            verbose=verbose,
        )

        # train PINN, solve FDM, compute errors
        results = experiment.run_experiment()
        # compute error metrics
        metrics = compute_error_metrics(results)

        # Build result row
        row = {
            "problem": exp_config["problem_name"],
            "nx": exp_config["fdm_solver_kwargs"]["nx"],
            "nt": exp_config["fdm_solver_kwargs"].get("nt"),
            "ny": exp_config["fdm_solver_kwargs"].get("ny"),
            "n_iterations": exp_config["pinn_kwargs"]["n_iterations"],
            "num_domain": exp_config["pinn_kwargs"]["num_domain"],
            "seed": seed,
            "pinn_time": results["pinn_time"],
            "fdm_time": results["e_fdm_time"],
            "residual_estimation_time": results["e_res_time"],
            "stability_flag": results["stability_flag"],
            **metrics,
        }

        # Add problem-specific kwargs
        for k, v in exp_config["problem_kwargs"].items():
            row[f"prob_{k}"] = v

        return row

    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR during experiment: {error_msg}")
        return None


### run all experiments for specified problems

def run_experiments(
    problems: List[str],
    output_path: str,
    seed: int = 42,
    verbose: bool = False,
    dry_run: bool = False,
    force: bool = False,
    save_interval: int = 10,
) -> pl.DataFrame:
    """Run all experiments for specified problems.
    
    Args:
        problems: List of problem names to run (e.g. ["heat", "wave"] or ["all"])
        output_path: Path to save results CSV
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output during experiments
        dry_run: If True, only generate and preview configurations without running experiments
        force: If True, ignore existing results and re-run all experiments
        save_interval: Save intermediate results to CSV every N experiments
    """

    # Generate all configs
    all_configs = []
    for problem in problems:
        if problem not in PROBLEM_CONFIGS:
            print(f"Warning: Unknown problem '{problem}', skipping.")
            continue
        configs = generate_experiment_configs(PROBLEM_CONFIGS[problem])
        all_configs.extend(configs)
        print(f"  {problem}: {len(configs)} configurations")
    
    # --- DEBUG: show a sample config key ---
    if all_configs:
        sample_key = create_config_key(all_configs[0], seed)
        print(f"\n  DEBUG sample key from config:\n    {sample_key}")

    print(f"\nTotal experiment configurations: {len(all_configs)}")

    # Dry run preview
    if dry_run:
        print("\n[DRY RUN] Preview of first 10 configurations:")
        for i, cfg in enumerate(all_configs[:10]):
            print(
                f"  {i+1}. {cfg['problem_name']}: nx={cfg['fdm_solver_kwargs']['nx']}, "
                f"nt={cfg['fdm_solver_kwargs'].get('nt')}, "
                f"n_iter={cfg['pinn_kwargs']['n_iterations']}, "
                f"num_dom={cfg['pinn_kwargs']['num_domain']}"
            )
        return pl.DataFrame()

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results (unless force flag is set)
    if force:
        print("\n[FORCE MODE] Ignoring existing results, re-running all experiments")
        results = []
        completed_keys = set()
    else:
        print("\nChecking for existing results...")
        results, completed_keys = load_existing_results(output_path)

    # Filter out already completed configs
    pending_configs = []
    for config in all_configs:
        key = create_config_key(config, seed)
        if key not in completed_keys:
            pending_configs.append(config)

    skipped_count = len(all_configs) - len(pending_configs)
    print(f"\nSkipping {skipped_count} already completed experiments")
    print(f"Running {len(pending_configs)} pending experiments")

    if len(pending_configs) == 0:
        print("\nAll experiments already completed!")
        return pl.DataFrame(results) if results else pl.DataFrame()

    # Run pending experiments
    start_time = time.time()
    new_results_count = 0

    for i, config in enumerate(pending_configs):
        exp_start = time.time()

        print(
            f"\n[{i+1}/{len(pending_configs)}] Running {config['problem_name']} "
            f"(nx={config['fdm_solver_kwargs']['nx']}, "
            f"nt={config['fdm_solver_kwargs'].get('nt')}, "
            f"n_iter={config['pinn_kwargs']['n_iterations']})..."
        )

        row = run_single_experiment(config, seed=seed, verbose=verbose)
        if row is not None:
            results.append(row)
        new_results_count += 1

        exp_time = time.time() - exp_start

        if row is not None:
            print(
                f"  Done in {exp_time:.2f}s | e_res_vs_true_l2={row['e_res_vs_true_l2']:.2e}"
            )

        # Periodic save
        if new_results_count % save_interval == 0:
            save_results(results, output_path)
            print(
                f"  [Saved checkpoint: {len(results)} total results to {output_path}]"
            )

    # Final save
    df = save_results(results, output_path)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed {new_results_count} new experiments in {total_time:.1f}s")
    print(
        f"Total results: {len(results)} (including {skipped_count} previously completed)"
    )
    if new_results_count - len(results) > 0:
        print(f"WARNING: {new_results_count - len(results)} experiments were expected but not completed successfully.")
    print(f"Results saved to: {output_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments across problem types",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--problems",
        nargs="+",
        default=["all"],
        choices=["all", "heat", "wave", "drift_diffusion", "poisson_1d", "poisson_2d"],
        help="Problems to run experiments for.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/all_experiment_results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview configurations without running experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all experiments, ignoring existing results.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N experiments.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve "all" to full list
    if "all" in args.problems:
        problems = list(PROBLEM_CONFIGS.keys())
    else:
        problems = args.problems

    print("=" * 60)
    print("COMPREHENSIVE EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Problems: {problems}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Force re-run: {args.force}")
    print("-" * 60)

    df = run_experiments(
        problems=problems,
        output_path=args.output,
        seed=args.seed,
        verbose=args.verbose,
        dry_run=args.dry_run,
        force=args.force,
        save_interval=args.save_interval,
    )

    if not args.dry_run and len(df) > 0:
        print("\nSample of final results:")
        print(df.head())


if __name__ == "__main__":
    main()
