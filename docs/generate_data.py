"""
generate_data.py
----------------
Pre-compute PINN experiment results and save them to data.json
for the interactive HTML demo (demo_version2.html).

The HTML expects data.json next to the HTML file with this schema:
{
  "problems": {
    "<problem_key>": {
      "label":               str,          # display name
      "description":         str,          # one-liner shown in the UI
      "grid_sizes":          [int, ...],   # FDM grid options (N → N×N or N×1)
      "collocation_points":  [int, ...],   # PINN num_domain options
      "results": {
        "g<N>_c<C>": {
          "x_axis":  [float, ...],         # 1-D array (length nx)
          "y_axis":  [float, ...],         # 1-D array (length nt or ny)
          "pinn":    [[float, ...],...],   # 2-D matrix [ny/nt][nx]
          "fdm":     [[float, ...],...],   # 2-D matrix (exact solution)
          "error":   [[float, ...],...],   # |pinn - exact|
          "stats": {
            "max_error":    float,
            "mean_error":   float,
            "l2_relative":  float
          }
        }
      }
    }
  }
}

Usage
-----
    python generate_data.py              # run all problems with all combos
    python generate_data.py --dry-run    # print what would be run, no training
    python generate_data.py --problems heat poisson_1d   # subset of problems
    python generate_data.py --output my_data.json        # custom output path

Requirements
------------
    pip install deepxde torch numpy scipy
    (your pinn_error package must be importable)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import deepxde as dde

# ── project imports ───────────────────────────────────────────────────────────
from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.utils.experiment_factory import ExperimentFactory
from pinn_error.utils.setup import set_default_device

set_default_device("cpu")  
dde.config.set_default_float("float64")

# ── experiment grid ────────────────────────────────────────────────────────────
# Each entry defines one problem tab in the demo.
# grid_sizes    → nx (and nt for time-dependent, ny for 2-D)
# collocations  → num_domain points for PINN
# pinn_iters    → training iterations (shared across collocation variants here;
#                 adjust if you want more nuance)
PROBLEMS = {
    "heat": {
        "label":        "1-D Heat Equation",
        "description":  "u_t = α u_xx  ·  sinusoidal IC, zero Dirichlet BCs",
        "grid_sizes":   [16, 32, 64],
        "collocations": [500, 2000, 5000],
        "pinn_iters":   5000,
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 1.0,
            "diffusivity": 0.05, "frequency": 2,
        },
        "time_dependent": True,
        "is_2d": False,
    },
    "wave": {
        "label":        "1-D Wave Equation",
        "description":  "u_tt = c² u_xx  ·  sinusoidal IC, zero Dirichlet BCs",
        "grid_sizes":   [16, 32, 64],
        "collocations": [500, 2000, 5000],
        "pinn_iters":   5000,
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 1.0,
            "propagation_speed": 0.5, "frequency": 1,
        },
        "time_dependent": True,
        "is_2d": False,
    },
    "drift_diffusion": {
        "label":        "1-D Drift-Diffusion",
        "description":  "u_t + β u_x = D u_xx  ·  sinusoidal IC",
        "grid_sizes":   [16, 32, 64],
        "collocations": [500, 2000, 5000],
        "pinn_iters":   5000,
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 0.5,
            "diffusivity": 0.05, "velocity_x": 2.0,
            "frequency": 2.0, "phase_shift": 0.0,
            "initial_concentration": 1.0,
        },
        "time_dependent": True,
        "is_2d": False,
    },
    "poisson_1d": {
        "label":        "1-D Poisson Equation",
        "description":  "-u_xx = f(x)  ·  zero Dirichlet BCs",
        "grid_sizes":   [16, 32, 64],
        "collocations": [500, 2000, 5000],
        "pinn_iters":   3000,
        "problem_kwargs": {"x_min": 0.0, "x_max": 1.0},
        "time_dependent": False,
        "is_2d": False,
    },
    "poisson_2d": {
        "label":        "2-D Poisson Equation",
        "description":  "-(u_xx + u_yy) = f(x,y)  ·  zero Dirichlet BCs",
        "grid_sizes":   [16, 32, 64],
        "collocations": [500, 2000, 5000],
        "pinn_iters":   5000,
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
        },
        "time_dependent": False,
        "is_2d": True,
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_2d_list(arr: np.ndarray) -> list:
    """Convert a numpy array to a JSON-serialisable nested list."""
    return arr.tolist()


def _stats(e_true: np.ndarray, u_exact: np.ndarray) -> dict:
    """Compute error statistics."""
    abs_err = np.abs(e_true)
    l2_rel = float(
        np.sqrt(np.mean(e_true**2)) / (np.sqrt(np.mean(u_exact**2)) + 1e-12)
    )
    return {
        "max_error":   float(np.max(abs_err)),
        "mean_error":  float(np.mean(abs_err)),
        "l2_relative": l2_rel,
    }


def _reshape_result(results: dict, prob_cfg: dict) -> dict:
    """
    Extract x_axis, y_axis, pinn, fdm (exact), error matrices
    from ExperimentFactory results dict.

    For time-dependent 1-D problems:   rows = t, cols = x
    For 2-D steady-state problems:     rows = y, cols = x
    For 1-D steady-state (Poisson 1D): rows = [single row], cols = x
    """
    x = results["x"]  # shape (nx,)
    u_pinn  = np.asarray(results["u_pinn"])
    u_exact = np.asarray(results["u_true"])
    e_true  = u_exact - u_pinn   # true error (pointwise)

    if prob_cfg["time_dependent"]:
        t = results["t"]                  # shape (nt,)
        nt, nx = len(t), len(x)
        u_pinn  = u_pinn.reshape(nt, nx)
        u_exact = u_exact.reshape(nt, nx)
        e_true  = e_true.reshape(nt, nx)
        y_axis  = t.tolist()

    elif prob_cfg["is_2d"]:
        y = results["y"]                  # shape (ny,)
        ny, nx = len(y), len(x)
        u_pinn  = u_pinn.reshape(ny, nx)
        u_exact = u_exact.reshape(ny, nx)
        e_true  = e_true.reshape(ny, nx)
        y_axis  = y.tolist()

    else:
        # 1-D steady state: wrap in extra dimension so Plotly heatmap works
        u_pinn  = u_pinn.reshape(1, -1)
        u_exact = u_exact.reshape(1, -1)
        e_true  = e_true.reshape(1, -1)
        y_axis  = [0.0]

    return {
        "x_axis": x.tolist(),
        "y_axis": y_axis,
        "pinn":   _to_2d_list(u_pinn),
        "fdm":    _to_2d_list(u_exact),       # "fdm" key in HTML = exact/reference
        "error":  _to_2d_list(np.abs(e_true)),
        "stats":  _stats(e_true, u_exact),
    }


# ── main generation loop ───────────────────────────────────────────────────────

def generate(
    selected_problems: list[str],
    output_path: Path,
    dry_run: bool = False,
    seed: int = 42,
):
    db = {"problems": {}}

    total_runs = sum(
        len(PROBLEMS[p]["grid_sizes"]) * len(PROBLEMS[p]["collocations"])
        for p in selected_problems
    )
    run_idx = 0

    for prob_key in selected_problems:
        cfg = PROBLEMS[prob_key]
        print(f"\n{'='*60}")
        print(f"  Problem: {cfg['label']}")
        print(f"{'='*60}")

        prob_entry = {
            "label":               cfg["label"],
            "description":         cfg["description"],
            "grid_sizes":          cfg["grid_sizes"],
            "collocation_points":  cfg["collocations"],
            "results":             {},
        }

        for nx in cfg["grid_sizes"]:
            for num_domain in cfg["collocations"]:
                run_idx += 1
                result_key = f"g{nx}_c{num_domain}"
                print(f"\n[{run_idx}/{total_runs}] {prob_key}  grid={nx}  colloc={num_domain}")

                if dry_run:
                    print("  (dry-run, skipping)")
                    prob_entry["results"][result_key] = {"skipped": True}
                    continue

                # ── build FDM kwargs ───────────────────────────────────────
                fdm_kwargs = {"nx": nx}
                if cfg["time_dependent"]:
                    fdm_kwargs["nt"] = nx          # square grid in (x, t)
                if cfg["is_2d"]:
                    fdm_kwargs["ny"] = nx          # square grid in (x, y)

                # ── PINN config ────────────────────────────────────────────
                layers = [2, 20, 20, 20, 1]
                if prob_key == "poisson_1d":
                    layers[0] = 1                  # 1-D input

                pinn_cfg = PINNConfig(
                    layers=layers,
                    n_iterations=cfg["pinn_iters"],
                    num_domain=num_domain,
                    seed=seed,
                    use_cache=True,               # cache trained models
                )

                # ── run experiment ─────────────────────────────────────────
                t0 = time.time()
                try:
                    factory = ExperimentFactory(
                        problem_name=prob_key,
                        problem_kwargs=cfg["problem_kwargs"],
                        fdm_solver_kwargs=fdm_kwargs,
                        pinn_config=pinn_cfg,
                        verbose=False,
                    )
                    results = factory.run_experiment()
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    prob_entry["results"][result_key] = {"error": str(exc)}
                    continue

                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.1f}s")

                # ── reshape + store ────────────────────────────────────────
                shaped = _reshape_result(results, cfg)
                print(
                    f"  max_err={shaped['stats']['max_error']:.3e}  "
                    f"l2_rel={shaped['stats']['l2_relative']:.3e}"
                )
                prob_entry["results"][result_key] = shaped

        db["problems"][prob_key] = prob_entry

    # ── write JSON ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f, separators=(",", ":"))   # compact — no pretty-print

    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✓ Saved {output_path}  ({size_mb:.2f} MB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data.json for the PINN interactive demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=list(PROBLEMS.keys()),
        default=list(PROBLEMS.keys()),
        help="Which problems to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PINN training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without training anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        selected_problems=args.problems,
        output_path=Path(args.output),
        dry_run=args.dry_run,
        seed=args.seed,
    )
