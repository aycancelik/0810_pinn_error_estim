"""
generate_demo_data_v2.py
------------------------
Generates demo data for all benchmark problems.

Output structure:
    data/
    ├── heat/
    │   ├── well_trained/
    │   │   ├── 8x8/
    │   │   │   ├── predictions.json
    │   │   │   ├── true_solution.json
    │   │   │   ├── error_true.json
    │   │   │   ├── error_approximation.json
    │   │   │   ├── error_fdm.json
    │   │   │   └── meta.json
    │   │   └── ...256x256/
    │   └── rand_init/
    │       └── ...
    ├── wave/         (same structure)
    ├── drift_diffusion/ (same structure)
    └── poisson_2d/   (same structure)

Usage
-----
    uv run docs/generate_demo_data_v2.py
    uv run docs/generate_demo_data_v2.py --problems wave
    uv run docs/generate_demo_data_v2.py --output-dir /path/to/repo/docs/data
    uv run docs/generate_demo_data_v2.py --no-skip
    uv run docs/generate_demo_data_v2.py --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import deepxde as dde

from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.utils.experiment_factory import ExperimentFactory, get_problem

# ── Device + dtype ─────────────────────────────────────────────────────────────
_MPS = torch.backends.mps.is_built() and torch.backends.mps.is_available()
if _MPS:
    print("MPS detected — forcing CPU + float64 for numerical stability.")
torch.set_default_device("cpu")
dde.config.set_default_float("float64")

# ── Fixed settings ─────────────────────────────────────────────────────────────
SEED       = 42
LAYERS     = [2, 20, 20, 20, 1]
GRID_SIZES = [8, 16, 32, 64, 128, 256]

MODEL_TYPES = {
    "well_trained": {"n_iterations": 10_000, "num_domain": 10_000, "trained": True},
    "rand_init":    {"n_iterations": 0,       "num_domain": 0,       "trained": False},
}

FILES = [
    "predictions.json",
    "true_solution.json",
    "error_true.json",
    "error_approximation.json",
    "error_fdm.json",
    "meta.json",
]

# ── Problem definitions ────────────────────────────────────────────────────────
PROBLEMS = {
    "heat": {
        "label":       "1-D Heat Equation",
        "kwargs":      {"x_min":0.0,"x_max":1.0,"t_max":1.0,"diffusivity":0.05,"frequency":2},
        "time_dep":    True,
        "is_2d":       False,
    },
    "wave": {
        "label":       "1-D Wave Equation",
        "kwargs":      {"x_min":0.0,"x_max":1.0,"t_max":1.0,"propagation_speed":0.5,"frequency":2},
        "time_dep":    True,
        "is_2d":       False,
    },
    "drift_diffusion": {
        "label":       "1-D Drift-Diffusion",
        "kwargs":      {"x_min":0.0,"x_max":1.0,"t_max":1.0,"diffusivity":0.05,"velocity_x":2.0,"frequency":2.0,"phase_shift":0.0,"initial_concentration":1.0},
        "time_dep":    True,
        "is_2d":       False,
    },
    "poisson_2d": {
        "label":       "2-D Poisson Equation",
        "kwargs":      {"x_min":0.0,"x_max":1.0,"y_min":0.0,"y_max":1.0},
        "time_dep":    False,
        "is_2d":       True,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def _run_well_trained(prob_key: str, prob_cfg: dict, nx: int):
    pinn_cfg = PINNConfig(
        layers=LAYERS, n_iterations=10_000, num_domain=10_000,
        seed=SEED, use_cache=True,
    )
    factory = ExperimentFactory(
        problem_name=prob_key,
        problem_kwargs=prob_cfg["kwargs"],
        fdm_solver_kwargs={"nx": nx, "nt": nx, "ny": nx},
        pinn_config=pinn_cfg,
        verbose=False,
    )
    res = factory.run_experiment()

    x       = res["x"]
    nt, nx_ = len(res["t"] if prob_cfg["time_dep"] else res["y"]), len(x)
    axis2   = res["t"] if prob_cfg["time_dep"] else res["y"]

    u_pinn  = np.asarray(res["u_pinn"]).reshape(nt, nx_)
    u_exact = np.asarray(res["u_true"]).reshape(nt, nx_)
    e_res   = np.asarray(res["e_res"]).reshape(nt, nx_)
    e_fdm   = np.asarray(res["e_fdm"]).reshape(nt, nx_)

    timing = {
        "pinn_training_s":  round(float(res.get("pinn_time", 0.0)), 3),
        "err_res_s":        round(float(res.get("e_res_time", 0.0)), 3),
        "err_fdm_s":        round(float(res.get("e_fdm_time", 0.0)), 3),
    }
    return x, axis2, u_pinn, u_exact, e_res, e_fdm, timing


def _run_rand_init(prob_key: str, prob_cfg: dict, nx: int):
    """Run ExperimentFactory with 0 training iterations so e_res and e_fdm
    are properly computed on the untrained (random-weight) PINN."""
    pinn_cfg = PINNConfig(
        layers=LAYERS, n_iterations=0, num_domain=100,
        seed=SEED, use_cache=False,
    )
    factory = ExperimentFactory(
        problem_name=prob_key,
        problem_kwargs=prob_cfg["kwargs"],
        fdm_solver_kwargs={"nx": nx, "nt": nx, "ny": nx},
        pinn_config=pinn_cfg,
        verbose=False,
    )
    res = factory.run_experiment()

    x      = res["x"]
    axis2  = res["t"] if prob_cfg["time_dep"] else res["y"]
    n2, nx_ = len(axis2), len(x)

    u_pinn  = np.asarray(res["u_pinn"]).reshape(n2, nx_)
    u_exact = np.asarray(res["u_true"]).reshape(n2, nx_)
    e_res   = np.asarray(res["e_res"]).reshape(n2, nx_)
    e_fdm   = np.asarray(res["e_fdm"]).reshape(n2, nx_)

    timing = {
        "pinn_training_s":  round(float(res.get("pinn_time", 0.0)), 3),
        "err_res_s":        round(float(res.get("e_res_time", 0.0)), 3),
        "err_fdm_s":        round(float(res.get("e_fdm_time", 0.0)), 3),
    }
    return x, axis2, u_pinn, u_exact, e_res, e_fdm, timing


def _save(base: Path, prob_key: str, prob_cfg: dict, model_key: str, model_cfg: dict,
          nx: int, x, axis2, u_pinn, u_exact, e_res, e_fdm, timing: dict):

    error_true = u_exact - u_pinn
    # Hard constraints: zero out boundaries for ALL error arrays
    # (should be 0 by construction but floating point leaves tiny residuals).
    def _zero_boundaries(arr):
        if prob_cfg["time_dep"]:
            arr[0, :]  = 0.0   # t = 0 (IC)
            arr[:, 0]  = 0.0   # x = x_min (BC)
            arr[:, -1] = 0.0   # x = x_max (BC)
        else:
            arr[0, :]  = 0.0   # y = y_min
            arr[-1, :] = 0.0   # y = y_max
            arr[:, 0]  = 0.0   # x = x_min
            arr[:, -1] = 0.0   # x = x_max
    _zero_boundaries(error_true)
    _zero_boundaries(e_res)
    _zero_boundaries(e_fdm)
    l2_rel = float(
        np.sqrt(np.mean(error_true**2)) / (np.sqrt(np.mean(u_exact**2)) + 1e-12)
    )

    axis2_label = "t" if prob_cfg["time_dep"] else "y"
    axis2_key   = "t_axis" if prob_cfg["time_dep"] else "y_axis"

    _write(base / "predictions.json",        {"data": np.round(u_pinn,              6).tolist()})
    _write(base / "true_solution.json",       {"data": np.round(u_exact,             6).tolist()})
    _write(base / "error_true.json",          {"data": np.round(error_true,          6).tolist()})
    _write(base / "error_approximation.json", {"data": np.round(e_res,               6).tolist()})
    _write(base / "error_fdm.json",           {"data": np.round(e_fdm,               6).tolist()})
    _write(base / "meta.json", {
        "problem":          prob_key,
        "label":            prob_cfg["label"],
        "model_type":       model_key,
        "grid":             f"{nx}x{nx}",
        "nx":               int(nx),
        "n2":               int(len(axis2)),
        "x_axis":           np.round(x,     6).tolist(),
        axis2_key:          np.round(axis2, 6).tolist(),
        "x_label":          "x",
        "y_label":          axis2_label,
        "max_error":        float(np.max(np.abs(error_true))),
        "mean_error":       float(np.mean(np.abs(error_true))),
        "l2_relative":      l2_rel,
        "pinn_iterations":  model_cfg["n_iterations"],
        "pinn_collocation": model_cfg["num_domain"],
        "pinn_layers":      LAYERS,
        "seed":             SEED,
        "pinn_training_s":  timing["pinn_training_s"],
        "err_res_s":        timing["err_res_s"],
        "err_fdm_s":        timing["err_fdm_s"],
    })

    return l2_rel, float(np.max(np.abs(error_true)))


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(
    selected: list,
    output_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    total   = len(selected) * len(MODEL_TYPES) * len(GRID_SIZES)
    run_idx = 0

    for prob_key in selected:
        prob_cfg = PROBLEMS[prob_key]
        for model_key, model_cfg in MODEL_TYPES.items():
            for nx in GRID_SIZES:
                run_idx   += 1
                grid_label = f"{nx}x{nx}"
                base       = output_dir / prob_key / model_key / grid_label

                print(f"\n[{run_idx}/{total}]  {prob_key} / {model_key} / {grid_label}")

                if dry_run:
                    print(f"  → {base}/  (dry-run)")
                    continue

                if skip_existing and all((base / f).exists() for f in FILES):
                    print(f"  ✓ already complete, skipping")
                    continue

                t0 = time.time()
                try:
                    if model_cfg["trained"]:
                        x, axis2, u_pinn, u_exact, e_res, e_fdm, timing = _run_well_trained(prob_key, prob_cfg, nx)
                    else:
                        x, axis2, u_pinn, u_exact, e_res, e_fdm, timing = _run_rand_init(prob_key, prob_cfg, nx)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    continue

                l2_rel, max_err = _save(base, prob_key, prob_cfg, model_key, model_cfg,
                                        nx, x, axis2, u_pinn, u_exact, e_res, e_fdm, timing)

                elapsed = time.time() - t0
                size_kb = sum((base / f).stat().st_size for f in FILES) / 1e3
                print(f"  ✓ {elapsed:.1f}s | {size_kb:.1f} KB | max_err={max_err:.3e} | l2={l2_rel:.3e}")

    print(f"\nDone → {output_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate PINN demo data for all problems.")
    p.add_argument("--problems", nargs="+", choices=list(PROBLEMS.keys()),
                   default=list(PROBLEMS.keys()), help="Which problems to run.")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--no-skip",  action="store_true")
    p.add_argument("--dry-run",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        selected=args.problems,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        skip_existing=not args.no_skip,
    )
