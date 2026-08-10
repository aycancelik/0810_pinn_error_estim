"""
generate_demo_data.py
---------------------
Generates heat equation demo data for the interactive PINN demo.

Output structure:
    data/
    └── heat/
        ├── well_trained/
        │   ├── 2x2/
        │   │   ├── error_pred.json    {"data": [[...]]}  |u_pinn - u_exact|
        │   │   ├── true_soln.json     {"data": [[...]]}  exact solution
        │   │   └── meta.json          stats + axes
        │   ├── 4x4/
        │   ├── 8x8/
        │   ├── 16x16/
        │   ├── 32x32/
        │   ├── 64x64/
        │   └── 128x128/
        └── rand_init/
            └── (same structure)

Usage
-----
    uv run docs/generate_demo_data.py
    uv run docs/generate_demo_data.py --output-dir /path/to/repo/docs/data
    uv run docs/generate_demo_data.py --no-skip
    uv run docs/generate_demo_data.py --dry-run
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
SEED   = 42
LAYERS = [2, 20, 20, 20, 1]

HEAT_KWARGS = {
    "x_min": 0.0, "x_max": 1.0, "t_max": 1.0,
    "diffusivity": 0.05, "frequency": 2,
}

GRID_SIZES = [8, 16, 32, 64, 128, 256]

MODEL_TYPES = {
    "well_trained": {
        "n_iterations": 10_000,
        "num_domain":   10_000,
        "trained":      True,
    },
    "rand_init": {
        "n_iterations": 0,
        "num_domain":   0,
        "trained":      False,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def _run_well_trained(nx: int):
    pinn_cfg = PINNConfig(
        layers=LAYERS, n_iterations=10_000, num_domain=10_000,
        seed=SEED, use_cache=True,
    )
    factory = ExperimentFactory(
        problem_name="heat",
        problem_kwargs=HEAT_KWARGS,
        fdm_solver_kwargs={"nx": nx, "nt": nx, "ny": nx},
        pinn_config=pinn_cfg,
        verbose=False,
    )
    res = factory.run_experiment()

    x      = res["x"]
    t      = res["t"]
    nt, nx_ = len(t), len(x)
    u_pinn  = np.asarray(res["u_pinn"]).reshape(nt, nx_)
    u_exact = np.asarray(res["u_true"]).reshape(nt, nx_)
    e_res   = np.asarray(res["e_res"]).reshape(nt, nx_)
    e_fdm   = np.asarray(res["e_fdm"]).reshape(nt, nx_)
    return x, t, u_pinn, u_exact, e_res, e_fdm


def _run_rand_init(nx: int):
    problem  = get_problem("heat", **HEAT_KWARGS)
    pinn_cfg = PINNConfig(
        layers=LAYERS, n_iterations=0, num_domain=100,
        seed=SEED, use_cache=False,
    )
    trainer = PINNTrainer(problem=problem, config=pinn_cfg)
    # intentionally skip trainer.train() — raw random weights

    x    = np.linspace(problem.domain.x_min, problem.domain.x_max, nx)
    t    = np.linspace(problem.domain.t_min, problem.domain.t_max, nx)
    X, T = np.meshgrid(x, t)
    u_pinn  = trainer.predict(np.column_stack((X.ravel(), T.ravel()))).reshape(nx, nx)
    u_exact = problem.exact_solution(X, T)
    e_res   = np.zeros_like(u_pinn)   # not available for untrained model
    e_fdm   = np.zeros_like(u_pinn)
    return x, t, u_pinn, u_exact, e_res, e_fdm


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(output_dir: Path, dry_run: bool = False, skip_existing: bool = True):
    total   = len(GRID_SIZES) * len(MODEL_TYPES)
    run_idx = 0
    FILES   = ["predictions.json", "true_solution.json", "error_approximation.json", "error_fdm.json", "error_true.json", "meta.json"]

    for model_key, model_cfg in MODEL_TYPES.items():
        for nx in GRID_SIZES:
            run_idx   += 1
            grid_label = f"{nx}x{nx}"
            base       = output_dir / "heat" / model_key / grid_label

            print(f"\n[{run_idx}/{total}]  heat / {model_key} / {grid_label}")

            if dry_run:
                print(f"  → {base}/  (dry-run)")
                continue

            if skip_existing and all((base / f).exists() for f in FILES):
                print(f"  ✓ already complete, skipping")
                continue

            t0 = time.time()
            try:
                if model_cfg["trained"]:
                    x, t, u_pinn, u_exact = _run_well_trained(nx)
                else:
                    x, t, u_pinn, u_exact = _run_rand_init(nx)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            error  = np.abs(u_pinn - u_exact)
            e_true = u_exact - u_pinn

            l2_rel = float(
                np.sqrt(np.mean(e_true**2)) / (np.sqrt(np.mean(u_exact**2)) + 1e-12)
            )

            _write(base / "predictions.json",        {"data": np.round(u_pinn, 6).tolist()})
            _write(base / "true_solution.json",       {"data": np.round(u_exact, 6).tolist()})
            _write(base / "error_approximation.json", {"data": np.round(error, 6).tolist()})
            _write(base / "error_fdm.json",           {"data": np.round(np.abs(e_fdm), 6).tolist()})
            _write(base / "error_true.json",           {"data": np.round(np.abs(u_exact - u_pinn), 6).tolist()})
            _write(base / "meta.json", {
                "problem":          "heat",
                "model_type":       model_key,
                "grid":             grid_label,
                "nx":               int(nx),
                "nt":               int(len(t)),
                "x_axis":           np.round(x, 6).tolist(),
                "t_axis":           np.round(t, 6).tolist(),
                "x_label":          "x",
                "y_label":          "t",
                "max_error":        float(np.max(error)),
                "mean_error":       float(np.mean(error)),
                "l2_relative":      l2_rel,
                "pinn_iterations":  model_cfg["n_iterations"],
                "pinn_collocation": model_cfg["num_domain"],
                "pinn_layers":      LAYERS,
                "seed":             SEED,
            })

            elapsed = time.time() - t0
            size_kb = sum((base / f).stat().st_size for f in FILES) / 1e3
            print(
                f"  ✓ {elapsed:.1f}s | {size_kb:.1f} KB | "
                f"max_err={np.max(error):.3e} | l2={l2_rel:.3e}"
            )

    print(f"\nDone → {output_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate heat equation demo data.")
    p.add_argument("--output-dir", default="data", help="Root output directory.")
    p.add_argument("--no-skip",  action="store_true", help="Overwrite existing files.")
    p.add_argument("--dry-run",  action="store_true", help="Print plan without running.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(Path(args.output_dir), dry_run=args.dry_run, skip_existing=not args.no_skip)
