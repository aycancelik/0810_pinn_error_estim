"""
Plotting utilities for PINN error estimation experiments.

Provides unified plotting functions for:
- Time slices (1D time-dependent problems)
- Y slices (2D steady-state problems)
- Heatmaps for solutions and errors
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive figure for experiment results.

    Handles:
    - 1D time-dependent problems (heat, wave, drift_diffusion)
    - 2D steady-state problems (poisson_2d)
    - 1D steady-state problems (poisson_1d)

    Parameters
    ----------
    results : dict
        Results dictionary from ExperimentFactory.run_experiment()
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    problem_name = results["problem_name"]

    if problem_name == "poisson_1d":
        fig = _plot_poisson_1d(results)
    elif problem_name == "poisson_2d":
        fig = _plot_poisson_2d(results)
    else:
        # Time-dependent problems: heat, wave, drift_diffusion
        fig = _plot_time_dependent(results)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def _plot_time_dependent(results: Dict[str, Any]) -> plt.Figure:
    """Plot results for time-dependent 1D problems."""
    x = results["x"]
    t = results["t"]
    u_pinn = results["u_pinn"]
    u_exact = results["u_true"]
    u_fdm = results["u_fdm"]
    err_estimated = results["e_res"]
    e_true = results["e_true"]
    problem_name = results["problem_name"]
    residual = results.get("residual")

    # Ensure correct shape (nt, nx)
    nt, nx = len(t), len(x)
    if u_fdm.shape != (nt, nx):
        u_fdm = u_fdm.reshape(nt, nx)
    if err_estimated.shape != (nt, nx):
        err_estimated = err_estimated.reshape(nt, nx)

    # Compute derived errors
    fdm_based_error = u_fdm - u_pinn  # Classic FDM error estimate
    residual_diff = e_true - err_estimated  # How well residual method works
    fdm_diff = e_true - fdm_based_error  # How well FDM method works

    # Create figure: 4 rows x 6 cols
    # Row 1: Time slices (PINN vs Exact)
    # Row 2: Time slices (Errors comparison)
    # Row 3: Heatmaps (u_pinn, u_exact, u_fdm)
    # Row 4: Error heatmaps (e_true, err_estimated, fdm_based_error, residual_diff, fdm_diff)

    fig = plt.figure(figsize=(24, 20))

    # Time indices for slices
    time_indices = [1, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    # Row 1: PINN vs Exact at different times
    for i, ti in enumerate(time_indices):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.plot(x, u_pinn[ti], "b-", lw=2, label="PINN")
        ax.plot(x, u_exact[ti], "k--", lw=2, label="Exact")
        ax.plot(x, u_fdm[ti], "g:", lw=2, label="FDM")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"t = {t[ti]:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Errors comparison at different times
    for i, ti in enumerate(time_indices):
        ax = fig.add_subplot(4, 5, 5 + i + 1)
        ax.plot(x, e_true[ti], linestyle="-", color='black', lw=2, label="True Error")
        ax.plot(x, err_estimated[ti], linestyle="--", color='#E69F00', lw=1.5, label="Residual Est.")
        ax.plot(x, fdm_based_error[ti], linestyle=":", color='#0072B2', lw=2, label="FDM Est.")
        ax.set_xlabel("x")
        ax.set_ylabel("Error")
        ax.set_title(f"Errors at t = {t[ti]:.3f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Create meshgrid for heatmaps
    X, T = np.meshgrid(x, t)

    # Row 3: Solution heatmaps
    heatmap_data_row3 = [
        (u_pinn, "PINN Solution"),
        (u_exact, "Exact Solution"),
        (u_fdm, "FDM Solution"),
        (
            np.abs(err_estimated) - np.abs(fdm_based_error),
            "| Residual Est. | - |FDM Est.|\nIf > 0, Our method is worse.",
        ),  # Which error estimate is larger?
        (residual, "Residual"),
    ]

    for i, (data, title) in enumerate(heatmap_data_row3):
        ax = fig.add_subplot(4, 5, 10 + i + 1)
        vmax = np.max(np.abs(data)) if "-" in title else None
        vmin = -vmax if vmax else None
        cmap = "seismic" if "-" in title else "viridis"
        im = ax.pcolormesh(X, T, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    # Row 4: Error heatmaps
    heatmap_data_row4 = [
        (e_true, "True PINN Error"),
        (err_estimated, "Residual Error Est."),
        (fdm_based_error, "FDM Error Est."),
        (e_true - err_estimated, "True - Residual Est."),
        (e_true - fdm_based_error, "True - FDM Est."),
    ]

    for i, (data, title) in enumerate(heatmap_data_row4):
        ax = fig.add_subplot(4, 5, 15 + i + 1)
        vmax = np.max(np.abs(data))
        im = ax.pcolormesh(
            X, T, data, shading="auto", cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig.suptitle(
        f"{problem_name.upper()} - Error Estimation Results", fontsize=14, y=1.01
    )
    plt.tight_layout()

    return fig


def _plot_poisson_2d(results: Dict[str, Any]) -> plt.Figure:
    """Plot results for 2D Poisson problem."""
    x = results["x"]
    y = results["y"]
    u_pinn = results["u_pinn"]
    u_exact = results["u_true"]
    u_fdm = results["u_fdm"]
    err_estimated = results["e_res"]
    e_true = results["e_true"]

    ny, nx = len(y), len(x)

    # Ensure correct shape
    if u_pinn.shape != (ny, nx):
        u_pinn = u_pinn.reshape(ny, nx)
    if u_fdm.shape != (ny, nx):
        u_fdm = u_fdm.reshape(ny, nx)
    if err_estimated.shape != (ny, nx):
        err_estimated = err_estimated.reshape(ny, nx)
    if e_true.shape != (ny, nx):
        e_true = e_true.reshape(ny, nx)

    # Compute derived errors
    fdm_based_error = u_fdm - u_pinn
    residual_diff = e_true - err_estimated
    fdm_diff = e_true - fdm_based_error

    # Create figure: 4 rows x 5 cols
    fig = plt.figure(figsize=(24, 20))

    # Y indices for slices
    y_indices = [1, ny // 4, ny // 2, 3 * ny // 4, ny - 2]

    # Row 1: PINN vs Exact at different y slices
    for i, yi in enumerate(y_indices):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.plot(x, u_pinn[yi, :], "b-", lw=2, label="PINN")
        ax.plot(x, u_exact[yi, :], "k--", lw=2, label="Exact")
        ax.plot(x, u_fdm[yi, :], "g:", lw=2, label="FDM")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"y = {y[yi]:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Errors comparison at different y slices
    for i, yi in enumerate(y_indices):
        ax = fig.add_subplot(4, 5, 5 + i + 1)
        ax.plot(x, e_true[yi, :], "r-", lw=2, label="True Error")
        ax.plot(x, err_estimated[yi, :], "b--", lw=2, label="Residual Est.")
        ax.plot(x, fdm_based_error[yi, :], "g:", lw=2, label="FDM Est.")
        ax.set_xlabel("x")
        ax.set_ylabel("Error")
        ax.set_title(f"Errors at y = {y[yi]:.3f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Create meshgrid for heatmaps
    X, Y = np.meshgrid(x, y)

    # Row 3: Solution heatmaps
    heatmap_data_row3 = [
        (u_pinn, "PINN Solution"),
        (u_exact, "Exact Solution"),
        (u_fdm, "FDM Solution"),
        (
            np.abs(err_estimated) - np.abs(fdm_based_error),
            "| Residual Est. | - |FDM Est.|\nIf > 0, Our method is worse.",
        ),  # Which error estimate is larger?
        (
            np.abs(err_estimated) - np.abs(fdm_based_error),
            "same (no idea what to plot here)",
        ),
    ]

    for i, (data, title) in enumerate(heatmap_data_row3):
        ax = fig.add_subplot(4, 5, 10 + i + 1)
        vmax = np.max(np.abs(data)) if "-" in title else None
        vmin = -vmax if vmax else None
        cmap = "seismic" if "-" in title else "jet"
        im = ax.pcolormesh(X, Y, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    # Row 4: Error heatmaps
    heatmap_data_row4 = [
        (e_true, "True PINN Error"),
        (err_estimated, "Residual Error Est."),
        (fdm_based_error, "FDM Error Est."),
        (e_true - err_estimated, "True - Residual Est."),
        (e_true - fdm_based_error, "True - FDM Est."),
    ]

    for i, (data, title) in enumerate(heatmap_data_row4):
        ax = fig.add_subplot(4, 5, 15 + i + 1)
        vmax = np.max(np.abs(data))
        im = ax.pcolormesh(
            X, Y, data, shading="auto", cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    fig.suptitle("POISSON 2D - Error Estimation Results", fontsize=14, y=1.01)
    plt.tight_layout()

    return fig


def _plot_poisson_1d(results: Dict[str, Any]) -> plt.Figure:
    """Plot results for 1D Poisson problem (single slice, no time)."""
    x = results["x"]
    u_pinn = results["u_pinn"]
    u_exact = results["u_true"]
    u_fdm = results["u_fdm"]
    err_estimated = results["e_res"]
    e_true = results["e_true"]

    # Ensure 1D arrays
    u_pinn = np.asarray(u_pinn).flatten()
    u_exact = np.asarray(u_exact).flatten()
    u_fdm = np.asarray(u_fdm).flatten()
    err_estimated = np.asarray(err_estimated).flatten()
    e_true = np.asarray(e_true).flatten()

    # Compute derived errors
    fdm_based_error = u_fdm - u_pinn
    residual_diff = e_true - err_estimated
    fdm_diff = e_true - fdm_based_error

    # Create figure: 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Solutions
    # Plot 1: PINN vs Exact
    axes[0, 0].plot(x, u_pinn, "b-", lw=2, label="PINN")
    axes[0, 0].plot(x, u_exact, "k--", lw=2, label="Exact")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].set_title("Solution Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: FDM vs Exact
    axes[0, 1].plot(x, u_fdm, "g-", lw=2, label="FDM")
    axes[0, 1].plot(x, u_exact, "k--", lw=2, label="Exact")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].set_title("FDM Solution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: All solutions
    axes[0, 2].plot(x, u_pinn, "b-", lw=2, label="PINN")
    axes[0, 2].plot(x, u_fdm, "g--", lw=2, label="FDM")
    axes[0, 2].plot(x, u_exact, "k:", lw=2, label="Exact")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("u")
    axes[0, 2].set_title("All Solutions")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Errors
    # Plot 4: Error estimates comparison
    axes[1, 0].plot(x, e_true, "r-", lw=2, label="True Error")
    axes[1, 0].plot(x, err_estimated, "b--", lw=2, label="Residual Est.")
    axes[1, 0].plot(x, fdm_based_error, "g:", lw=2, label="FDM Est.")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].set_title("Error Estimates")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Error estimation quality
    axes[1, 1].plot(x, residual_diff, "b-", lw=2, label="True - Residual Est.")
    axes[1, 1].plot(x, fdm_diff, "g--", lw=2, label="True - FDM Est.")
    axes[1, 1].axhline(0, color="k", linestyle=":", alpha=0.5)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Difference")
    axes[1, 1].set_title("Uncertainty in Error Estimation")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Absolute errors comparison
    axes[1, 2].plot(x, np.abs(residual_diff), "b-", lw=2, label="|True - Residual|")
    axes[1, 2].plot(x, np.abs(fdm_diff), "g--", lw=2, label="|True - FDM|")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("|Error|")
    axes[1, 2].set_title("Absolute Uncertainty in Error Estimation")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("POISSON 1D - Error Estimation Results", fontsize=14)
    plt.tight_layout()

    return fig


def print_error_metrics(results: Dict[str, Any]):
    """Print error metrics for the experiment results."""
    e_true = results["e_true"]
    err_estimated = results["e_res"]
    u_pinn = results["u_pinn"]
    u_fdm = results["u_fdm"]

    # Flatten if needed
    e_true = np.asarray(e_true).flatten()
    err_estimated = np.asarray(err_estimated).flatten()
    u_pinn = np.asarray(u_pinn).flatten()
    u_fdm = np.asarray(u_fdm).flatten()

    fdm_based_error = u_fdm - u_pinn
    residual_diff = e_true - err_estimated
    fdm_diff = e_true - fdm_based_error

    print("=" * 60)
    print("ERROR METRICS")
    print("=" * 60)

    print(f"\nTrue PINN Error:")
    print(f"  L2:  {np.sqrt(np.mean(e_true**2)):.6e}")
    print(f"  Max: {np.max(np.abs(e_true)):.6e}")

    print(f"\nResidual Integration Method:")
    print(f"  Est. L2:  {np.sqrt(np.mean(err_estimated**2)):.6e}")
    print(f"  Est. Max: {np.max(np.abs(err_estimated)):.6e}")
    print(f"  Diff L2:  {np.sqrt(np.mean(residual_diff**2)):.6e}")
    print(f"  Diff Max: {np.max(np.abs(residual_diff)):.6e}")

    print(f"\nFDM-based Method:")
    print(f"  Est. L2:  {np.sqrt(np.mean(fdm_based_error**2)):.6e}")
    print(f"  Est. Max: {np.max(np.abs(fdm_based_error)):.6e}")
    print(f"  Diff L2:  {np.sqrt(np.mean(fdm_diff**2)):.6e}")
    print(f"  Diff Max: {np.max(np.abs(fdm_diff)):.6e}")

    # Which method is better?
    res_l2 = np.sqrt(np.mean(residual_diff**2))
    fdm_l2 = np.sqrt(np.mean(fdm_diff**2))

    print(
        f"\n{'Residual' if res_l2 < fdm_l2 else 'FDM'} method is better by factor: {max(res_l2, fdm_l2) / min(res_l2, fdm_l2):.2f}x"
    )
    print("=" * 60)
