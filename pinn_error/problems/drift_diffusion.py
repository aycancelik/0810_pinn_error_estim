"""1D Drift Diffusion equation problem"""

import deepxde as dde
import numpy as np
import torch

from pinn_error.core.problem import BaseProblem, ProblemDomain


class DriftDiffusion(BaseProblem):
    """1D Drift Diffusion equation problem definition"""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        t_max: float,
        initial_concentration: float = 1.0,
        frequency: float = 2.0,
        phase_shift: float = np.pi / 4,
        diffusivity: float = 1.0,
        velocity_x: float = 20.0,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = (0.0, t_max)
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)

        self.initial_concentration = initial_concentration
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.velocity_x = velocity_x
        self.diffusivity = diffusivity

    def pde(self, x, u) -> torch.Tensor:
        """Defines the PDE for the 1D heat equation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 2) where columns are (x, t).
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x,t).

        Returns
        -------
        torch.Tensor
            The residual of the PDE.
        """
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - self.diffusivity * u_xx + self.velocity_x * u_x

    def output_transform(self, x, u) -> torch.Tensor:
        """Hard constraint for initial and boundary conditions.

            Structure:
            u(x,t) = IC(x) + spatial_BC_correction(x,t) + vanishing_factor(x,t) * NN(x,t)

            This ensures:

            - u(x, 0) = IC(x)           [initial condition]
            - u(x_min, t) = g_min(t)    [left boundary from exact solution]

            - u(x_max, t) = g_max(t)    [right boundary from exact solution]

        #### Parameters
            x : torch.Tensor
                Input tensor with shape (N, 2) where columns are (x, t).
            u : torch.Tensor
                Raw NN output with shape (N, 1).

        #### Returns
            torch.Tensor
                Transformed output satisfying BCs/ICs.
        """
        n = self.frequency
        A = self.initial_concentration
        phi = self.phase_shift
        D = self.diffusivity
        beta = self.velocity_x

        _x = x[:, 0:1]
        _t = x[:, 1:2]

        x_min = torch.tensor(self.domain.x_min)
        x_max = torch.tensor(self.domain.x_max)
        L = x_max - x_min  # domain width

        # Wavenumber (consistent with initial_condition)
        k = n * np.pi / L

        # --- Initial condition terms ---
        # IC at current x: u(x, 0) = A * sin(k * x + φ)
        # Note: k * x = n * π * x / L = n * π * x / x_max (when x_min = 0)
        ic_at_x = A * torch.sin(k * _x + phi)

        # IC evaluated at boundaries
        ic_at_x_min = A * torch.sin(k * x_min + phi)
        ic_at_x_max = A * torch.sin(k * x_max + phi)

        # --- Boundary conditions from exact solution ---
        # g(x, t) = A * sin(φ + k * (x - β * t)) * exp(-D * k² * t)
        decay = torch.exp(-D * k**2 * _t)

        g_min_t = A * torch.sin(phi + k * (x_min - beta * _t)) * decay
        g_max_t = A * torch.sin(phi + k * (x_max - beta * _t)) * decay

        # --- Spatial interpolation for BC correction ---
        # Linear weights for smooth transition between boundaries
        weight_min = (x_max - _x) / L  # = 1 at x_min, = 0 at x_max
        weight_max = (_x - x_min) / L  # = 0 at x_min, = 1 at x_max

        # BC correction: difference between time-dependent BC and IC at boundaries
        spatial_bc_correction = weight_min * (g_min_t - ic_at_x_min) + weight_max * (
            g_max_t - ic_at_x_max
        )

        # --- NN influence factor ---
        # This factor is zero at:
        # - t = 0 (ensures IC is satisfied)
        # - x = x_min (ensures left BC is satisfied)
        # - x = x_max (ensures right BC is satisfied)
        nn_factor = _t * (_x - x_min) * (x_max - _x)

        # --- Final transformed output ---
        return ic_at_x + spatial_bc_correction + nn_factor * u

    def initial_condition(self, x) -> torch.Tensor | np.ndarray:
        """Initial condition u(x,0) = initial_concentration * sin(frequency * pi * x + phase_shift)

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Spatial coordinates.

        Returns
        -------
        torch.Tensor or np.ndarray
            Initial condition values at t=0.
        """
        n = self.frequency
        A = self.initial_concentration
        phi = self.phase_shift
        L = self.domain.x_max - self.domain.x_min
        k = n * np.pi / L

        if isinstance(x, torch.Tensor):
            return A * torch.sin(k * x + phi)
        else:
            return A * np.sin(k * x + phi)

    def exact_solution(self, x, t) -> np.ndarray:
        """Returns the exact solution for given x and t.

        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates.
        t : np.ndarray
            Temporal coordinates.

        Returns
        -------
        np.ndarray
            Exact solution values at (x, t).
        """
        n = self.frequency
        A = self.initial_concentration
        phi = self.phase_shift
        beta = self.velocity_x
        D = self.diffusivity
        L = self.domain.x_max - self.domain.x_min
        k = n * np.pi / L

        return A * np.sin(phi + k * (x - beta * t)) * np.exp(-D * k**2 * t)
