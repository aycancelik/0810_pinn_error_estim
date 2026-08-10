"""1D Heat equation problem"""

import deepxde as dde
import numpy as np
import torch

from pinn_error.core.problem import BaseProblem, ProblemDomain


class Wave1D(BaseProblem):
    """1D Wave equation problem definition"""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        t_max: float,
        propagation_speed: float = 2.0,
        frequency: int = 1,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = (0.0, t_max)
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)

        self.propagation_speed = propagation_speed
        self.frequency = frequency

    def pde(self, x, u) -> torch.Tensor:
        """Defines the PDE for the 1D wave equation.

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
        u_tt = dde.grad.hessian(u, x, i=1, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_tt - (self.propagation_speed**2) * u_xx

    def exact_solution(self, x, t) -> np.ndarray:
        """Exact solution for general propagation speed"""
        L = self.domain.x_max
        n = self.frequency
        c = self.propagation_speed

        # Single mode solution: u(x,t) = sin(nπx/L) * cos(cnπt/L)
        return np.sin(n * np.pi * x / L) * np.cos(c * n * np.pi * t / L)

    def initial_condition(self, x) -> torch.Tensor | np.ndarray:
        """Simplified IC: u(x,0) = sin(nπx/L)"""
        L = self.domain.x_max
        n = self.frequency
        if isinstance(x, torch.Tensor):
            return torch.sin(n * torch.pi * x / L)
        else:
            return np.sin(n * np.pi * x / L)

    def output_transform(self, x, u) -> torch.Tensor:
        """Hard constraint for zero BCs and simple sinusoidal IC"""
        n = self.frequency
        _x = x[:, 0:1]
        _t = x[:, 1:2]
        x_min = self.domain.x_min
        x_max = self.domain.x_max

        return u * _t**2 * (_x - x_min) * (x_max - _x) + torch.sin(
            n * torch.pi * _x / x_max
        )

    # def output_transform(self, x, u) -> torch.Tensor:
    #     """Hard constraint for initial and boundary conditions with sinusoidal IC.

    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Input tensor with shape (N, 2) where columns are (x, t).
    #     u : torch.Tensor
    #         Output tensor with shape (N, 1) representing u(x,t).

    #     Returns
    #     -------
    #     torch.Tensor
    #         Transformed output tensor satisfying BCs/ICs.
    #     """
    #     n = self.mode

    #     _x = x[:, 0:1]
    #     _t = x[:, 1:2]

    #     x_min = self.domain.x_min
    #     x_max = self.domain.x_max

    #     # not sure if this accurately represents any other
    #     # modes than n=1
    #     # and propagation speeds != 2.0
    #     return (
    #         20 * u * _t * (_x - x_min) * (x_max - _x)
    #         + torch.sin(n * torch.pi * _x / x_max)
    #         + 0.5 * torch.sin(self.propagation_speed**2 * n * torch.pi * _x / x_max)
    #     )

    # def initial_condition(self, x) -> torch.Tensor | np.ndarray:
    #     """Initial condition u(x,0) = sin(n * pi * x / L)

    #     Parameters
    #     ----------
    #     x : torch.Tensor or np.ndarray
    #         Spatial coordinates.

    #     Returns
    #     -------
    #     torch.Tensor or np.ndarray
    #         Initial condition values at t=0.
    #     """
    #     L = self.domain.x_max
    #     n = self.mode
    #     if isinstance(x, torch.Tensor):
    #         return torch.sin(n * torch.pi * x / L) + 0.5 * torch.sin(self.propagation_speed**2 * n * torch.pi * x / L)
    #     else:
    #         return np.sin(n * np.pi * x / L) + 0.5 * np.sin(self.propagation_speed**2 * n * np.pi * x / L)

    # def exact_solution(self, x, t) -> np.ndarray:
    #     """Returns the exact solution for given x and t.

    #     Parameters
    #     ----------
    #     x : np.ndarray
    #         Spatial coordinates.
    #     t : np.ndarray
    #         Temporal coordinates.

    #     Returns
    #     -------
    #     np.ndarray
    #         Exact solution values at (x, t).
    #     """
    #     L = self.domain.x_max
    #     n = self.mode
    #     propagation_speed = self.propagation_speed

    #     return (
    #         np.sin(n * np.pi * x / L) * np.cos(propagation_speed * n * np.pi * t)
    #         + 0.5 * np.sin(propagation_speed**2 * n * np.pi * x / L)
    #         * np.cos(2 * propagation_speed**2 * n * np.pi * t)
    #     )
