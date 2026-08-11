"""1D Heat equation problem"""

import deepxde as dde
import numpy as np
import torch

from pinn_error.core.problem import BaseProblem, ProblemDomain


class Heat1DProblemSineIC(BaseProblem):
    """1D Heat equation problem definition"""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        t_max: float,
        diffusivity: float,
        frequency: int = 1,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = (0.0, t_max)
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)
        self.diffusivity = diffusivity
        self.frequency = frequency

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
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - self.diffusivity * u_xx

    def output_transform(self, x, u) -> torch.Tensor:
        """Hard constraint for initial and boundary conditions with sinusoidal IC.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 2) where columns are (x, t).
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x,t).

        Returns
        -------
        torch.Tensor
            Transformed output tensor satisfying BCs/ICs.
        """
        n = self.frequency
        _x = x[:, 0:1]
        _t = x[:, 1:2]
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        return (
            torch.sin(n * torch.pi * _x / x_max) + _t * (_x - x_min) * (x_max - _x) * u
        )

    def output_transform_bc_only(self, x, u):
        """
        Hard constraint for boundary conditions only, leaving IC as soft constraint.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 2) where columns are (x, t).
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x,t).

        Returns
        -------
        torch.Tensor
            Transformed output tensor satisfying BCs only.
        """
        _x = x[:, 0:1]
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        return (_x - x_min) * (x_max - _x) * u
    
    def initial_condition(self, x) -> torch.Tensor | np.ndarray:
        """Initial condition u(x,0) = sin(n * pi * x / L)

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Spatial coordinates.

        Returns
        -------
        torch.Tensor or np.ndarray
            Initial condition values at t=0.
        """
        L = self.domain.x_max
        n = self.frequency
        if isinstance(x, torch.Tensor):
            return torch.sin(n * torch.pi * x / L)
        else:
            # if x has both space and time
            if len(x.shape) > 1 and x.shape[1] > 1:
                x = x[:, 0:1]  # extract x coord
            values = np.sin(n * np.pi * x / L)
            return np.asarray(values).reshape(-1)


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
        L = self.domain.x_max
        n = self.frequency
        alpha = self.diffusivity
        return np.exp(-(n**2 * np.pi**2 * alpha * t) / (L**2)) * np.sin(
            n * np.pi * x / L
        )
