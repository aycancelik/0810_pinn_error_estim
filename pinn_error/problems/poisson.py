"""Poisson equation problems (1D and 2D)"""

from typing import Union

import deepxde as dde
import numpy as np
import torch

from pinn_error.core.problem import BaseProblem, ProblemDomain


class Poisson1D(BaseProblem):
    """1D Heat equation problem definition"""

    def __init__(
        self,
        x_min: float,
        x_max: float,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = None
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)

    def source_term(
        self, x: Union[torch.Tensor, np.ndarray, float]
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """Source term f(x) = (pi / L)^2 * sin(pi * x / L)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 1) where column is x.

        Returns
        -------
        torch.Tensor
            Source term values at x.
        """
        if isinstance(x, torch.Tensor):
            return (torch.pi / self.domain.x_max) ** 2 * torch.sin(
                torch.pi * x / self.domain.x_max
            )
        else:
            return (np.pi / self.domain.x_max) ** 2 * np.sin(
                np.pi * x / self.domain.x_max
            )

    def pde(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Defines the PDE for the 1D Poisson equation.
        -u'' - f(x) = 0

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 1) where column is x.
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x).

        Returns
        -------
        torch.Tensor
            The residual of the PDE.
        """
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        f_x = self.source_term(x[:, 0:1])
        return -u_xx - f_x

    def output_transform(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Hard constraint for Dirichlet BCs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 1) where column is x.
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x).

        Returns
        -------
        torch.Tensor
            Transformed output tensor satisfying BCs/ICs.
        """
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        _x = x[:, 0:1]

        # is zero at boundaries
        return (_x - x_min) * (x_max - _x) * u

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        """Returns the exact solution for given x.

        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates.

        Returns
        -------
        np.ndarray
            Exact solution values at x.
        """
        return np.sin(np.pi * x / self.domain.x_max)

    def initial_condition(self, *args) -> None:
        """No initial condition for steady-state Poisson equation."""
        return None


class Poisson2D(BaseProblem):
    """2D Poisson equation problem definition

    -u_xx - u_yy = f(x, y) on [x_min, x_max] x [y_min, y_max]
    with zero Dirichlet BCs on all boundaries.

    Exact solution: u(x,y) = sin(pi*x/Lx) * sin(pi*y/Ly)
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):
        self.spatial_bounds = (x_min, x_max, y_min, y_max)
        self.temporal_bounds = None
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)

    def source_term(
        self,
        x: Union[torch.Tensor, np.ndarray, float],
        y: Union[torch.Tensor, np.ndarray, float],
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """Source term f(x,y) = (pi^2/Lx^2 + pi^2/Ly^2) * sin(pi*x/Lx) * sin(pi*y/Ly)"""
        Lx = self.domain.x_max - self.domain.x_min
        Ly = self.domain.y_max - self.domain.y_min
        if isinstance(x, torch.Tensor):
            pi = torch.pi
            coeff = (pi / Lx) ** 2 + (pi / Ly) ** 2
            return (
                coeff
                * torch.sin(pi * (x - self.domain.x_min) / Lx)
                * torch.sin(pi * (y - self.domain.y_min) / Ly)
            )
        else:
            pi = np.pi
            coeff = (pi / Lx) ** 2 + (pi / Ly) ** 2
            return (
                coeff
                * np.sin(pi * (x - self.domain.x_min) / Lx)
                * np.sin(pi * (y - self.domain.y_min) / Ly)
            )

    def pde(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Defines the PDE: -u_xx - u_yy - f(x,y) = 0"""
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f_xy = self.source_term(x[:, 0:1], x[:, 1:2])
        return -u_xx - u_yy - f_xy

    def output_transform(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Hard constraint for zero Dirichlet BCs on all 4 edges."""
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        y_min = self.domain.y_min
        y_max = self.domain.y_max
        _x = x[:, 0:1]
        _y = x[:, 1:2]
        return (_x - x_min) * (x_max - _x) * (_y - y_min) * (y_max - _y) * u

    def exact_solution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns the exact solution u(x,y) = sin(pi*(x-x_min)/Lx) * sin(pi*(y-y_min)/Ly)"""
        Lx = self.domain.x_max - self.domain.x_min
        Ly = self.domain.y_max - self.domain.y_min
        return np.sin(np.pi * (x - self.domain.x_min) / Lx) * np.sin(
            np.pi * (y - self.domain.y_min) / Ly
        )

    def initial_condition(self, *args) -> None:
        """No initial condition for steady-state Poisson equation."""
        return None
