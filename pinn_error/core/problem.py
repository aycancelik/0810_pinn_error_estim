"""Problem abstraction module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import deepxde as dde
import numpy as np
import torch


@dataclass
class ProblemDomain:
    """Defines spatial and temporal domain for the problem"""

    # spatial bounds (for 1D: (x_min, x_max) for 2D: (x_min, x_max, y_min, y_max))
    spatial_bounds: Tuple[float, ...]

    # temporal bounds (t_min, t_max)
    temporal_bounds: Optional[Tuple[float, float]]

    @property
    def spatial_dim(self) -> int:
        """Returns the spatial dimension based on the length of spatial_bounds"""
        return len(self.spatial_bounds) // 2

    @property
    def is_time_dependent(self) -> bool:
        """Checks if the problem is time-dependent based on temporal_bounds"""
        return self.temporal_bounds is not None

    @property
    def x_min(self) -> float:
        """Returns the minimum spatial bound"""
        return self.spatial_bounds[0]

    @property
    def x_max(self) -> float:
        """Returns the maximum spatial bound"""
        return self.spatial_bounds[1]

    @property
    def y_min(self) -> Optional[float]:
        """Returns the minimum y spatial bound if 2D"""
        if self.spatial_dim >= 2:
            return self.spatial_bounds[2]
        return None

    @property
    def y_max(self) -> Optional[float]:
        """Returns the maximum y spatial bound if 2D"""
        if self.spatial_dim >= 2:
            return self.spatial_bounds[3]
        return None

    @property
    def t_min(self) -> Optional[float]:
        """Returns the minimum temporal bound if time-dependent"""
        if self.temporal_bounds:
            return self.temporal_bounds[0]
        return None

    @property
    def t_max(self) -> Optional[float]:
        """Returns the maximum temporal bound if time-dependent"""
        if self.temporal_bounds:
            return self.temporal_bounds[1]
        return None


class BaseProblem(ABC):
    """Abstract base class for defining a PINN problem"""

    # Whether the PDE is second order in time (e.g. the wave equation). Such
    # problems need TWO initial conditions to be well posed: the value
    # u(x,0) AND the velocity du/dt(x,0). Subclasses that are second order in
    # time must set this to True and implement `initial_velocity`, so that
    # soft-constrained training adds a loss term for the velocity IC and the
    # FDM error integration accounts for the initial error velocity.
    is_second_order_in_time: bool = False

    def __init__(self, domain: ProblemDomain):
        self.domain = domain

    def initial_velocity(
        self, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Returns the initial velocity du/dt(x,0) for second-order-in-time PDEs.

        Not abstract: only meaningful when `is_second_order_in_time` is True
        (first-order-in-time problems like the heat equation are fully
        determined by u(x,0) alone).

        Args:
            x: Spatial coordinates tensor or array

        Returns:
            Tensor or array of initial velocity values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement initial_velocity() "
            "(only needed when is_second_order_in_time is True)."
        )

    @abstractmethod
    def pde(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Defines the PDE residual

        Args:
            x: Input tensor (spatial and temporal coordinates)
            u: Output tensor (PINN prediction)

        Returns:
            Tensor representing the PDE residual D[u](x)
        """
        pass

    @abstractmethod
    def initial_condition(
        self, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Returns the initial condition values at given spatial coordinates (for time t=0)

        Args:
            x: Spatial coordinates tensor or array

        Returns:
            Tensor or array of initial condition values
        """
        pass

    def boundary_condition(
        self, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Returns the boundary condition values at given spatial coordinates

        Not abstract: no subclass currently implements this (FDM boundary
        error/BC targets are derived from `exact_solution` instead, see
        FDMSolverDriftDiffusion._get_boundary_values for precedent). Left as
        an optional override for problems that want to define BC data
        independently of the exact solution.

        Args:
            x: Spatial coordinates tensor or array

        Returns:
            Tensor or array of boundary condition values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement boundary_condition()."
        )

    @abstractmethod
    def exact_solution(self, *args) -> Union[float, np.ndarray]:
        """Returns the exact solution for validation if available"""
        pass

    @abstractmethod
    def output_transform(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Applies output transformation for hard constraints

        Args:
            x: Input tensor
            u: Raw output from the neural network

        Returns:
            Transformed output tensor (such that BCs/ICs are satisfied)
        """
        pass

    def output_transform_bc_only(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Applies output transformation for hard constraints on boundary conditions only

        Not abstract: only needed for `constraint_mode="soft_ic"` (see
        PINNConfig/PINNTrainer). Time-independent problems (Poisson) have no
        initial condition to soften and don't need this; time-dependent
        problems that want to support the soft-IC experiment should override
        it (see Heat1DProblemSineIC for an example).

        Args:
                x: Input tensor
                u: Raw output from the neural network
        Returns:
                Transformed output tensor (such that BCs are satisfied, ICs are soft)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement output_transform_bc_only() "
            "(needed for constraint_mode='soft_ic')."
        )

    def create_geometry(self) -> dde.geometry.Geometry:
        """Creates the geometry for the problem based on the domain"""
        if self.domain.spatial_dim == 1:
            geom = dde.geometry.Interval(self.domain.x_min, self.domain.x_max)
        elif self.domain.spatial_dim == 2:
            geom = dde.geometry.Rectangle(
                [self.domain.x_min, self.domain.y_min],
                [self.domain.x_max, self.domain.y_max],
            )
        else:
            raise NotImplementedError("Only 1D and 2D spatial domains are supported.")

        if self.domain.is_time_dependent:
            timedomain = dde.geometry.TimeDomain(self.domain.t_min, self.domain.t_max)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            return geomtime
        return geom

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(domain={self.domain})"