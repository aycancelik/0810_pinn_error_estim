import deepxde as dde
import numpy as np
import torch
from pinn_error.core.problem import BaseProblem, ProblemDomain


class AllenCahn():

    def __init__(
            self,
            x_min: float,
            x_max: float,
            t_max:float,
            d: float = 0.001,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = (0.0, t_max)
        self.d = d
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super.__init__(domain)

    def pde(self, x, u) -> torch.Tensor:
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - self.d * u_xx - 5*(u - u**3)

    def initial_condition(self, x) -> torch.Tensor | np.array:
        return x**2 * torch.cos(torch.pi * x) if isinstance(x, torch.Tensor) else x**2 * np.cos(np.pi * x)

    def output_transform(self, x, u) -> torch.Tensor: 
        return x[:, 0:1]**2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * u

    def exact_solution(self, x, u) -> torch.Tensor:
        "An approximation to the exact solution based on the FDM method"
        return None