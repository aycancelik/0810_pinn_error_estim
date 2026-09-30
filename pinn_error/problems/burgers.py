import deepxde as dxe
import numpy as np
import torch
from pinn_error.core.problem import BaseProblem, ProblemDomain
from scipy.special import logsumexp


class Burgers1D(BaseProblem):
    """1D Burgrs' equation"""

    def __init__(
            self,
            x_min: float,
            x_max: float,
            t_max: float,
            viscosity: float = 0.01 / np.pi,
    ):
        self.spatial_bounds = (x_min, x_max)
        self.temporal_bounds = (0.0, t_max)
        domain = ProblemDomain(
            spatial_bounds=self.spatial_bounds, temporal_bounds=self.temporal_bounds
        )
        super().__init__(domain)

        self.viscosity = viscosity


    def pde(self, x, u) -> torch.Tensor:
        """Defines the PDE for the 1D Burgers' equation.

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
        u_t = dxe.grad.jacobian(u, x, i=0, j=1)
        u_x = dxe.grad.jacobian(u, x, i=0, j=0)
        u_xx = dxe.grad.hessian(u, x, i=0, j=0)
        return u_t + u * u_x - self.viscosity * u_xx


    def initial_condition(self, x) -> torch.Tensor | np.ndarray:
            """Initial condition u(x,0) = -sin(pi*x)
            
            Parameters
            ----------
            x: torch.Tensor or np.ndarray
                Spatial coordinates
    
            Returns
            -------
            torch.Tensor or np.ndarray
                Initial condition values at t=0.
            """
    
            return -torch.sin(np.pi * x) if isinstance(x, torch.Tensor) else -np.sin(np.pi * x)
            

    def output_transform(self, x, u) -> torch.Tensor:
        """Hard constraint for initial and boundary conditions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 2) where columns are (x, t).
        u : torch.Tensor
            Output tensor with shape (N, 1) representing u(x,t).

        Returns
        -------
        torch.Tensor
            The transformed output tensor satisfying the initial and boundary conditions.
        """
        _x = x[:, 0:1]
        _t = x[:, 1:2]
        x_min = self.domain.x_min
        x_max = self.domain.x_max
        return -torch.sin(np.pi * _x) + _t * (_x - x_min) * (x_max - _x) * u


    def exact_solution(self, x, t, n_quad: int = 4001, eta_max: float = 15.0,
                    chunk: int = 2000):
        """Hopf-Cole exact solution for u0 = -sin(pi x) on [-1, 1]."""
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()
            t = t.detach().cpu().numpy()

        nu = self.viscosity
        x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
        shape = x.shape
        x, t = x.ravel(), t.ravel()

        u = -np.sin(np.pi * x)                       # t = 0: initial condition
        eta = np.linspace(-eta_max, eta_max, n_quad)
        idx = np.flatnonzero(t > 0)

        for s in range(0, idx.size, chunk):          # chunk to limit memory
            i = idx[s:s + chunk]
            c = np.sqrt(4 * nu * t[i])[:, None]
            y = x[i][:, None] - c * eta[None, :]
            logw = -eta**2 - np.cos(np.pi * y) / (2 * np.pi * nu)
            w = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # softmax
            u[i] = -np.sum(w * np.sin(np.pi * y), axis=1)

        u = u.reshape(shape)
        return torch.as_tensor(u, dtype=dtype, device=device) if is_torch else u

         


    

    
    
