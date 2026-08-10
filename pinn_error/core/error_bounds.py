"""
Semigroup-based a posteriori error bounds for PINNs (heat equation).

Reimplementation of:
    B. Hillebrecht and B. Unger, "Certified machine learning: Rigorous a posteriori
    error bounds for PDE defined PINNs", IEEE TNNLS, 2023.
    arXiv: 2210.03426

Original code: https://github.com/bhillebrecht/CertifiedML_PDE

Hard-coded for the 1D heat equation with hard-constrained BCs and ICs,
as featured in our work.

For the heat equation du/dt = alpha * d^2u/dx^2 on [0, L] with semigroup S(t)
satisfying ||S(t)|| <= M * exp(omega * t), the PINN error e = u_hat - u satisfies:

    ||e(t)||_{L^2} <= M * exp(omega*t) * zeta_0
                    + integral_0^t M * exp(omega*(t-s)) * zeta(s) ds

where:
    - zeta_0 corresponds to the initial condition error (0 for hard IC constraints)
        - zeta_0 = ||u_hat(0) - u(0)||_{L^2} + integration error 
    - zeta(s) corresponds to the PDE residual norm at time s, with some smoothing:
        - zeta(s) = sqrt(||R(s, .)||_{L^2}^2 + mu^2)
    - M, omega: semigroup growth parameters (||S(t)|| <= M * exp(omega * t))
"""

import time

import numpy as np
import torch
from tqdm import tqdm 

from pinn_error.core.pinn import PINNTrainer
from pinn_error.core.problem import BaseProblem, ProblemDomain


def heat_equation_semigroup_params(
    diffusivity: float,
    domain_length: float,
    mode: str = "exponential",
) -> tuple[float, float]:
    """
    Semigroup parameters for the heat equation: du/dt = alpha * d^2u/dx^2
    on [0, L] with homogeneous Dirichlet BCs.

    Parameters
    ----------
    diffusivity : float
        Thermal diffusivity alpha.
    domain_length : float
        Domain length L.
    mode : str
        'contraction' for M=1, omega=0;
        'exponential' for M=1, omega=-alpha * pi^2 / L^2.

    Returns
    -------
    M : float
        Semigroup scaling constant.
    omega : float
        Semigroup growth/decay rate.
    """
    M = 1.0
    if mode == "contraction":
        omega = 0.0
    elif mode == "exponential":
        omega = -diffusivity * np.pi**2 / domain_length**2
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'contraction' or 'exponential'.")
    return M, omega


class PINNErrorBoundEstimator:
    def __init__(
        self,
        nx: int,
        problem: BaseProblem,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
        semigroup_mode: str = "exponential",
        mu_factor: float = 0.1,
        epsilon: float = 0.33,
        use_trapezoid_norm: bool = False,
    ):
        """
        Initialize the Hillebrecht-Unger semigroup error bound estimator.

        Parameters
        ----------
        nx : int
            Number of spatial grid points for L^2 norm quadrature.
        problem : BaseProblem
            The heat equation problem definition (must have a `diffusivity`
            attribute).
        domain : ProblemDomain
            The problem domain.
        pinn_model : PINNTrainer
            The trained PINN model.
        semigroup_mode : str
            'exponential' uses decay rate omega = -alpha * pi^2 / L^2 for a
            tighter bound; 'contraction' uses omega = 0 for a more conservative
            but simpler bound.
        mu_factor : float
            Smoothing parameter as fraction of delta_mean.  The original code
            uses 0.1 (the CLI default for ``--mu_factor``).
        epsilon : float
            Fraction of expected ML error allowed for integration error.  The
            original code uses 0.33 (the CLI default for ``--epsilon``).
        use_trapezoid_norm : bool
            Whether to use trapezoidal rule for spatial L^2 norm (more accurate)
            or simple Riemann sum. The original code uses trapezoidal rule
            but this is much more expensive. 
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model
        self.semigroup_mode = semigroup_mode
        self.mu_factor = mu_factor
        self.epsilon = epsilon
        
        # set norm method
        if use_trapezoid_norm:
            self._compute_spatial_l2_norm = self._compute_spatial_l2_norm_trapezoid
        else:
            self._compute_spatial_l2_norm = self._compute_spatial_l2_norm_discrete

        self.nx = nx
        self.dx = (domain.spatial_bounds[1] - domain.spatial_bounds[0]) / (nx - 1)
        self.x = np.linspace(
            domain.spatial_bounds[0], domain.spatial_bounds[1], nx
        )

        domain_length = domain.spatial_bounds[1] - domain.spatial_bounds[0]
        self.M, self.omega = heat_equation_semigroup_params(
            problem.diffusivity, domain_length, mode=semigroup_mode
        )

        # Parameters populated by _extract_parameters
        self.K = None
        self.mu = None
        self.delta_mean = None
        u_0 = problem.exact_solution(self.x, np.zeros_like(self.x))
        u_0_pinn = pinn_model.predict(np.column_stack([self.x, np.zeros_like(self.x)])).flatten()
        r0, _, err = self._compute_spatial_l2_norm(u_0 - u_0_pinn)
        self.r0 = r0 + err  # initial condition error with "spatial integration error"

        self._run_time = 0.0


    def _compute_spatial_l2_norm_discrete(
            self, R: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute ||R||_{L^2} = sqrt(integral R^2 dx) via discrete sum on the
        spatial grid, with Richardson error estimate.
        # TODO: the 1/3 factor seems arbitrary
        # I believe 1/3 is more standard
        
        Following the original code (helpers/l2_norm.py + helpers/integrate.py):
          I1 = sum(R^2 * dx)          on fine grid
          I2 = sum(R^2[::2] * 2*dx)   on coarse grid
          E  = 1/3 * |I1 - I2|        error estimate
          return sqrt(I1), sqrt(I2), sqrt(E) 

        Parameters
        ----------
        R : np.ndarray, shape (nx,)
            Residual values on the spatial grid.

        Returns
        -------
        norm_fine : float
            Fine grid norm: sqrt(I1).
        norm_coarse : float
            Coarse grid norm: sqrt(I2).
        error_estimate : float
            Error estimate: sqrt(E).
        """
        norm_fine = np.sqrt(np.sum(R**2) * self.dx)
        norm_coarse = np.sqrt(np.sum(R[::2]**2) * 2 * self.dx)
        # error_estimate = np.sqrt(1/3 * abs(norm_fine**2 - norm_coarse**2))
        error_estimate = 0 
        return norm_fine, norm_coarse, error_estimate

    def _compute_spatial_l2_norm_trapezoid(
        self, R: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute ||R||_{L^2} = sqrt(integral R^2 dx) via composite trapezoidal
        rule on the spatial grid, with Richardson error estimate.

        Following the original code (helpers/l2_norm.py + helpers/integrate.py):
          I1 = trapz(R^2, dx)          on fine grid
          I2 = trapz(R^2[::2], 2*dx)   on coarse grid
          E  = 1/3 * |I1 - I2|
          return sqrt(I1), sqrt(I2), sqrt(E)     as conservative upper bound on the norm

        Parameters
        ----------
        R : np.ndarray, shape (nx,)
            Residual values on the spatial grid.

        Returns
        -------
        norm_fine : float
            Fine grid norm: sqrt(I1).
        norm_coarse : float
            Coarse grid norm: sqrt(I2).
        error_estimate : float
            Error estimate: sqrt(E).
        """
        R_sq = R**2
        norm_fine = np.trapezoid(R_sq, self.x)
        norm_coarse = np.trapezoid(R_sq[::2], self.x[::2])
        # error_estimate = 1/3 * abs(norm_fine - norm_coarse)
        error_estimate = 0
        return np.sqrt(max(norm_fine, 0.0)), np.sqrt(max(norm_coarse, 0.0)), np.sqrt(max(error_estimate, 0.0))

    # ------------------------------------------------------------------
    # EXTRACT phase: compute parameters (K, mu, delta_mean)
    # ------------------------------------------------------------------

    def _extract_parameters(self, n_collocation: int = 10000) -> None:
        """
        Extract key parameters needed for the error bound, mirroring the
        original ``extract`` CLI command (helpers/extract.py).

        Computes:
        - delta_mean: RMS of the PDE residual, scaled by domain length for
          domain-wise mode (approximates spatial integral contribution).
        - mu: smoothing parameter = mu_factor * delta_mean.
        - K: sup|d^2/dt^2(exp(-omega*t) * sqrt(R^2 + mu^2))| estimated via
          finite differences on collocation points sorted by time.

        Parameters
        ----------
        n_collocation : int
            Number of collocation points for parameter estimation.
        """
        # generate random collocation points in the domain
        # note that boundaries are hard constrained
        # hence we're only interested in interior points
        X = self.pinn_model.geom.random_points(n_collocation)
        X = torch.tensor(X, requires_grad=True)

        # mean residual 
        R = self.pinn_model.residual(X).flatten()
        R_sq = R**2
        delta_mean = torch.sqrt(torch.mean(R_sq))

        # scale by domain length
        domain_length = (
            self.domain.spatial_bounds[1] - self.domain.spatial_bounds[0]
        )
        delta_mean *= domain_length
        self.delta_mean = delta_mean.item()

        # smoothing parameter mu (for smoothing zeta(s) = sqrt(||R||^2 + mu^2))
        self.mu = self.mu_factor * self.delta_mean

        # estimate K = sup|d^2/dt^2(exp(-omega*t) * sqrt(R^2 + mu^2))| via autograd
        # sort collocation points by time
        t = X[:, 1]
        zeta = torch.sqrt(R_sq + self.mu**2)
        f = torch.exp(-self.omega * t) * zeta
        f_t = torch.autograd.grad(f, X, grad_outputs=torch.ones_like(f), create_graph=True)[0][:, 1]
        f_tt = torch.autograd.grad(f_t, X, grad_outputs=torch.ones_like(f_t), create_graph=True)[0][:, 1]
        self.K = torch.max(torch.abs(f_tt)).item()

        
    @staticmethod
    def _compute_expected_ml_error(
        t: float, 
        omega: float, 
        M: float, 
        r0: float, 
        delta_mean: float
    ) -> float:
        """
        A priori estimate of the ML error at time t (run.py:54-66).

        Used only for scaling the acceptable integration error when choosing
        the number of quadrature points.

        Parameters
        ----------
        t : float
            Target time.
        omega : float
            Semigroup growth/decay rate.
        M : float
            Semigroup scaling constant.
        r0 : float
            Initial condition error (0 for hard IC).
        delta_mean : float
            Mean PDE residual scaled by domain length.
        """
        if omega == 0:
            return M * (r0 + delta_mean * t)
        else:
            return M * np.exp(omega * t) * (
                r0 + (1 - np.exp(-omega * t)) * delta_mean / omega
            )


    def _compute_n_support_points(
            self, 
            t: float
        ) -> int:
        """
        Determine number of time quadrature subintervals (run.py:42-52).

        N_SP = ceil(sqrt(K * t^3 / (12 * E_ML * epsilon)))

        The exp(omega*t) factor from the paper's formula is absorbed into
        E_ML (which already contains M * exp(omega*t) * ...).

        Parameters
        ----------
        t : float
            Target time.

        Returns
        -------
        n_sp : int
            Number of quadrature subintervals to use for time integration.
        """
        if t <= 0 or self.K <= 0:
            print(f"Warning: t={t}: non-positive time or K (K={self.K}). Using minimum N_SP=10.")
            return 10  # minimum

        E_ML = self._compute_expected_ml_error(
            t, self.omega, self.M, self.r0, self.delta_mean
        )
        if E_ML <= 0:
            print(f"Warning: t={t}: non-positive expected ML error (E_ML={E_ML}). Using minimum N_SP=10.")
            return 10

        n = int(np.ceil(np.sqrt(self.K * t**3 / (12 * E_ML * self.epsilon))))
        if n > 10_000:
            print(f"Warning: t={t}: required N_SP={n} exceeds cap. Reducing to 10 000.")
        # cap at 10k to avoid excessive computation, minimum 10
        n = max(min(n, 10_000), 10)  
        return n 


    def _compute_bound_at_time(self, t_target: float, _depth: int = 0) -> dict:
        """
        Compute the domain-wise L^2 error bound at a single target time,
        mirroring Compute_Function_Error (run.py:121-169).

        Steps:
        1. Choose N_SP quadrature subintervals adaptively via K.
        2. Build time grid: linspace(0, t_target, N_SP + 1).
        3. At each time s_k, evaluate ||R(s_k, .)||_{L^2} on the spatial
           grid with Richardson error on the spatial integral.
        4. Smooth: zeta_smooth = sqrt(zeta^2 + mu^2).
        5. Time-integrate: trapz(exp(-omega*s) * zeta_smooth) with
           Richardson error E = 1/3 * |I1 - I2|.
        6. Assemble: epsilon = M * exp(omega*t) * (I1 + E).
        7. If E is too large compared to expected ML error at t_target, increase
           N_SP and recompute (adaptive refinement).

        Parameters
        ----------
        t_target : float
            Target time at which to evaluate the bound.
        _depth : int
            Recursion depth for adaptive refinement (internal use only).

        Returns
        -------
        dict with keys:
            'epsilon'      : total bound (scalar)
            'epsilon_init' : IC contribution (0 for hard IC)
            'epsilon_eq'   : equation/integral contribution
            'n_sp'         : number of support point subintervals used
            'zeta'         : residual norms at quadrature times, shape (N_SP+1,)
            'zeta_errs'    : residual norm error estimates at quadrature times, shape (N_SP+1,)
            't_quad'       : time quadrature grid, shape (N_SP+1,)
        """
        if t_target <= 0:
            return {
                "epsilon": self.M * self.r0,
                "epsilon_init": self.M * self.r0,
                "epsilon_eq": 0.0,
                "n_sp": 0,
                "zeta": np.array([0.0]),
                "zeta_errs": np.array([0.0]),
                "t_quad": np.array([0.0]),
            }

        # Step 1: retrieve N_SP based on K and expected ML error at t_target
        N_SP = self._compute_n_support_points(t_target)

        # Step 2: time quadrature grid (N_SP subintervals → N_SP+1 points)
        # NOTE: the original code uses 2*N_SP + 1 
        # but I don't see the reason for it since the error estimate is based on 
        # N_SP subintervals, not 2*N_SP.
        # maybe by accident?
        t_quad = np.linspace(0, t_target, N_SP + 1)

        # Step 3: compute zeta(s_k) = ||R(s_k, .)||_{L^2} + spatial error
        zeta = np.zeros(N_SP + 1)
        zeta_errs = np.zeros(N_SP + 1)
        for k in range(N_SP + 1):
            x_t = np.column_stack([self.x, np.full(self.nx, t_quad[k])])
            R = self.pinn_model.residual(x_t).flatten()
            R[0] = 0.0  # hard BC
            R[-1] = 0.0
            # norm_upper includes spatial Richardson error: sqrt(I1) + sqrt(E)
            norm_fine, _, err_estimate = self._compute_spatial_l2_norm(R)
            zeta[k] = norm_fine + err_estimate
            zeta_errs[k] = err_estimate

        # Step 4: smooth integrand
        zeta_smooth = np.sqrt(zeta**2 + self.mu**2)

        # Step 5: time integration with Richardson error
        integrand = np.exp(-self.omega * t_quad) * zeta_smooth
        I1 = np.trapezoid(integrand, t_quad)
        I2 = np.trapezoid(integrand[::2], t_quad[::2])
        E_time = 1/3 * abs(I1 - I2)

        # Step 6: assemble bound
        exp_factor = self.M * np.exp(self.omega * t_target)
        epsilon_init = exp_factor * self.r0
        epsilon_eq = exp_factor * (I1 + E_time)

        # Step 7: adaptive refinement 
        # if the estimated integration error is too large compared to the expected ML error:
        # increase N_SP and recompute until it's acceptable or we hit the cap
        if N_SP > 0 and N_SP < 10_000 and _depth < 4:
            K_upd = 12 * E_time * N_SP**2 / (t_target**3)
            E_ML = self._compute_expected_ml_error(
                t_target, self.omega, self.M, self.r0, self.delta_mean
            )
            if E_ML > 0:
                N_SP_2 = int(np.ceil(np.sqrt(
                    K_upd * t_target**3 / (12 * E_ML * self.epsilon)
                )))
                if N_SP_2 > N_SP:
                    # Need more points — update K and recurse
                    self.K = K_upd
                    return self._compute_bound_at_time(t_target, _depth=_depth + 1)


        return {
            "epsilon": epsilon_init + epsilon_eq,
            "epsilon_init": epsilon_init,
            "epsilon_eq": epsilon_eq,
            "n_sp": N_SP,
            "zeta": zeta,
            "zeta_errs": zeta_errs,
            "t_quad": t_quad,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, t_eval: np.ndarray | None = None) -> dict:
        """
        Compute the semigroup-based L^2 error bound at specified times.

        This mirrors the full original pipeline:
        1. Extract parameters (K, mu, delta_mean) — the ``extract`` step.
        2. For each target time, adaptively compute the bound — the ``run``
           step.

        Parameters
        ----------
        t_eval : np.ndarray or None
            Times at which to evaluate the bound.  If None, uses 50
            linearly spaced points over the temporal domain.

        Returns
        -------
        dict with keys:
            'epsilon'      : total bound at each eval time, shape (n_eval,)
            'epsilon_init' : IC contribution, shape (n_eval,)
            'epsilon_eq'   : equation contribution, shape (n_eval,)
            'n_sp'         : support points used per time, shape (n_eval,)
            'zeta'         : zeta values at quadrature points, shape (n_eval, N_SP+1)
            'zeta_errs'    : zeta error estimates at quadrature points, shape (n_eval, N_SP+1)
            't'            : evaluation times, shape (n_eval,)
        """
        start_time = time.time()

        if t_eval is None:
            t_eval = np.linspace(
                self.domain.temporal_bounds[0],
                self.domain.temporal_bounds[1],
                50,
            )
        t_eval = np.atleast_1d(t_eval)

        # Step 1: extract parameters (K, mu, delta_mean) needed for the bound computation
        self._extract_parameters()
        print(f"Extracted parameters: K={self.K:.4e}, mu={self.mu:.4e}, delta_mean={self.delta_mean:.4e}, M={self.M:.4e}, omega={self.omega:.4e}")

        # Step 2: compute bound at each target time
        n_eval = len(t_eval)
        epsilon = np.zeros(n_eval)
        epsilon_init = np.zeros(n_eval)
        epsilon_eq = np.zeros(n_eval)
        n_sp = np.zeros(n_eval, dtype=int)

        for i, t in tqdm(enumerate(t_eval), total=n_eval, desc="Computing bounds"):
            result = self._compute_bound_at_time(t)
            epsilon[i] = result["epsilon"]
            epsilon_init[i] = result["epsilon_init"]
            epsilon_eq[i] = result["epsilon_eq"]
            n_sp[i] = result["n_sp"]

        self._run_time = time.time() - start_time

        return {
            "epsilon": epsilon,
            "epsilon_init": epsilon_init,
            "epsilon_eq": epsilon_eq,
            "n_sp": n_sp,
            "t": t_eval,
        }

    @property
    def run_time(self) -> float:
        """Returns the bound computation time in seconds."""
        return self._run_time