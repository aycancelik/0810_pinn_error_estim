"""FDM solver and error estimation methods"""

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pinn_error.core.pinn import PINNTrainer
from pinn_error.core.problem import BaseProblem, ProblemDomain
from pinn_error.problems.drift_diffusion import DriftDiffusion
from pinn_error.problems.poisson import Poisson1D, Poisson2D
from pinn_error.problems.wave import Wave1D


class FDMMatrixBuilder:
    """Generate spatial derivative matrices using finite differences."""

    @staticmethod
    def get_derivative_matrix_1d(
        n: int,
        spacing: float,
        order: int = 2,
        periodic: bool = False,
    ) -> sparse.csr_matrix:
        """
        Build 1D derivative matrix.

        Parameters
        ----------
        n : int
            Number of grid points
        spacing : float
            Grid spacing (dx or dt)
        order : int
            Derivative order (1 or 2)
        periodic : bool
            Use periodic boundary conditions

        Returns
        -------
        sparse.csr_matrix
            Derivative matrix (n × n)

        Notes
        -----
        For order=2 (second derivative):
            Matrix represents (1/dx²) * [1, -2, 1] stencil
        """
        if order == 1:
            # First derivative: central difference
            # (u_{i+1} - u_{i-1}) / (2*dx)
            diag_upper = np.ones(n - 1)
            diag_lower = -np.ones(n - 1)
            A = sparse.diags(
                [diag_lower, diag_upper], [-1, 1], shape=(n, n), format="csr"
            )
            return A / (2 * spacing)

        elif order == 2:
            # Second derivative: central difference
            # (u_{i+1} - 2*u_i + u_{i-1}) / dx²
            diag_main = -2 * np.ones(n)
            diag_off = np.ones(n - 1)
            A = sparse.diags(
                [diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr"
            )

            if periodic:
                # Add wraparound connections
                A_lil = A.tolil()
                A_lil[0, -1] = 1.0
                A_lil[-1, 0] = 1.0
                return A_lil.tocsr() / (spacing**2)

            return A / (spacing**2)

        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

    @staticmethod
    def get_laplacian_2d(
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> sparse.csr_matrix:
        """
        Build 2D Laplacian matrix: ∂²/∂x² + ∂²/∂y²

        Uses 5-point stencil with lexicographic ordering:
        point (i, j) → index k = j * nx + i

        Parameters
        ----------
        nx, ny : int
            Number of grid points in x and y
        dx, dy : float
            Grid spacings

        Returns
        -------
        sparse.csr_matrix
            Laplacian matrix (nx*ny × nx*ny)
        """
        n = nx * ny
        rx = 1.0 / (dx**2)
        ry = 1.0 / (dy**2)

        # Main diagonal: -2*(1/dx² + 1/dy²)
        diag_main = -2 * (rx + ry) * np.ones(n)

        # x-neighbors (±1 in index)
        diag_x = rx * np.ones(n - 1)
        # Zero out connections across row boundaries
        for j in range(1, ny):
            diag_x[j * nx - 1] = 0.0

        # y-neighbors (±nx in index)
        diag_y = ry * np.ones(n - nx)

        A = sparse.diags(
            [diag_y, diag_x, diag_main, diag_x, diag_y],
            [-nx, -1, 0, 1, nx],
            format="csr",
        )

        return A


class FDMSolverPoisson1D:
    def __init__(
        self,
        nx: int,
        problem: Poisson1D,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
        hard_constrain_boundary: bool = True,
    ):
        """Initialize the FDM solver for the 1D Poisson equation.

        Steady-state problem (see Section 3.2)
            L @ u = f   (solving for u)
            L @ e = -R  (error estimation)

        Note: Poisson is time-independent, so there's no initial condition
        here -- BC is the only constraint, unlike the time-dependent solvers.

        Args:
            nx (int): Number of spatial grid points.
            problem (Poisson1D): The problem definition.
            domain (ProblemDomain): The problem domain.
            pinn_model (PINNTrainer): The trained PINN model.
            hard_constrain_boundary (bool, optional): Whether BC is
                    hard-constrained in the PINN. If False (constraint_mode=
                    "soft_full"), the actual boundary error is computed
                    directly instead of assumed to be zero. Defaults to True.
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model
        self.hc_boundary = hard_constrain_boundary

        # number grid points
        self.nx = nx

        # grid spacing (distance between points)
        self.dx = (self.domain.spatial_bounds[1] - self.domain.spatial_bounds[0]) / (
            nx - 1
        )

        # create spatial and temporal grids
        self.x = np.linspace(
            self.domain.spatial_bounds[0], self.domain.spatial_bounds[1], nx
        )

        self._run_time = 0.0

        self.source = self.problem.source_term(self.x)

        # flag for stability issues
        # always stable since steady
        self.stability_flag = False

    def _build_matrices(self):
        n = self.nx

        # Step 1: Build spatial derivative matrix (full size)
        A_xx = FDMMatrixBuilder.get_derivative_matrix_1d(n, self.dx, order=2)
        spatial_operator = -1 * A_xx  # Poisson: -u'' = f

        # Step 3: Apply Dirichlet BCs (modify boundary rows)
        L_lil = spatial_operator.tolil()
        # First boundary point (x=0)
        L_lil[0, :] = 0
        L_lil[0, 0] = 1
        # Last boundary point (x=L)
        L_lil[-1, :] = 0
        L_lil[-1, -1] = 1

        self.L = L_lil.tocsr()

    def solve(self):
        """Solve the problem using FDM"""
        self._build_matrices()

        start_time = time.time()

        # initialize solution matrix
        # note this covers boundary conditions as well
        # (they're zero here)
        u = np.zeros((1, self.nx))

        # Solve for steady state (Poisson)
        # only need to solve once
        b = self.source.copy()
        b[0] = 0  # BC at x=0
        b[-1] = 0  # BC at x=L

        # solve everything at once !?
        u_int = spsolve(self.L, b)
        u = u_int.reshape(1, -1)

        self._run_time = time.time() - start_time
        return u

    def residual_integration(self):
        self._build_matrices()

        # initialize error matrix
        e = np.zeros((1, self.nx))

        start_time = time.time()

        # Get residual values
        x_int = self.x
        R = self.pinn_model.residual(x_int.reshape(-1, 1)).flatten()

        # Solve for error steady state
        b = -R
        if self.hc_boundary:
            b[0] = 0  # BC at x=0
            b[-1] = 0  # BC at x=L
        else:
            # BC is soft: use the actual boundary error directly (known/
            # prescribed data), same convention as FDMSolverHeatEq
            x_bd = np.array([x_int[0], x_int[-1]])
            b[[0, -1]] = self.problem.exact_solution(x_bd) - self.pinn_model.predict(
                x_bd.reshape(-1, 1)
            ).flatten()
        # solve everything at once
        e_int = spsolve(self.L, b)
        e = e_int.reshape(1, -1)

        self._run_time = time.time() - start_time

        return e

    @property
    def run_time(self) -> float:
        """Returns the FDM solve/error approx. time in seconds

        Note: it assumes that either solve() or residual_integration() has been called (exclusively).
        """
        return self._run_time


class FDMSolverPoisson2D:
    def __init__(
        self,
        nx: int,
        ny: int,
        problem: Poisson2D,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
    ):
        """Initialize the 2D FDM solver for Poisson equation.

        Steady-state problem (see Section 3.2)
            L @ u = f   (solving for u)
            L @ e = -R  (error estimation)

        Args:
            nx (int): Number of spatial grid points in x.
            ny (int): Number of spatial grid points in y.
            problem (Poisson2D): The problem definition.
            domain (ProblemDomain): The problem domain.
            pinn_model (PINNTrainer): The trained PINN model.
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model

        self.nx = nx
        self.ny = ny

        self.dx = (self.domain.x_max - self.domain.x_min) / (nx - 1)
        self.dy = (self.domain.y_max - self.domain.y_min) / (ny - 1)

        self.x = np.linspace(self.domain.x_min, self.domain.x_max, nx)
        self.y = np.linspace(self.domain.y_min, self.domain.y_max, ny)

        self._run_time = 0.0

        # Source term on interior grid
        X_int, Y_int = np.meshgrid(self.x, self.y, indexing="xy")
        self.source = self.problem.source_term(X_int.ravel(), Y_int.ravel())

        # Reshape source to 2D and apply BCs (zero Dirichlet)
        self.source = self.source.reshape(self.ny, self.nx)
        self.source[0, :] = 0  # y = y_min
        self.source[-1, :] = 0  # y = y_max
        self.source[:, 0] = 0  # x = x_min
        self.source[:, -1] = 0  # x = x_max

        self.source = self.source.ravel()  # flatten to 1D for solver

        self.stability_flag = False  # Poisson is always stable

    def _build_matrices(self):
        nx = self.nx
        ny = self.ny

        # Build 2D Laplacian matrix for interior points
        self.A_xx_yy = FDMMatrixBuilder.get_laplacian_2d(nx, ny, self.dx, self.dy)

        # apply PDE coefficient (for Poisson it's just -1)
        spatial_operator = -self.A_xx_yy

        # Apply Dirichlet BCs (modify boundary rows)
        L_lil = spatial_operator.tolil()
        for j in range(ny):
            for i in range(nx):
                k = j * nx + i  # lexicographic index
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    L_lil[k, :] = 0
                    L_lil[k, k] = 1
        self.L = L_lil.tocsr()

    def solve(self):
        """Solve the 2D Poisson equation using FDM."""
        self._build_matrices()

        start_time = time.time()

        u = np.zeros((self.ny, self.nx))

        # already in lexicographic order (flattened)
        b = self.source 
        # solve L @ u_int = b for interior points (flattened)
        u_int = spsolve(self.L, b)

        # Reshape interior solution back to 2D and place in full grid
        u_int_2d = u_int.reshape(self.ny, self.nx)  
        u = u_int_2d

        self._run_time = time.time() - start_time
        return u

    def residual_integration(self):
        """Estimate PINN error via residual integration on 2D grid."""
        self._build_matrices()

        start_time = time.time()

        # Interior grid points for PINN residual evaluation
        X_int, Y_int = np.meshgrid(self.x, self.y, indexing="xy")
        points = np.column_stack([X_int.ravel(), Y_int.ravel()])

        R = self.pinn_model.residual(points).flatten()

        R = R.reshape(self.ny, self.nx)
        R[0, :] = 0  # y = y_min
        R[-1, :] = 0  # y = y_max
        R[:, 0] = 0  # x = x_min
        R[:, -1] = 0  # x = x_max
        R = R.ravel()

        # Solve L @ e_int = -R (flattened)
        e_int = spsolve(self.L, -R)

        # Reshape to full grid
        e = np.zeros((self.ny, self.nx))
        e_int_2d = e_int.reshape(self.ny, self.nx) 
        e = e_int_2d
        self._run_time = time.time() - start_time
        return e

    @property
    def run_time(self) -> float:
        """Returns the FDM solve/error approx. time in seconds"""
        return self._run_time
    

class FDMSolverHeatEq:
    def __init__(
        self,
        nx: int,
        nt: int,
        problem: BaseProblem,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
        hard_constrain_initial: bool = False,
        hard_constrain_boundary: bool = True,
    ):
        """Initialize the FDM solver for the 1D heat equation.
        Uses Crank-Nicolson time stepping (see Section 3.2)

        solving for u:
            (I - 0.5*dt*L) @ u^{n+1} = (I + 0.5*dt*L) @ u^n
        error estimation (see Section 4)
            (I - 0.5*dt*L) @ e^{n+1} = (I + 0.5*dt*L) @ e^n - 0.5*dt*(R^n + R^{n+1})

        Args:
            nx (int): Number of spatial grid points.
            nt (int): Number of temporal grid points.
            problem (BaseProblem): The problem definition.
            domain (ProblemDomain): The problem domain.
            pinn_model (PINNTrainer): The trained PINN model.
            hard_constrain_initial (bool, optional): Whether IC is hard-constrained
                    in the PINN (i.e. constraint_mode="hard"). If False, the actual
                    pointwise error at t=0 is computed instead of assuming it's zero.
                    Should match the PINNTrainer's constraint_mode used to train
                    pinn_model: True for "hard", False for "soft_ic"/"soft_full".
                    Note that even with hard constraints we encounter some
                    numerical error at t=0, hence there is a non-zero
                    (but very close to zero) initial error step.
            hard_constrain_boundary (bool, optional): Whether BC is hard-constrained
                    in the PINN. If False, the actual boundary error is computed at
                    every time step instead of assuming it's zero. Should be False
                    only for constraint_mode="soft_full" (both "hard" and "soft_ic"
                    hard-constrain BC). Defaults to True.
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model
        self.hc_initial = hard_constrain_initial
        self.hc_boundary = hard_constrain_boundary

        # number grid points
        self.nx = nx
        self.nt = nt

        # grid spacing (distance between points)
        self.dx = (self.domain.spatial_bounds[1] - self.domain.spatial_bounds[0]) / (
            nx - 1
        )
        self.dt = (self.domain.temporal_bounds[1] - self.domain.temporal_bounds[0]) / (
            nt - 1
        )

        # create spatial and temporal grids
        self.x = np.linspace(
            self.domain.spatial_bounds[0], self.domain.spatial_bounds[1], nx
        )
        self.t = np.linspace(
            self.domain.temporal_bounds[0], self.domain.temporal_bounds[1], nt
        )

        # flag for stability issues -- Crank-Nicolson is unconditionally stable
        self.stability_flag = False  

        self._run_time = 0.0

    def _build_matrices(self):
        """Build time-stepping matrices with boundary conditions baked in."""
        n = self.nx  # Full size (including boundaries)

        # Build spatial derivative matrix (see above)
        A_xx = FDMMatrixBuilder.get_derivative_matrix_1d(n, self.dx, order=2)

        # Apply PDE coefficient
        # spatial operator is diffusivity * d²/dx²
        spatial_operator = self.problem.diffusivity * A_xx

        # Build time-stepping matrices (Crank-Nicolson)
        I = sparse.eye(n, format="csr")
        M_lhs = I - 0.5 * self.dt * spatial_operator
        M_rhs = I + 0.5 * self.dt * spatial_operator

        # Apply Dirichlet BCs (modify boundary rows)
        M_lhs_lil = M_lhs.tolil()
        M_rhs_lil = M_rhs.tolil()

        # First boundary point (x=0)
        M_lhs_lil[0, :] = 0
        M_lhs_lil[0, 0] = 1  # Identity row
        M_rhs_lil[0, :] = 0  # Zero row (gives rhs=0)

        # Last boundary point (x=L)
        M_lhs_lil[-1, :] = 0
        M_lhs_lil[-1, -1] = 1  # Identity row
        M_rhs_lil[-1, :] = 0  # Zero row (gives rhs=0)

        # Convert to sparse CSR format
        self.M_lhs = M_lhs_lil.tocsr()
        self.M_rhs = M_rhs_lil.tocsr()

    def solve(self):
        """Solve the problem using FDM."""
        self._build_matrices()

        start_time = time.time()

        # Initialize solution (full grid)
        u = np.zeros((self.nt, self.nx))
        # Set u^0 = initial condition
        u[0, :] = self.problem.initial_condition(self.x)

        # Time stepping, n+1 from n
        for n in range(self.nt - 1):
            # right hand side is known
            rhs = self.M_rhs @ u[n]
            # solve for next time step u^{n+1}
            u[n + 1] = spsolve(self.M_lhs, rhs)

        # Record run time
        self._run_time = time.time() - start_time
        return u

    def residual_integration(self):
        """Integrate PINN residuals."""
        self._build_matrices()

        start_time = time.time()

        # Initialize error (full grid)
        e = np.zeros((self.nt, self.nx))
        if not (self.hc_initial and self.hc_boundary):
            # potentially less stable but accounts for any numerical issues at t0
            # (also covers the boundary entries when BC is soft: at t=0 they're
            # just the same pointwise error formula evaluated at x=x_min/x_max)
            e[0, :] = self.problem.initial_condition(self.x) - self.pinn_model.predict(
                np.column_stack([self.x, self.t[0] * np.ones_like(self.x)])
            ).flatten()

        # Get residual values at initial time step
        R_curr = self.pinn_model.residual(
            np.column_stack([self.x, self.t[0] * np.ones_like(self.x)])
        ).flatten()
        # PDE residual at BCs is meaningless (deepxde applies the PDE operator
        # there too); always zero it out regardless of hc_boundary, since the
        # boundary error itself is handled separately below via direct
        # injection, not via residual integration.
        R_curr[0] = 0.0
        R_curr[-1] = 0.0

        # Time stepping, n+1 from n
        for n in range(self.nt - 1):
            t_next = self.t[n + 1]
            rhs = self.M_rhs @ e[n]

            # Compute PINN residuals at current and next time steps
            x_int = self.x

            R_next = self.pinn_model.residual(
                np.column_stack([x_int, t_next * np.ones_like(x_int)])
            ).flatten()

            # since deepxde applies PDE residual on boundary we
            # have to set them to zero here
            R_next[0] = 0.0  # BC points
            R_next[-1] = 0.0

            # Add residual source term (only to interior points)
            # residual_source = np.zeros(self.nx)
            residual_source = - 0.5 * self.dt * (
                R_curr + R_next
            )
            rhs += residual_source

            # M_lhs/M_rhs bake in identity/zero boundary rows unconditionally
            # (see _build_matrices), so rhs[0]/rhs[-1] are already ~0 at this
            # point regardless of hc_boundary. If BC is hard-constrained, that's
            # exactly what we want (e stays 0 at the boundary for all t). If BC
            # is soft, overwrite with the actual boundary error directly --
            # this is known/prescribed data (same convention already used in
            # FDMSolverDriftDiffusion._get_boundary_values), not something that
            # needs residual integration.
            if not self.hc_boundary:
                x_bd = np.array([self.x[0], self.x[-1]])
                t_bd = t_next * np.ones_like(x_bd)
                rhs[[0, -1]] = self.problem.exact_solution(x_bd, t_bd) - self.pinn_model.predict(
                    np.column_stack([x_bd, t_bd])
                ).flatten()

            # Solve for next error step e^{n+1}
            e[n + 1] = spsolve(self.M_lhs, rhs)

            R_curr = R_next  # for next iteration

        self._run_time = time.time() - start_time
        return e

    @property
    def run_time(self) -> float:
        """Returns the FDM solve/error approx. time in seconds

        Note: it assumes that either solve() or residual_integration() has been called (exclusively).
        """
        return self._run_time
    

class FDMSolverDriftDiffusion:
    def __init__(
        self,
        nx: int,
        nt: int,
        problem: DriftDiffusion,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
        hard_constrain_initial: bool = False,
        hard_constrain_boundary: bool = True,

    ):
        """Initialize the FDM solver for the 1D Drift-Diffusion equation.

        PDE: u_t = diffusivity * u_xx - velocity_x * u_x 

        Uses Crank-Nicolson time stepping (see Section 3.2)
            (I - 0.5*dt*L) @ u^{n+1} = (I + 0.5*dt*L) @ u^n

        Args:
            nx (int): Number of spatial grid points.
            nt (int): Number of temporal grid points.
            problem (DriftDiffusion): The problem definition.
            domain (ProblemDomain): The problem domain.
            pinn_model (PINNTrainer): The trained PINN model.
            hard_constrain_initial (bool, optional): Whether to hard-constrain 
                    initial condition in error integration. Defaults to True.
                    Note that even with hard constraints we encounter some 
                    numerical error at t=0, hence there is a non-zero 
                    (but very close to zero) initial error step.
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model
        self.hard_constrain_initial = hard_constrain_initial
        self.hard_constrain_boundary = hard_constrain_boundary

        self.nx = nx
        self.nt = nt

        # Grid spacing
        self.dx = (self.domain.spatial_bounds[1] - self.domain.spatial_bounds[0]) / (
            nx - 1
        )
        self.dt = (self.domain.temporal_bounds[1] - self.domain.temporal_bounds[0]) / (
            nt - 1
        )

        # Create grids
        self.x = np.linspace(
            self.domain.spatial_bounds[0], self.domain.spatial_bounds[1], nx
        )
        self.t = np.linspace(
            self.domain.temporal_bounds[0], self.domain.temporal_bounds[1], nt
        )

        self._run_time = 0.0

        self.stability_flag = False  # flag for stability issues

    def _build_matrices(self):
        """Build time-stepping matrices for advection-diffusion equation.

        The spatial operator is: L = D * d²/dx² - β * d/dx

        Time stepping:
            M_lhs @ u^{n+1} = M_rhs @ u^n + BC_contribution
        """
        n = self.nx
        D = self.problem.diffusivity
        beta = self.problem.velocity_x

        # Build derivative matrices
        A_xx = FDMMatrixBuilder.get_derivative_matrix_1d(n, self.dx, order=2)
        A_x = FDMMatrixBuilder.get_derivative_matrix_1d(n, self.dx, order=1)

        # Spatial operator: D * u_xx - β * u_x
        # PDE: u_t = D * u_xx - β * u_x
        spatial_operator = D * A_xx - beta * A_x

        # Build time-stepping matrices
        I = sparse.eye(n, format="csr")
        M_lhs = I - 0.5 * self.dt * spatial_operator
        M_rhs = I + 0.5 * self.dt * spatial_operator

        # Apply Dirichlet BCs (modify boundary rows)
        M_lhs_lil = M_lhs.tolil()
        M_rhs_lil = M_rhs.tolil()

        # First boundary point (x = x_min)
        M_lhs_lil[0, :] = 0
        M_lhs_lil[0, 0] = 1
        M_rhs_lil[0, :] = 0

        # Last boundary point (x = x_max)
        M_lhs_lil[-1, :] = 0
        M_lhs_lil[-1, -1] = 1
        M_rhs_lil[-1, :] = 0

        self.M_lhs = M_lhs_lil.tocsr()
        self.M_rhs = M_rhs_lil.tocsr()

        # Stability analysis
        # Peclet number: Pe = β * dx / D
        Pe = abs(beta) * self.dx / D if D > 0 else float("inf")

        if Pe > 2.0:
            print(
                f"Note: Peclet number Pe = {Pe:.2f} > 2 (advection-dominated). "
                "FDM may be unstable"
            )
            self.stability_flag = True

    def _get_boundary_values(self, t: float) -> tuple[float, float]:
        """Get boundary values at time t from exact solution.
        (Periodic BC requires left boundary == right boundary)

        Args:
            t: Time value.

        Returns:
            Tuple of (u_left, u_right) boundary values.
        """
        x_min = self.domain.spatial_bounds[0]
        x_max = self.domain.spatial_bounds[1]

        u_left = self.problem.exact_solution(x_min, t)
        u_right = self.problem.exact_solution(x_max, t)

        return float(u_left), float(u_right)

    def solve(self):
        """Solve the drift-diffusion equation using FDM.

        Returns:
            np.ndarray: Solution array with shape (nt, nx).
        """
        self._build_matrices()

        start_time = time.time()

        # Initialize solution
        u = np.zeros((self.nt, self.nx))

        # Set initial condition
        u[0, :] = self.problem.initial_condition(self.x)

        # Time stepping
        for n in range(self.nt - 1):
            # RHS from previous step (interior contribution)
            rhs = self.M_rhs @ u[n]

            # Add boundary values for next time step
            bc_left_next, bc_right_next = self._get_boundary_values(self.t[n + 1])

            # Boundary rows of M_rhs are zero, so we just set the BC values directly
            rhs[0] = bc_left_next
            rhs[-1] = bc_right_next

            # Solve for next time step
            u[n + 1] = spsolve(self.M_lhs, rhs)

        self._run_time = time.time() - start_time
        return u

    def residual_integration(self):
        """Integrate PINN residuals to estimate error.
 
        Returns:
            np.ndarray: Error estimate array with shape (nt, nx).
        """
        self._build_matrices()
 
        start_time = time.time()
 
        # Initialize error
        e = np.zeros((self.nt, self.nx))
        if not (self.hard_constrain_initial and self.hard_constrain_boundary):
            # potentially less stable but accounts for any numerical issues at t0
            # (also covers the boundary entries when BC is soft: at t=0 they're
            # just the same pointwise error formula evaluated at x=x_min/x_max)
            e[0, :] = self.problem.initial_condition(self.x) - self.pinn_model.predict(
                np.column_stack([self.x, self.t[0] * np.ones_like(self.x)])
            ).flatten()
 
        def _get_boundary_error(t: float) -> np.ndarray:
            """Actual pointwise error at x_min/x_max at a given time. This is
            known/prescribed data (see _get_boundary_values above, which does
            the same thing for the true solution u), not something that needs
            residual integration -- used when BC is soft."""
            x_bd = np.array([self.domain.spatial_bounds[0], self.domain.spatial_bounds[1]])
            t_bd = t * np.ones_like(x_bd)
            u_exact = self.problem.exact_solution(x_bd, t_bd)
            u_pinn = self.pinn_model.predict(np.column_stack([x_bd, t_bd])).flatten()
            return u_exact - u_pinn
 
        # Get residual at initial time
        R_curr = self.pinn_model.residual(
            np.column_stack([self.x, self.t[0] * np.ones_like(self.x)])
        ).flatten()
        R_curr[0] = 0.0  # BC points
        R_curr[-1] = 0.0
 
        # Time stepping
        for n in range(self.nt - 1):
            t_next = self.t[n + 1]
 
            # RHS from previous step
            rhs = self.M_rhs @ e[n]
 
            # Compute residual at next time step
            R_next = self.pinn_model.residual(
                np.column_stack([self.x, t_next * np.ones_like(self.x)])
            ).flatten()
            R_next[0] = 0.0  # BC points
            R_next[-1] = 0.0
 
            residual_source = - 0.5 * self.dt * (
                R_curr + R_next
            )
            rhs += residual_source
 
            # M_lhs/M_rhs bake in identity/zero boundary rows unconditionally
            # (see _build_matrices), so rhs[0]/rhs[-1] are already ~0 at this
            # point regardless of hard_constrain_boundary. If BC is hard-
            # constrained, that's exactly what we want (e stays 0 at the
            # boundary for all t). If BC is soft, overwrite with the actual
            # boundary error directly via _get_boundary_error.
            if self.hard_constrain_boundary:
                rhs[0] = 0.0
                rhs[-1] = 0.0
            else:
                rhs[[0, -1]] = _get_boundary_error(t_next)
 
            # Solve for next error step
            e[n + 1] = spsolve(self.M_lhs, rhs)
 
            R_curr = R_next  # Store for next iteration
 
        self._run_time = time.time() - start_time
        return e

    @property
    def run_time(self) -> float:
        """Returns the FDM solve/error approx. time in seconds."""
        return self._run_time


class FDMSolverWave1D:
    def __init__(
        self,
        nx: int,
        nt: int,
        problem: Wave1D,
        domain: ProblemDomain,
        pinn_model: PINNTrainer,
        hard_constrain_initial: bool = False,
        hard_constrain_boundary: bool = True,
    ):
        """Initialize the FDM solver for the 1D wave equation.

        Uses central differences for second order time stepping.

        Time stepping (see Section 3.2.):
           u^{n+1} = 2u^n + dt^2 * L @ u_n - u^{n-1}


        Args:
            nx (int): Number of spatial grid points.
            nt (int): Number of temporal grid points.
            problem (BaseProblem): The problem definition.
            domain (ProblemDomain): The problem domain.
            pinn_model (PINNTrainer): The trained PINN model.
            hard_constrain_initial (bool, optional): Whether IC is hard-constrained
                    in the PINN (i.e. constraint_mode="hard"). If False, the actual
                    pointwise error at t=0 is computed instead of assuming it's zero.
                    Should match the PINNTrainer's constraint_mode used to train
                    pinn_model: True for "hard", False for "soft_ic"/"soft_full".
                    Note that even with hard constraints we encounter some
                    numerical error at t=0, hence there is a non-zero
                    (but very close to zero) initial error step.
            hard_constrain_boundary (bool, optional): Whether BC is hard-constrained
                    in the PINN. If False, the actual boundary error is computed at
                    every time step instead of assuming it's zero. Should be False
                    only for constraint_mode="soft_full" (both "hard" and "soft_ic"
                    hard-constrain BC). Defaults to True.
        """
        self.problem = problem
        self.domain = domain
        self.pinn_model = pinn_model
        self.hard_constrain_initial = hard_constrain_initial
        self.hard_constrain_boundary = hard_constrain_boundary

        self.nx = nx
        self.nt = nt

        self.dx = (self.domain.spatial_bounds[1] - self.domain.spatial_bounds[0]) / (
            nx - 1
        )
        self.dt = (self.domain.temporal_bounds[1] - self.domain.temporal_bounds[0]) / (
            nt - 1
        )

        self.x = np.linspace(
            self.domain.spatial_bounds[0], self.domain.spatial_bounds[1], nx
        )
        self.t = np.linspace(
            self.domain.temporal_bounds[0], self.domain.temporal_bounds[1], nt
        )

        self._run_time = 0.0

        self.stability_flag = False 

    def _build_spatial_operator(self):
        """Build spatial operator matrix L for wave equation."""
        n = self.nx
        A_xx = FDMMatrixBuilder.get_derivative_matrix_1d(n, self.dx, order=2)
        self.spatial_operator = self.problem.propagation_speed**2 * A_xx
        
        self.r = self.problem.propagation_speed * self.dt / self.dx
        if self.r > 1:
            print(f"Warning: Courant number r={self.r:.2f} > 1, solution may be unstable.")
            self.stability_flag = True

    def solve(self):
        """Solve the 1D wave equation using FDM."""
        self._build_spatial_operator()

        start_time = time.time()

        # Initialize solution array (full grid)
        u = np.zeros((self.nt, self.nx))
        # Set initial condition u(x,0) = f(x)
        u[0, :] = self.problem.initial_condition(self.x)

        # First step: Neumann IC du/dt(x,0) = 0 implies u^{-1} = u^1.
        # Substituting into the central-difference stencil at n=0:
        #   (2*u^1 - 2*u^0) / dt^2 = L @ u^0
        #   u^1 = u^0 + 0.5 * dt^2 * L @ u^0
        u[1] = u[0] + 0.5 * self.dt**2 * (self.spatial_operator @ u[0])
        u[1, 0] = 0.0  # BC at x=0
        u[1, -1] = 0.0  # BC at x=L

        for n in range(1, self.nt - 1):
            u[n + 1] = 2 * u[n] + self.dt**2 * (self.spatial_operator @ u[n]) - u[n - 1]
            u[n + 1, 0] = 0.0  # BC at x=0
            u[n + 1, -1] = 0.0  # BC at x=L

        self._run_time = time.time() - start_time
        return u

    def residual_integration(self):
        """Estimate PINN error via residual integration.

        Error equation (see Section 4)
            e^{n+1} = 2*e^n - e^{n-1} + dt^2 * (L @ e^n - R^n)
        """
        self._build_spatial_operator()

        e = np.zeros((self.nt, self.nx))
        if not (self.hard_constrain_initial and self.hard_constrain_boundary):  
            # potentially less stable but accounts for any numerical issues at t0
            e[0, :] = self.problem.initial_condition(self.x) - self.pinn_model.predict(
                np.column_stack([self.x, self.t[0] * np.ones_like(self.x)])
            ).flatten()

        start_time = time.time()
        x_int = self.x

        def _boundary_error(t_val: float) -> np.ndarray:
            """Actual pointwise error at x_min/x_max at a given time. This is
            known/prescribed data (same convention as
            FDMSolverDriftDiffusion._get_boundary_values), not something that
            needs residual integration -- used when BC is soft."""
            x_bd = np.array([self.x[0], self.x[-1]])
            t_bd = t_val * np.ones_like(x_bd)
            return self.problem.exact_solution(x_bd, t_bd) - self.pinn_model.predict(
                np.column_stack([x_bd, t_bd])
            ).flatten()
 
        # First step: de/dt(x,0) = 0 implies e^{-1} = e^1.
        # With e^0 = 0 this simplifies to:
        #   e^1 = -0.5 * dt^2 * R^0
        # (or more generally: e^1 = e^0 + 0.5*dt^2*(L@e^0 - R^0))
        # NOTE: this still assumes the PINN's initial velocity du/dt(x,0)
        # matches the true velocity IC (both taken as 0). That's only
        # guaranteed under constraint_mode="hard" (output_transform's `_t**2`
        # factor enforces it structurally); under soft_ic/soft_full nothing
        # currently supervises the velocity IC, so there's an unaccounted
        # error contribution from that mismatch in this first step.
        R_0 = self.pinn_model.residual(
            np.column_stack([x_int, self.t[0] * np.ones_like(x_int)])
        ).flatten()
        # PDE residual at BCs is meaningless (deepxde applies the PDE operator
        # there too); boundary error is handled separately below via direct
        # injection, not via residual integration.
        R_0[0] = 0.0
        R_0[-1] = 0.0
        e[1] = e[0] + 0.5 * self.dt**2 * (self.spatial_operator @ e[0] - R_0)
        if self.hard_constrain_boundary:
            e[1, 0] = 0.0  # error at x=0
            e[1, -1] = 0.0  # error at x=L
        else:
            e[1, [0, -1]] = _boundary_error(self.t[1])
 
        for n in range(1, self.nt - 1):
            R_n = self.pinn_model.residual(
                np.column_stack([x_int, self.t[n] * np.ones_like(x_int)])
            ).flatten()
            R_n[0] = 0.0
            R_n[-1] = 0.0
            e[n + 1] = 2 * e[n] - e[n - 1] + self.dt**2 * (self.spatial_operator @ e[n] - R_n)
            if self.hard_constrain_boundary:
                e[n + 1, 0] = 0.0  # error at x=0
                e[n + 1, -1] = 0.0  # error at x=L
            else:
                e[n + 1, [0, -1]] = _boundary_error(self.t[n + 1])
 
        self._run_time = time.time() - start_time
        return e


    @property
    def run_time(self) -> float:
        """Returns the FDM solve/error approx. time in seconds"""
        return self._run_time