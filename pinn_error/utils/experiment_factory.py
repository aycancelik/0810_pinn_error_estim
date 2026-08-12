import numpy as np
import time 

from pinn_error.core.fdm import (FDMSolverDriftDiffusion, FDMSolverHeatEq,
                                 FDMSolverPoisson1D, FDMSolverPoisson2D,
                                 FDMSolverWave1D)
from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.core.problem import BaseProblem, ProblemDomain
from pinn_error.problems import (DriftDiffusion, Heat1DProblemSineIC,
                                 Poisson1D, Poisson2D, Wave1D)

# Define accepted kwargs for each problem type
PROBLEM_KWARGS = {
    "heat": ["x_min", "x_max", "t_max", "diffusivity", "frequency"],
    "wave": ["x_min", "x_max", "t_max", "propagation_speed", "frequency"],
    "drift_diffusion": [
        "x_min",
        "x_max",
        "t_max",
        "initial_concentration",
        "frequency",
        "phase_shift",
        "diffusivity",
        "velocity_x",
    ],
    "poisson_1d": ["x_min", "x_max"],
    "poisson_2d": ["x_min", "x_max", "y_min", "y_max"],
}


def _filter_kwargs(problem_name: str, kwargs: dict) -> dict:
    """Filter kwargs to only include those accepted by the problem class."""
    accepted = PROBLEM_KWARGS.get(problem_name, [])
    return {k: v for k, v in kwargs.items() if k in accepted and v is not None}


def get_problem(problem_name: str, **kwargs):
    filtered_kwargs = _filter_kwargs(problem_name, kwargs)
    if problem_name == "heat":
        return Heat1DProblemSineIC(**filtered_kwargs)
    elif problem_name == "poisson_1d":
        return Poisson1D(**filtered_kwargs)
    elif problem_name == "poisson_2d":
        return Poisson2D(**filtered_kwargs)
    elif problem_name == "wave":
        return Wave1D(**filtered_kwargs)
    elif problem_name == "drift_diffusion":
        return DriftDiffusion(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")


_TIME_INDEPENDENT_PROBLEMS = ("poisson_1d", "poisson_2d")

# problems whose FDM solver currently supports soft IC/BC error estimation
# (i.e. has hard_constrain_initial/hard_constrain_boundary wired up)
_SOFT_CONSTRAINT_SUPPORTED = ("heat", "wave", "drift_diffusion")


def _derive_hc_flags(problem_name: str, constraint_mode: str) -> dict:
    """
    Derive the FDM solver's hard_constrain_initial/hard_constrain_boundary
    flags from a single PINNConfig.constraint_mode, so PINN training and FDM
    error estimation can't drift out of sync with each other.
    """
    is_time_dependent = problem_name not in _TIME_INDEPENDENT_PROBLEMS

    if constraint_mode == "hard":
        return {"hard_constrain_initial": True, "hard_constrain_boundary": True}
    elif constraint_mode == "soft_ic":
        if not is_time_dependent:
            raise ValueError(
                f"constraint_mode='soft_ic' doesn't apply to '{problem_name}' "
                "(time-independent, no initial condition). Use 'hard' or 'soft_full'."
            )
        return {"hard_constrain_initial": False, "hard_constrain_boundary": True}
    elif constraint_mode == "soft_full":
        return {"hard_constrain_initial": False, "hard_constrain_boundary": False}
    else:
        raise ValueError(f"Unknown constraint_mode: {constraint_mode!r}")


def get_fdm_solver(
    problem_name: str,
    problem: BaseProblem,
    domain: ProblemDomain,
    pinn_model: PINNTrainer,
    nx: int,
    ny: int,
    nt: int,
    constraint_mode: str = "hard",
):
    """
    Construct the appropriate FDM solver based on the problem name and provided parameters.

    constraint_mode should match the constraint_mode used to train pinn_model
    (PINNConfig.constraint_mode) so the FDM error estimation's assumptions
    about hard-constrained IC/BC stay consistent with how the PINN was
    actually trained.
    """
    if problem_name not in _SOFT_CONSTRAINT_SUPPORTED and constraint_mode != "hard":
        raise NotImplementedError(
            f"FDM solver for '{problem_name}' doesn't yet support soft IC/BC "
            f"error estimation (constraint_mode={constraint_mode!r}). Only "
            f"{_SOFT_CONSTRAINT_SUPPORTED} currently do; '{problem_name}' would "
            "need the same hard_constrain_initial/hard_constrain_boundary "
            "treatment added to its residual_integration()."
        )

    hc_flags = _derive_hc_flags(problem_name, constraint_mode)

    if problem_name == "heat":
        return FDMSolverHeatEq(
            problem=problem,
            domain=domain,
            pinn_model=pinn_model,
            nt=nt,
            nx=nx,
            hard_constrain_initial=hc_flags["hard_constrain_initial"],
            hard_constrain_boundary=hc_flags["hard_constrain_boundary"],
        )
    elif problem_name == "poisson_1d":
        return FDMSolverPoisson1D(
            problem=problem,
            domain=domain,
            pinn_model=pinn_model,
            nx=nx,
            hard_constrain_boundary=hc_flags["hard_constrain_boundary"],
        )
    elif problem_name == "poisson_2d":
        return FDMSolverPoisson2D(
            problem=problem,
            domain=domain,
            pinn_model=pinn_model,
            nx=nx,
            ny=ny,
            hard_constrain_boundary=hc_flags["hard_constrain_boundary"],
        )
    elif problem_name == "wave":
        return FDMSolverWave1D(
            problem=problem,
            domain=domain,
            pinn_model=pinn_model,
            nx=nx,
            nt=nt,
            hard_constrain_initial=hc_flags["hard_constrain_initial"],
            hard_constrain_boundary=hc_flags["hard_constrain_boundary"],
        )
    elif problem_name == "drift_diffusion":
        return FDMSolverDriftDiffusion(
            problem=problem,
            domain=domain,
            pinn_model=pinn_model,
            nx=nx,
            nt=nt,
            hard_constrain_initial=hc_flags["hard_constrain_initial"],
            hard_constrain_boundary=hc_flags["hard_constrain_boundary"],
        )
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")


def get_pinn_trainer(problem, config=None):
    """
    Construct a PINN trainer for the given problem and configuration.
    """
    if config is None:
        config = PINNConfig()  # use default config
    return PINNTrainer(problem=problem, config=config)


def get_problem_bundle(
    problem_name: str,
    problem_kwargs: dict,
    fdm_solver_kwargs: dict,
    pinn_config: PINNConfig,
):
    """
    Construct a problem, PINN trainer, and FDM solver bundle based on the problem name and provided configurations.
    """
    # if the problem is 1D Poisson, ensure the PINN input layer has 1 neuron
    if problem_name == "poisson_1d":
        pinn_config.layers[0] = 1
    # define the problem (PDE, domain, exact solution)
    problem = get_problem(problem_name, **problem_kwargs)
    # define the PINN trainer
    pinn_trainer = get_pinn_trainer(problem, config=pinn_config)
    # define the FDM solver
    # constraint_mode is taken from pinn_config (not fdm_solver_kwargs) so the
    # error estimation always matches how the PINN was actually trained
    fdm_solver = get_fdm_solver(
        problem_name,
        problem=problem,
        domain=problem.domain,
        pinn_model=pinn_trainer,
        constraint_mode=pinn_config.constraint_mode,
        **fdm_solver_kwargs,
    )
    return problem, pinn_trainer, fdm_solver


class ExperimentFactory:
    """
    Class to run experiments and retrieve desirable metrics for each run
    """
    def __init__(
        self,
        problem_name: str,
        problem_kwargs: dict,
        fdm_solver_kwargs: dict,
        pinn_config: PINNConfig,
        verbose: bool = True,
    ):
        """
        Args:
            problem_name (str): name of the PDE problem to solve 
                                (e.g. "heat", "poisson_1d", "poisson_2d", "wave", "drift_diffusion")
            problem_kwargs (dict): dict of kwargs to construct the problem 
                                (e.g. x_min, x_max, t_max, diffusivity, etc.)
            fdm_solver_kwargs (dict): dict of kwargs to construct the FDM solver 
                                (e.g. nx, ny, nt)
            pinn_config (PINNConfig): PINNConfig object containing the configuration for the PINN trainer 
                                (e.g. layers, activation, optimizer, etc.)
            verbose (bool): whether to print out info
        """
        self.is_2d = problem_name != "poisson_1d"
        self.is_time_dependent = problem_name not in ["poisson_1d", "poisson_2d"]

        self.problem_name = problem_name
        self.problem_kwargs = problem_kwargs
        self.fdm_solver_kwargs = fdm_solver_kwargs
        self.pinn_config = pinn_config
        self.verbose = verbose
        self.err_estimated = None
        self.u_fdm = None

        self.x = None
        self.t = None
        self.y = None
        self.pinn_prediction_time = 0

        self.problem, self.pinn_trainer, self.fdm_solver = get_problem_bundle(
            problem_name,
            problem_kwargs,
            fdm_solver_kwargs,
            pinn_config,
        )

    def _train_pinn(self):
        self.pinn_trainer.train()
        return self.pinn_trainer.run_time

    def _integrate_residual(self):
        err_approx = self.fdm_solver.residual_integration()
        return err_approx, self.fdm_solver.run_time

    def _fdm_solve(self):
        u_fdm = self.fdm_solver.solve()
        return u_fdm, self.fdm_solver.run_time

    def _create_grids(self):
        self.x = self.fdm_solver.x
        if self.is_time_dependent:
            self.t = self.fdm_solver.t
        elif self.is_2d:
            self.y = self.fdm_solver.y

    def _predict_pinn(self):
        start_time = time.time()
        if self.is_time_dependent:
            X, T = np.meshgrid(self.x, self.t)
            XT = np.column_stack((X.ravel(), T.ravel()))
            u_pinn = self.pinn_trainer.predict(XT).reshape(len(self.t), len(self.x))
        elif self.is_2d:
            X, Y = np.meshgrid(self.x, self.y)
            XY = np.column_stack((X.ravel(), Y.ravel()))
            u_pinn = self.pinn_trainer.predict(XY).reshape(X.shape)
        else:
            u_pinn = self.pinn_trainer.predict(self.x.reshape(-1, 1)).flatten()
        self.pinn_prediction_time = time.time() - start_time
        return u_pinn

    def _get_exact_solution(self):
        if self.is_time_dependent:
            X, T = np.meshgrid(self.x, self.t)
            u_exact = self.problem.exact_solution(X, T)
        elif self.is_2d:
            X, Y = np.meshgrid(self.x, self.y)
            u_exact = self.problem.exact_solution(X, Y)
        else:
            u_exact = self.problem.exact_solution(self.x)
        return u_exact

    def _get_fdm_error(self):
        u_exact = self._get_exact_solution()
        if self.is_time_dependent:
            X, T = np.meshgrid(self.x, self.t)
            u_fdm = self.u_fdm.reshape(len(self.t), len(self.x))
            fdm_error = u_fdm - u_exact
        elif self.is_2d:
            X, Y = np.meshgrid(self.x, self.y)
            u_fdm = self.u_fdm.reshape(X.shape)
            fdm_error = u_fdm - u_exact
        else:
            fdm_error = self.u_fdm.flatten() - u_exact
        return fdm_error

    def _get_residual(self):
        if self.is_time_dependent:
            X, T = np.meshgrid(self.x, self.t)
            residual = self.pinn_trainer.residual(
                np.column_stack((X.ravel(), T.ravel()))
            ).reshape(X.shape)
        elif self.is_2d:
            X, Y = np.meshgrid(self.x, self.y)
            residual = self.pinn_trainer.residual(
                np.column_stack((X.ravel(), Y.ravel()))
            ).reshape(X.shape)
        else:
            residual = self.pinn_trainer.residual(self.x.reshape(-1, 1)).flatten()
        return residual
    
    def _run(self):
        pinn_time = self._train_pinn()
        if self.verbose:
            print(f"PINN training run time: {pinn_time:.4f} seconds")

        self.err_estimated, fdm_residual_int_time = self._integrate_residual()
        if self.verbose:
            print(f"FDM error estimation run time: {fdm_residual_int_time:.4f} seconds")

        self.u_fdm, fdm_solver_solution_time = self._fdm_solve()
        fdm_solver_solution_time += self.pinn_prediction_time  # include time to predict PINN solution at FDM grid points
        if self.verbose:
            print(f"FDM solution run time: {fdm_solver_solution_time:.4f} seconds (including PINN prediction time: {self.pinn_prediction_time:.4f} seconds)")

        return {
            "pinn_time": pinn_time,
            "e_res": self.err_estimated,
            "e_fdm_time": fdm_solver_solution_time,
            "u_fdm": self.u_fdm,
            "e_res_time": fdm_residual_int_time,
        }

    def run_experiment(self):
        """
        run experiment and return a dict of results
        """
        # Run the experiment steps and collect results
        results = self._run()
        # recreate discretization grids for error analysis and visualization
        self._create_grids()
        # get approximated and exact solutions
        u_pinn = self._predict_pinn()
        u_exact = self._get_exact_solution()

        # compute/retrieve errors
        e_true = u_exact - u_pinn
        # fdm_error = u_exact - self.u_fdm
        # because: originally 
        # fdm_err = (u_fdm - u_pinn) - (u_exact - u_pinn) 
        #         = u_fdm - u_exact 
        e_fdm_vs_true = self._get_fdm_error()

        e_fdm = self.u_fdm - u_pinn
        # solved via FDM (our method)
        e_res = results["e_res"]

        # get PINN residual for visualization
        residual = self._get_residual()

        nx = self.fdm_solver.nx if hasattr(self.fdm_solver, "nx") else -1
        ny = self.fdm_solver.ny if hasattr(self.fdm_solver, "ny") else -1
        nt = self.fdm_solver.nt if hasattr(self.fdm_solver, "nt") else -1

        # return results
        return {
            "problem_name": self.problem_name,
            # pinn trainer config 
            **self.pinn_trainer.config.__dict__,
            # results from experiment
            **results,
            "e_true": e_true,
            "u_pinn": u_pinn,
            "u_true": u_exact,
            "e_fdm_vs_true": e_fdm_vs_true,
            "e_fdm": e_fdm,
            "e_res": e_res,
            # discretization info for error analysis and visualization
            "x": self.x,
            "t": self.t,
            "y": self.y,
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "stability_flag": self.fdm_solver.stability_flag,
            "residual": residual,
        }