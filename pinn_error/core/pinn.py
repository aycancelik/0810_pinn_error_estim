"""PINN training module"""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import deepxde as dde
import numpy as np
import torch

from pinn_error.core.problem import BaseProblem

# Project root for model storage
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_ZOO_DIR = PROJECT_ROOT / "model_zoo"


@dataclass
class PINNConfig:
    """Configuration for PINN training"""

    # network config
    # need to wrap it in this weird field for whatever reason
    layers: List[int] = field(default_factory=lambda: [2, 20, 20, 20, 1])
    activation: str = "tanh"

    # n training iter
    # low number: for testing and error estimate for bad models
    n_iterations: int = 10

    # num points to train
    num_domain: int = 100
    # num points to eval loss (residual) during train
    num_test: int = 1000
    
    num_initial: int = 100 # num points to train initial condition
    num_boundary: int = 100 # num points to train boundary condition

    # constraint mode: how IC/BC are enforced during training
    #   "hard"      -> both IC and BC hard-constrained via output_transform (baseline)
    #   "soft_ic"   -> BC hard-constrained via output_transform_bc_only, IC learned via soft loss
    #   "soft_full" -> no hard constraints at all; both IC and BC learned via soft loss
    # Note: time-independent problems (e.g. Poisson) have no IC, so "soft_ic"
    # is not a valid choice for them -- use "hard" or "soft_full" instead.
    constraint_mode: str = "hard"

    # restore best model after training
    # based on validation loss
    # necessary for SI model to show collapse well
    restore_best: bool = False

    # random seed for reproducibility
    seed: Optional[int] = None

    # model caching
    use_cache: bool = False  # If True, load cached model if available --for soft constraints we set it to false 
    cache_dir: Optional[str] = None  # Override default model_zoo directory



class PINNTrainer:
    """PINN trainer class"""

    def __init__(
        self,
        problem: BaseProblem,
        config: PINNConfig,
    ):
        self.problem = problem
        self.config = config

        self.model = None
        self.network = None
        self.callbacks = []
        self._run_time = 0.0
        self._loaded_from_cache = False

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = f"{self.tmp_dir.name}/pinn_model.ckpt"

        if self.config.seed is None:
            self.config.seed = 42
        # Set random seed for reproducibility
        dde.config.set_random_seed(self.config.seed)

        # Setup cache directory
        self.cache_dir = (
            Path(self.config.cache_dir) if self.config.cache_dir else MODEL_ZOO_DIR
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique model identifier
        self._model_hash = self._compute_model_hash()
        self._cache_path = self.cache_dir / f"{self._model_hash}.pt"
        self._meta_path = self.cache_dir / f"{self._model_hash}.json"

        self.geom = self.problem.create_geometry()
        self._init_model()

    def _bc_target(self, X: np.ndarray) -> np.ndarray:
        """Target values for a soft Dirichlet BC loss term.

        Reuses `exact_solution` (the same convention already used in
        FDMSolverDriftDiffusion._get_boundary_values) rather than a separate
        `boundary_condition` method, since for these manufactured problems the
        prescribed boundary data and the exact solution restricted to the
        boundary are the same thing.
        """
        if self.problem.domain.is_time_dependent:
            return self.problem.exact_solution(X[:, 0:1], X[:, 1:2])
        elif self.problem.domain.spatial_dim == 2:
            return self.problem.exact_solution(X[:, 0:1], X[:, 1:2])
        else:
            return self.problem.exact_solution(X[:, 0:1])

    def _init_model(self):
        """Initializes the model etc"""

        is_time_dependent = self.problem.domain.is_time_dependent
        mode = self.config.constraint_mode

        if mode not in ("hard", "soft_ic", "soft_full"):
            raise ValueError(
                f"Unknown constraint_mode {mode!r}; expected 'hard', 'soft_ic', or 'soft_full'."
            )
        if mode == "soft_ic" and not is_time_dependent:
            raise ValueError(
                f"{self.problem.__class__.__name__} is time-independent (no initial "
                "condition), so constraint_mode='soft_ic' doesn't apply. Use 'hard' "
                "or 'soft_full' instead."
            )

        self.network = dde.nn.FNN(
            self.config.layers, self.config.activation, "Glorot normal"
        )

        if mode == "hard":
            # both IC and BC hard-constrained
            transform = self.problem.output_transform
        elif mode == "soft_ic":
            # BC hard-constrained, IC learned via soft loss
            transform = self.problem.output_transform_bc_only
        else:  # soft_full
            # no hard constraints at all; identity output transform
            transform = lambda x, u: u
        self.network.apply_output_transform(transform)

        # BC needs a soft loss term only when it isn't hard-constrained
        # structurally (i.e. only in "soft_full" mode -- "hard" and "soft_ic"
        # both bake BC into the network via output_transform above)
        needs_soft_bc = mode == "soft_full"

        # Second-order-in-time PDEs (wave) need the velocity IC du/dt(x,0)
        # supervised too. Under "hard" the output transform's t**2 factor
        # enforces it structurally; under soft_ic/soft_full nothing does, so
        # without this term the problem is under-determined at t=0 and the
        # network is free to start with an arbitrary initial velocity.
        needs_soft_velocity_ic = mode in ("soft_ic", "soft_full") and getattr(
            self.problem, "is_second_order_in_time", False
        )

        if is_time_dependent:
            ic_bcs = [
                dde.icbc.IC(self.geom, self.problem.initial_condition, lambda _, on_initial: on_initial)
            ]
            if needs_soft_bc:
                ic_bcs.append(
                    dde.icbc.DirichletBC(self.geom, self._bc_target, lambda _, on_boundary: on_boundary)
                )
            if needs_soft_velocity_ic:
                # NOTE: dde.icbc.OperatorBC can't be used here -- for a
                # GeometryXTime it filters via geom.on_boundary(), which
                # inspects only the SPATIAL columns and therefore selects the
                # spatial boundary at all times rather than t=0. So we supply
                # explicit t=0 points via PointSetOperatorBC instead.
                X_initial = self.geom.random_initial_points(
                    self.config.num_initial, random="Hammersley"
                )
                v_initial = self.problem.initial_velocity(X_initial)
                # index of the time column (x columns are spatial, then t)
                t_idx = self.problem.domain.spatial_dim
                ic_bcs.append(
                    dde.icbc.PointSetOperatorBC(
                        X_initial,
                        v_initial,
                        lambda inputs, outputs, X, _j=t_idx: dde.grad.jacobian(
                            outputs, inputs, i=0, j=_j
                        ),
                    )
                )
            data = dde.data.TimePDE(
                geometryxtime=self.geom,
                pde=self.problem.pde,
                ic_bcs=ic_bcs,
                num_domain=self.config.num_domain,
                num_test=self.config.num_test,
                num_initial=self.config.num_initial,
                num_boundary=self.config.num_boundary,
            )
        else:
            bcs = []
            if needs_soft_bc:
                bcs.append(
                    dde.icbc.DirichletBC(self.geom, self._bc_target, lambda _, on_boundary: on_boundary)
                )
            data = dde.data.PDE(
                geometry=self.geom,
                pde=self.problem.pde,
                bcs=bcs,
                num_domain=self.config.num_domain,
                num_test=self.config.num_test,
                num_boundary=self.config.num_boundary if needs_soft_bc else 0,
            )

        self.model = dde.Model(data, self.network)

        self.callbacks = []
        if self.config.restore_best:
            self.callbacks.append(
                dde.callbacks.ModelCheckpoint(
                    self.checkpoint_path,
                    save_better_only=True,
                    period=1,
                )
            )

        self.model.compile("adam", lr=1e-3)

    def _compute_model_hash(self) -> str:
        """Compute a unique hash based on config and problem parameters."""
        # Collect all relevant parameters
        config_dict = {
            "layers": self.config.layers,
            "activation": self.config.activation,
            "n_iterations": self.config.n_iterations,
            "num_domain": self.config.num_domain,
            "num_test": self.config.num_test,
            "seed": self.config.seed,
        }

        # Problem parameters
        problem_dict = {
            "class": self.problem.__class__.__name__,
            "spatial_bounds": self.problem.spatial_bounds,
            "temporal_bounds": getattr(self.problem, "temporal_bounds", None),
        }

        # Add problem-specific attributes
        for attr in [
            "diffusivity",
            "mode",
            "propagation_speed",
            "wave_speed",
            "velocity_x",
            "frequency",
            "phase_shift",
            "initial_concentration",
        ]:
            if hasattr(self.problem, attr):
                problem_dict[attr] = getattr(self.problem, attr)

        # Combine and hash
        combined = {"config": config_dict, "problem": problem_dict}
        hash_str = json.dumps(combined, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def _get_model_metadata(self) -> dict:
        """Get metadata about the trained model."""
        return {
            "model_hash": self._model_hash,
            "config": {
                "layers": self.config.layers,
                "activation": self.config.activation,
                "n_iterations": self.config.n_iterations,
                "num_domain": self.config.num_domain,
                "num_test": self.config.num_test,
                "seed": self.config.seed,
            },
            "problem": {
                "class": self.problem.__class__.__name__,
                "spatial_bounds": self.problem.spatial_bounds,
                "temporal_bounds": getattr(self.problem, "temporal_bounds", None),
            },
            "training_time": self._run_time,
        }

    def _save_to_cache(self):
        """Save the trained model to cache."""
        # Save model weights
        torch.save(self.network.state_dict(), self._cache_path)

        # Save metadata
        metadata = self._get_model_metadata()
        with open(self._meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to cache: {self._cache_path}")

    def _load_from_cache(self) -> bool:
        """Try to load model from cache. Returns True if successful."""
        if not self._cache_path.exists():
            return False

        try:
            # Load model weights
            state_dict = torch.load(self._cache_path, weights_only=True)
            self.network.load_state_dict(state_dict)

            # Load metadata for training time
            if self._meta_path.exists():
                with open(self._meta_path, "r") as f:
                    metadata = json.load(f)
                    self._run_time = metadata.get("training_time", 0.0)

            self._loaded_from_cache = True
            print(f"Model loaded from cache: {self._cache_path}")
            return True
        except Exception as e:
            print(f"Failed to load cached model: {e}")
            return False

    def train(self, force_retrain: bool = False):
        """Trains the PINN model.

        Args:
            force_retrain: If True, train even if cached model exists.
        """
        # Try to load from cache
        if self.config.use_cache and not force_retrain:
            if self._load_from_cache():
                return

        start_time = time.time()

        self.model.train(
            iterations=self.config.n_iterations,
            callbacks=self.callbacks,
        )

        end_time = time.time()
        self._run_time = end_time - start_time

        if self.config.restore_best:
            self.model.restore(self._get_best_model_path())

        # Save to cache
        if self.config.use_cache:
            self._save_to_cache()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts using the trained PINN model

        Args:
            X: Input data for prediction

        Returns:
            Predicted output from the PINN model
        """
        return self.model.predict(X)

    def residual(
            self, 
            X: np.ndarray | torch.Tensor
        ) -> np.ndarray | torch.Tensor:
        """Computes the PDE residual at given points

        Args:
            X: Input data points where residual is computed

        Returns:
            Residual values at the input points
        """
        residual = None
        if isinstance(X, np.ndarray):
            residual = self.model.predict(
                X,
                operator=self.problem.pde,
            )
        else:
            out = self.network(X)
            residual = self.problem.pde(X, out)
        return residual

    def time_derivative(
            self,
            X: np.ndarray | torch.Tensor
        ) -> np.ndarray | torch.Tensor:
        """Computes du/dt of the PINN prediction at given points.

        Needed for second-order-in-time problems (wave), where the FDM error
        integration's first step depends on the initial error velocity
        de/dt(x,0) = u_t(x,0) - u_hat_t(x,0).

        Args:
            X: Input data points, shape (N, dim) with the time column last

        Returns:
            du/dt values at the input points
        """
        # index of the time column (x columns are spatial, then t)
        t_idx = self.problem.domain.spatial_dim

        def _dudt(x, u):
            return dde.grad.jacobian(u, x, i=0, j=t_idx)

        if isinstance(X, np.ndarray):
            return self.model.predict(X, operator=_dudt)
        else:
            out = self.network(X)
            return _dudt(X, out)

    @property
    def run_time(self) -> float:
        """Returns the training time in seconds"""
        return self._run_time

    @property
    def loaded_from_cache(self) -> bool:
        """Returns True if model was loaded from cache instead of training."""
        return self._loaded_from_cache

    @property
    def model_hash(self) -> str:
        """Returns the unique hash identifier for this model configuration."""
        return self._model_hash

    @property
    def cache_path(self) -> Path:
        """Returns the cache path for this model."""
        return self._cache_path

    def clear_cache(self):
        """Remove the cached model for this configuration."""
        if self._cache_path.exists():
            self._cache_path.unlink()
            print(f"Removed cached model: {self._cache_path}")
        if self._meta_path.exists():
            self._meta_path.unlink()
            print(f"Removed cached metadata: {self._meta_path}")

    def _get_best_model_path(self) -> str:
        """Get path to best model checkpoint in temporary directory."""
        model_files = os.listdir(self.tmp_dir.name)
        model_steps = [
            int(f.split("-")[1].split(".pt")[0])
            for f in model_files
            if f.startswith("pinn_model.ckpt-") and f.endswith(".pt")
        ]
        best_step = np.max(model_steps)
        return f"{self.tmp_dir.name}/pinn_model.ckpt-{best_step}.pt"