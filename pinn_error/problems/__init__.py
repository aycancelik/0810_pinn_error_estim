from pinn_error.problems.drift_diffusion import DriftDiffusion
from pinn_error.problems.heat_1d import Heat1DProblemSineIC
from pinn_error.problems.poisson import Poisson1D, Poisson2D
from pinn_error.problems.wave import Wave1D

__all__ = [
    "DriftDiffusion",
    "Heat1DProblemSineIC",
    "Poisson1D",
    "Poisson2D",
    "Wave1D",
]
