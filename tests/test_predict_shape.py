import numpy as np

from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.problems.heat_1d import Heat1DProblemSineIC


class _DummyPredictModel:
    def predict(self, X):
        n = len(X)
        return np.arange(n, dtype=float).reshape(n, 1)


def test_predict_flattens_singleton_output_axis():
    problem = Heat1DProblemSineIC(
        x_min=0.0,
        x_max=1.0,
        t_max=0.1,
        diffusivity=0.01,
        frequency=1,
    )
    config = PINNConfig(
        n_iterations=1,
        num_domain=8,
        num_test=8,
        num_boundary=4,
        num_initial=4,
        use_cache=False,
    )
    trainer = PINNTrainer(problem, config)
    trainer.model = _DummyPredictModel()

    X = np.column_stack((np.linspace(0.0, 1.0, 8), np.zeros(8)))
    y = trainer.predict(X)

    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.shape == (8,)
