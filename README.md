# Building Trust in PINNs: Error Estimation through Finite Difference Methods

This repository accompanies the paper "[Building Trust in PINNs: Error Estimation through Finite Difference Methods](https://arxiv.org/abs/2603.15526)" (submitted to the XAI World Conference 2026). 

We propose a post-hoc method that estimates the pointwise error of physics-informed neural networks by solving the associated error equation via finite difference methods, requiring no knowledge of the true solution.

## Quickstart

Just install the package with pip (make sure to run from the project root directory)
```bash
pip install .
```

If you wish to work with Jupyter use this instead
```bash
pip install ".[jupyter]"
```

## Quickstart using `uv`

I recommend using [uv](https://uv.sh) for environment management and quick pip installation.

### 1. Create environment

Go to the project root and create a virtual environment

```bash
uv venv --python 3.13
```

### 2. Activate the environment

```bash
source .venv/bin/activate
```

### 3. Install the package

```bash
uv pip install .
```

Alternatively with jupyter dependencies:

```bash
uv pip install ".[jupyter]"
```

## Running an example

You can execute individual examples using the experiment runner script

```bash
python examples/run_experiment.py --problem heat --frequency 2 --diffusivity 0.05 --nx 16 --nt 16
```

For accepted arguments please run 
```bash
python examples/run_experiment.py --help
``` 

Currently implemented problems include
- `heat` (1D heat equation) 
    - PDE: $\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$
    - IC: $u(x,0) = \sin(n \pi \frac{x}{x_{\mathrm{max}}})$
    - BCs: 
        - $u(x_{\mathrm{min}}, t) = 0$ (Dirichlet)
        - $u(x_{\mathrm{max}}, t) = 0$ (Dirichlet)
    - named parameters: 
        - `frequency` $n$
        - `diffusivity` $\alpha$ 
- `wave` (1D wave equation)
    - PDE: $\frac{\partial^2 u}{\partial t^2} - c^2 \frac{\partial^2 u}{\partial x^2} = 0$
    - IC: $u(x,0)=\sin(n \pi \frac{x}{x_{\mathrm{max}}})$
    - BCs:
        - $u(x_{\mathrm{min}}, t) = 0$ (Dirichlet)
        - $u(x_{\mathrm{max}}, t) = 0$ (Dirichlet)
        - $\frac{\partial u}{\partial t}(x,0) = 0$ (Neumann - initial velocity)
    - named parameters:
        - `frequency` $n$
        - `propagation_speed` $c$
- `drift_diffusion` (1D convection-diffusion equation)
    - PDE: $\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} - v \frac{\partial u}{\partial x} = 0$
    - IC: $u(x,0)= A + \sin(n \pi \frac{x}{x_{\mathrm{max}}} + \varphi)$
    - BCs:
        - $u(x_{\mathrm{min}},t) = (x_{\mathrm{max}},t)$ (Periodic)
    - named parameters:
        - `frequency` $n$
        - `diffusivity` $\alpha$ 
        - `velocity_x` $v$
        - `initial_concentration` $A$
        - `phase_shift` $\varphi$ 
- `poisson_1d` (1D Poisson equation - steady)
    - PDE: $- \frac{\partial^2 u}{\partial x^2} - f(x) = 0$
    - with source term: $f(x) = (\frac{\pi}{x_{\mathrm{max}}})^2 \cdot \sin(\pi \frac{x}{x_{\mathrm{max}}})$
    - BCs:
        - $u(x_\mathrm{min})=0$ (Dirichlet)
        - $u(x_\mathrm{max})=0$ (Dirichlet)
- `poisson_2d` (2D Poisson equation - steady)
    - PDE: $\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - f(x) = 0$
    - with source term: $f(x) = ((\frac{\pi}{x_{\mathrm{max}}})^2 + (\frac{\pi}{y_{\mathrm{max}}})^2) \cdot \sin(\pi \frac{x}{x_{\mathrm{max}}}) \sin(\pi \frac{y}{y_{\mathrm{max}}})$
    - BCs:
        - $u(x_\mathrm{min}, y)=0$ (Dirichlet)
        - $u(x_\mathrm{max}, y)=0$ (Dirichlet)
        - $u(x, y_\mathrm{min})=0$ (Dirichlet)
        - $u(x, y_\mathrm{max})=0$ (Dirichlet)

# Citing

Please use the following citation when referencing this work in literature:

```bibtex
@misc{krasowski2026buildingtrustpinnserror,
      title={Building Trust in PINNs: Error Estimation through Finite Difference Methods}, 
      author={Aleksander Krasowski and René P. Klausen and Aycan Celik and Sebastian Lapuschkin and Wojciech Samek and Jonas Naujoks},
      year={2026},
      eprint={2603.15526},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.15526}, 
}
```