from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import gpytorch
except ImportError:  # optional dependency
    torch = None
    gpytorch = None

_DEPS_MSG = (
    "GPyTorch backend requires `torch` and `gpytorch`. Install with:\n"
    "pip install torch gpytorch"
)


@dataclass
class GPyTorchConfig:
    epochs: int = 200
    lr: float = 0.1
    use_cuda: bool = False
    jitter: float = 1e-4


def _check_deps() -> None:
    if torch is None or gpytorch is None:
        raise ImportError(_DEPS_MSG)


def _device(use_cuda: bool) -> "torch.device":
    _check_deps()
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(x: np.ndarray, device: "torch.device") -> "torch.Tensor":
    _check_deps()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _build_exact_gp_model(train_x, train_y, likelihood):
    """Build an ExactGP model. Defined as a function to avoid importing gpytorch at module level."""
    _check_deps()

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return ExactGPModel(train_x, train_y, likelihood)


def train_exact_gp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: GPyTorchConfig,
) -> tuple["gpytorch.models.ExactGP", "gpytorch.likelihoods.GaussianLikelihood", "torch.device"]:
    _check_deps()
    device = _device(config.use_cuda)
    train_x_t = _to_tensor(train_x, device)
    train_y_t = _to_tensor(train_y, device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = _build_exact_gp_model(train_x_t, train_y_t, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(config.epochs):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()

    return model, likelihood, device


def predict(
    model: "gpytorch.models.ExactGP",
    likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    test_x: np.ndarray,
    device: "torch.device",
    jitter: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    _check_deps()
    model.eval()
    likelihood.eval()

    test_x_t = _to_tensor(test_x, device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(jitter):
        preds = likelihood(model(test_x_t))
        mean = preds.mean.detach().cpu().numpy()
        std = preds.stddev.detach().cpu().numpy()
    return mean, std
