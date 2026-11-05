"""SGPA."""
from copy import deepcopy

import torch


def fit_sgpa(old_features, new_features, ridge):
    old_mean = old_features.mean(dim=0)
    new_mean = new_features.mean(dim=0)
    old_centered = old_features - old_mean
    new_centered = new_features - new_mean
    eye = torch.eye(old_features.shape[1], device=old_features.device, dtype=old_features.dtype)
    lhs = old_centered.T @ old_centered + ridge * eye
    rhs = old_centered.T @ new_centered + ridge * eye
    W = torch.linalg.solve(lhs, rhs)
    b = new_mean - old_mean @ W
    mse = float(torch.nn.functional.mse_loss(old_features @ W + b, new_features).detach())
    return W, b, mse


@torch.no_grad()
def sgpa(prototypes, W, b, basis):
    if basis is None:
        raise RuntimeError("Previous-task Hessian is unavailable")
    W, b = W.detach().cpu(), b.detach().cpu()
    basis = basis.detach().to(W)
    means = torch.stack(prototypes.prototype).to(W)
    norm_squared = means.square().sum(dim=1).clamp_min(torch.finfo(means.dtype).eps)
    stability = ((means @ basis).square().sum(dim=1) / norm_squared).clamp(0, 1)
    displacement = (1 - stability).unsqueeze(1) * (means @ W + b - means)
    for i, (mean, covariance, label) in enumerate(zip(
            means + displacement, prototypes.running_proto_variance, prototypes.class_label)):
        prototypes.prototype[i] = mean
        prototypes.update_gaussian(label, mean, covariance)
    prototypes.running_proto = deepcopy(prototypes.prototype)
    return {
        "stability_min": float(stability.min()),
        "stability_mean": float(stability.mean()),
        "stability_max": float(stability.max()),
        "mean_update_norm": float(displacement.norm(dim=1).mean()),
        "stable_rank": basis.shape[1],
    }
