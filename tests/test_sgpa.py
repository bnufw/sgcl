from copy import deepcopy

import pytest
import torch

from continual_learning.utils.proto import ProtoGenerator
from continual_learning.utils.sgpa import fit_sgpa, sgpa


def make_prototypes(means):
    prototypes = ProtoGenerator("cpu", {0: list(range(len(means)))})
    for label, mean in enumerate(means):
        covariance = torch.tensor([[2., 0.25], [0.25, 1.]], dtype=means.dtype)
        prototypes.prototype.append(mean.clone())
        prototypes.running_proto_variance.append(covariance)
        prototypes.class_label.append(label)
        prototypes.update_gaussian(label, mean, covariance)
    prototypes.running_proto = deepcopy(prototypes.prototype)
    return prototypes


def test_sgpa_prototype_state():
    means = torch.tensor([[2., 0.], [0., 2.], [1., 1.], [0., 0.]], dtype=torch.double)
    prototypes = make_prototypes(means)
    covariance = deepcopy(prototypes.running_proto_variance)
    gaussian_covariance = [prototypes.gaussians[i].covariance_matrix.clone() for i in range(4)]
    W = torch.tensor([[0., -2.], [3., 0.]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([0.5, -0.25], dtype=torch.double, requires_grad=True)
    basis = torch.tensor([[1.], [0.]], dtype=torch.double, requires_grad=True)
    diagnostics = sgpa(prototypes, W, b, basis)
    expected = torch.tensor([[2., 0.], [6.5, -0.25], [2.25, -0.625], [0.5, -0.25]], dtype=torch.double)
    torch.testing.assert_close(torch.stack(prototypes.prototype), expected)
    torch.testing.assert_close(torch.stack(prototypes.running_proto), expected)
    for i in range(4):
        torch.testing.assert_close(prototypes.gaussians[i].mean, expected[i])
        assert torch.equal(prototypes.running_proto_variance[i], covariance[i])
        assert torch.equal(prototypes.gaussians[i].covariance_matrix, gaussian_covariance[i])
        assert not prototypes.prototype[i].requires_grad
    assert diagnostics["stability_min"] == 0 and diagnostics["stability_max"] == 1
    assert diagnostics["stability_mean"] == pytest.approx(0.375)


@pytest.mark.parametrize("drift", [[0., 0.], [0.5, -0.25]])
def test_sgpa_translation(drift):
    means = torch.tensor([[2., 0.], [0., 2.], [1., 1.]], dtype=torch.double)
    prototypes = make_prototypes(means)
    old = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.double)
    delta = torch.tensor(drift, dtype=torch.double)
    W, b, mse = fit_sgpa(old, old + delta, 1e-3)
    sgpa(prototypes, W, b, torch.eye(2, dtype=torch.double)[:, :1])
    torch.testing.assert_close(torch.stack(prototypes.prototype), means + torch.tensor([[0.], [1.], [0.5]]) * delta)
    assert mse < 1e-18


def test_sgpa_zero_rank_and_missing_basis():
    means = torch.tensor([[1., 2.], [0., 0.]])
    prototypes = make_prototypes(means)
    W, b = torch.diag(torch.tensor([2., 3.])), torch.tensor([0.5, -1.])
    with pytest.raises(RuntimeError, match="unavailable"):
        sgpa(prototypes, W, b, None)
    assert torch.equal(torch.stack(prototypes.prototype), means)
    diagnostics = sgpa(prototypes, W, b, torch.empty(2, 0))
    torch.testing.assert_close(torch.stack(prototypes.prototype), means @ W + b)
    assert diagnostics["stable_rank"] == 0 and diagnostics["stability_max"] == 0


def test_sgpa_singular_features():
    old = torch.ones(16, 4)
    W, b, mse = fit_sgpa(old, old * 2 + 3, 1e-3)
    assert torch.equal(W, torch.eye(4))
    assert torch.equal(b, torch.full((4,), 4.))
    assert mse == 0
