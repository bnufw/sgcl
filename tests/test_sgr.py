import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.sgr import FeatureHessian, sgr_losses


class SmallModel(BaseModel):
    def __init__(self, classes=3, dimensions=6):
        nn.Module.__init__(self)
        self.backbone = nn.Linear(dimensions, dimensions, bias=False)
        self.backbone.feature_space_size = dimensions
        self.heads = nn.ModuleList()
        if classes:
            self.add_classification_head(classes)


@pytest.mark.parametrize("classes,dimensions", [(3, 6), (9, 4)])
def test_hessian_matches_dense_matrix(classes, dimensions):
    torch.manual_seed(13)
    model = SmallModel(classes, dimensions)
    images = torch.randn(19, dimensions)
    loader = DataLoader(TensorDataset(images, torch.zeros(19)), batch_size=8)
    computer = FeatureHessian("cpu")
    basis, values = computer.compute(model, loader)
    with torch.no_grad():
        features = model.backbone(images[:16])
        W = model.heads[0][0].weight
        probabilities = model.heads[0](features).softmax(dim=1)
        C = (torch.diag(probabilities.sum(0)) - probabilities.T @ probabilities) / 16
        expected = W.T @ C @ W
    torch.testing.assert_close((basis * values) @ basis.T, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(basis.T @ basis, torch.eye(len(values)), atol=1e-6, rtol=1e-5)
    assert computer.n_samples == 16
    assert len(values) <= min(classes - 1, dimensions)
    assert not basis.requires_grad and not values.requires_grad


def test_sgr_losses_and_gradients():
    torch.manual_seed(2)
    basis = torch.linalg.qr(torch.randn(6, 3, dtype=torch.double)).Q
    values = torch.tensor([0.8, 0.3, 0.02], dtype=torch.double)
    features = torch.randn(7, 6, dtype=torch.double, requires_grad=True)
    old = torch.randn_like(features, requires_grad=True)
    stable, plastic = sgr_losses(features, old, basis, values)
    loss = 5 * stable + 0.03 * plastic
    matrix = 5 * (basis * values) @ basis.T + 0.03 * (torch.eye(6) - basis @ basis.T)
    delta = features - old.detach()
    torch.testing.assert_close(loss, (delta * (delta @ matrix)).sum(1).mean())
    loss.backward()
    torch.testing.assert_close(features.grad, 2 * delta.detach() @ matrix / len(features))
    assert old.grad is None


def test_zero_rank_and_incomplete_data():
    model = SmallModel()
    with torch.no_grad():
        model.heads[0][0].weight.zero_()
    loader = DataLoader(TensorDataset(torch.randn(8, 6), torch.zeros(8)), batch_size=8)
    basis, values = FeatureHessian("cpu").compute(model, loader)
    features = torch.randn(4, 6)
    stable, plastic = sgr_losses(features, torch.zeros_like(features), basis, values)
    assert basis.shape == (6, 0) and stable == 0
    torch.testing.assert_close(plastic, features.square().sum(1).mean())
    short_loader = DataLoader(TensorDataset(torch.randn(3, 6), torch.zeros(3)), batch_size=8)
    with pytest.raises(ValueError, match="complete batch"):
        FeatureHessian("cpu").compute(model, short_loader)
