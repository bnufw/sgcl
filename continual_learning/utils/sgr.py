"""SGR."""
import itertools

import torch


class FeatureHessian:
    def __init__(self, device):
        self.device = device
        self.n_samples = 0

    @torch.no_grad()
    def compute(self, model, trn_loader):
        model.eval()
        W = torch.cat([head[0].weight for head in model.heads], dim=0)
        classes, dimensions = W.shape
        batches = len(trn_loader.dataset) // trn_loader.batch_size
        if batches == 0:
            raise ValueError("Hessian estimation requires at least one complete batch")
        Q, singular_values, Vh = torch.linalg.svd(W.T, full_matrices=False)
        R = singular_values[:, None] * Vh
        mean_covariance = W.new_zeros(classes, classes)
        self.n_samples = 0
        for images, _ in itertools.islice(trn_loader, batches):
            features = model.backbone(images.to(self.device))
            p = torch.cat([head(features) for head in model.heads], dim=1).softmax(dim=1)
            mean_covariance += torch.diag(p.sum(dim=0)) - p.T @ p
            self.n_samples += images.shape[0]
        if self.n_samples == 0:
            raise ValueError("Hessian estimation received no samples")
        mean_covariance /= self.n_samples
        reduced = (R @ mean_covariance @ R.T).double().cpu()
        reduced = (reduced + reduced.T) * 0.5
        if not torch.isfinite(reduced).all():
            raise ValueError("Hessian contains non-finite values")
        values, vectors = torch.linalg.eigh(reduced)
        values, vectors = values.flip(0), vectors.flip(1)
        rank = min(int((values > 1e-7).sum()), classes - 1, dimensions)
        basis = Q @ vectors[:, :rank].to(W)
        values = values[:rank].to(W)
        print("Hessian: {} samples, {} stable directions".format(self.n_samples, rank))
        return basis.detach(), values.detach()


def sgr_losses(features, old_features, basis, values):
    """Unweighted losses: sum over directions, mean over samples."""
    delta = features - old_features.detach()
    projected = delta @ basis
    stable = (projected.square() * values).sum(dim=1).mean()
    residual = delta - projected @ basis.T
    plastic = residual.square().sum(dim=1).mean()
    return stable, plastic
