from copy import deepcopy

import torch
from torch.distributions import MultivariateNormal


class ProtoGenerator:
    def __init__(self, device, task_dict):
        self.device = device
        self.task_dict = task_dict
        self.prototype = []
        self.class_label = []
        self.running_proto = None
        self.running_proto_variance = []
        self.gaussians = {}
        self.class_stats = {}

    @torch.no_grad()
    def compute(self, model, loader, current_task):
        model.eval()
        features_list, label_list = [], []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.type(dtype=torch.int64).to(self.device)
            _, features = model(images)
            label_list.append(labels.cpu())
            features_list.append(features.cpu())
        labels = torch.cat(label_list)
        features = torch.cat(features_list)
        for label in self.task_dict[current_task]:
            class_features = features[labels == label]
            self.class_stats[label] = class_features.shape[0]
            mean = class_features.mean(dim=0)
            covariance = torch.cov(class_features.T)
            self.running_proto_variance.append(covariance)
            self.prototype.append(mean)
            self.class_label.append(label)
            self.update_gaussian(label, mean, covariance)
        self.running_proto = deepcopy(self.prototype)

    def update_gaussian(self, label, mean, covariance):
        self.gaussians[label] = MultivariateNormal(
            mean.cpu(), covariance_matrix=covariance + 1e-5 * torch.eye(covariance.shape[0]).cpu())
