import torch
from sklearn.metrics import accuracy_score


class MetricEvaluator:
    def __init__(self):
        self.taw_probabilities, self.taw_labels = [], []
        self.tag_probabilities, self.tag_labels = [], []

    def update(self, original_labels, labels, tag_probabilities, taw_probabilities):
        self.taw_probabilities.append(taw_probabilities)
        self.taw_labels.append(labels)
        self.tag_probabilities.append(tag_probabilities)
        self.tag_labels.append(original_labels)

    def get(self):
        taw_probabilities = torch.cat(self.taw_probabilities)
        taw_labels = torch.cat(self.taw_labels).cpu().numpy()
        tag_probabilities = torch.cat(self.tag_probabilities)
        tag_labels = torch.cat(self.tag_labels).cpu().numpy()
        taw = accuracy_score(taw_labels, torch.max(taw_probabilities, dim=1)[1].cpu().numpy())
        tag = accuracy_score(tag_labels, torch.max(tag_probabilities, dim=1)[1].cpu().numpy())
        return taw, tag
