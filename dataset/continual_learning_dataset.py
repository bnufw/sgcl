import numpy as np
from torch.utils.data import Subset


class ContinualLearningDataset:
    def __init__(self, dataset, task_dictionary, total_classes, train=True):
        self.dataset = dataset
        self.task_dictionary = task_dictionary
        self.total_classes = total_classes
        self.train = train

    def collect(self):
        indices = [[] for _ in self.task_dictionary]
        targets = np.array(self.dataset.targets)
        per_class = len(self.dataset) // self.total_classes
        for label in range(self.total_classes):
            selected = np.where(targets == label)[0]
            if self.train:
                selected = selected[:per_class]
            for task_id, classes in self.task_dictionary.items():
                if label in classes:
                    indices[task_id].extend(list(selected))
        return [Subset(self.dataset, ids) for ids in indices], [len(ids) for ids in indices]
