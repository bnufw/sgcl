import os
import random
import string
from copy import deepcopy

import numpy as np
import torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def experiment_folder(root_path, approach_name="sgcl"):
    name = approach_name + "_" + "".join(random.choices(string.ascii_letters + string.digits, k=8))
    path = os.path.join(root_path, name)
    os.mkdir(path)
    return path, name


def get_task_dict(n_task, total_classes):
    classes_per_task = total_classes // n_task
    return {task: list(range(task * classes_per_task, (task + 1) * classes_per_task))
            for task in range(n_task)}


def remap_targets(train_set, test_set, total_classes):
    labels = list(range(total_classes))
    shuffled = deepcopy(labels)
    random.shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))
    for dataset in (train_set, test_set):
        for i in range(len(dataset.targets)):
            dataset.targets[i] = mapping[dataset.targets[i]]
        for key in dataset.class_to_idx:
            dataset.class_to_idx[key] = mapping[dataset.class_to_idx[key]]
    return train_set, test_set, mapping
