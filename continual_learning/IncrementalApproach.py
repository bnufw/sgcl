import torch

from continual_learning.utils.OptimizerManager import OptimizerManager


class IncrementalApproach:
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        self.device = device
        self.out_path = out_path
        self.class_per_task = class_per_task
        self.task_dict = task_dict
        self.n_class_first_task = args.n_class_first_task
        self.backbone = args.backbone
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.image_size = {"cifar100": 32, "tiny-imagenet": 64, "imagenet-subset": 224}[args.dataset]
        self.epochs_first_task = args.epochs_first_task
        self.epochs_next_task = args.epochs_next_task
        self.rotation = True
        self.auxiliary_classifier = None
        self.optimizer_manager = OptimizerManager(self.dataset, self.rotation)

    def pre_train(self, task_id):
        self.optimizer, self.scheduler = self.optimizer_manager.get_optimizer(
            task_id, self.model, self.auxiliary_classifier)
        self.total_epochs = self.epochs_first_task if task_id == 0 else self.epochs_next_task

    def rescale_targets(self, targets, task_id):
        offset = (task_id - 1) * self.class_per_task + self.n_class_first_task if task_id > 0 else 0
        return targets - offset

    def tag_probabilities(self, outputs):
        return torch.softmax(torch.cat(list(outputs.values()), dim=1), dim=1)

    def taw_probabilities(self, outputs, head_id):
        return torch.softmax(outputs[head_id], dim=1)
