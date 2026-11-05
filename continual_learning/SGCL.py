from copy import deepcopy
from time import time

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm import tqdm

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.proto import ProtoGenerator
from continual_learning.utils.sgpa import fit_sgpa, sgpa
from continual_learning.utils.sgr import FeatureHessian, sgr_losses
from continual_learning.utils.training_utils import compute_rotations, get_old_new_features


class SGCL(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        super().__init__(args, device, out_path, class_per_task, task_dict)
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.old_model = None
        self.lambda_s, self.lambda_p = args.lambda_s, args.lambda_p
        self.hessian_basis = self.hessian_values = None
        self.sgpa_ridge = args.sgpa_ridge
        self.ca_batch_size = args.ca_batch_size
        self.ca_epochs = args.ca_epochs
        self.ca_lr = args.ca_lr
        self.proto_generator = ProtoGenerator(device, task_dict)
        print("SGCL: {}, {} tasks, epochs {}/{}, batch size {}".format(
            self.dataset, len(task_dict), self.epochs_first_task, self.epochs_next_task, self.batch_size))
        print("CA: epochs={}, batch size={}, lr={}".format(
            self.ca_epochs, self.ca_batch_size, self.ca_lr))

    def pre_train(self, task_id):
        if task_id == 0 and self.rotation:
            self.auxiliary_classifier = nn.Linear(self.model.get_feat_size(), len(self.task_dict[task_id]) * 3)
            self.auxiliary_classifier.to(self.device)
        else:
            self.auxiliary_classifier = None
            self.model.freeze_bn()
        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        super().pre_train(task_id)

    def sgr_loss(self, features, old_features):
        if self.hessian_basis is None or self.hessian_values is None:
            raise RuntimeError("Previous-task Hessian is unavailable")
        stable, plastic = sgr_losses(features, old_features, self.hessian_basis, self.hessian_values)
        return self.lambda_s * stable + self.lambda_p * plastic

    def train_criterion(self, outputs, targets, task_id, features, old_features):
        regularizer = self.sgr_loss(features, old_features) if task_id > 0 else 0
        classification = nn.functional.cross_entropy(outputs[task_id], self.rescale_targets(targets, task_id))
        return classification, regularizer

    def train(self, task_id, trn_loader, epoch, epochs):
        if task_id == 0:
            self.model.train()
        else:
            self.model.eval()
        if task_id == 0 and self.rotation:
            self.auxiliary_classifier.train()
        start = time()
        for images, targets in tqdm(trn_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            if task_id == 0 and self.rotation:
                images_rot, targets_rot = compute_rotations(images, self.image_size, self.task_dict, targets, task_id)
                images = torch.cat([images, images_rot], dim=0)
                targets = torch.cat([targets, targets_rot], dim=0)
            if task_id > 0:
                with torch.no_grad():
                    _, old_features = self.old_model(images)
            else:
                old_features = None
            outputs, features = self.model(images)
            if task_id == 0 and self.rotation:
                outputs[task_id] = torch.cat([outputs[task_id], self.auxiliary_classifier(features)], dim=1)
            classification, regularizer = self.train_criterion(outputs, targets, task_id, features, old_features)
            loss = classification + regularizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("Task {}, epoch {}/{}, elapsed {:.1f}s".format(task_id, epoch, epochs, time() - start))

    def eval_criterion(self, outputs, targets, task_id, verbose):
        if verbose:
            classification = nn.functional.cross_entropy(outputs[task_id], self.rescale_targets(targets, task_id))
        else:
            classification = nn.functional.cross_entropy(torch.cat(list(outputs.values()), dim=1), targets)
        return classification

    @torch.no_grad()
    def eval(self, current_training_task, test_id, loader, verbose):
        metric_evaluator = MetricEvaluator()
        classification_loss, n_samples = 0, 0
        self.model.eval()
        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.type(dtype=torch.int64).to(self.device)
            batch_size = images.shape[0]
            outputs, _ = self.model(images)
            classification = self.eval_criterion(outputs, targets, current_training_task, verbose)
            classification_loss += classification * batch_size
            n_samples += batch_size
            metric_evaluator.update(targets, self.rescale_targets(targets, test_id),
                                    self.tag_probabilities(outputs), self.taw_probabilities(outputs, test_id))
        if verbose:
            print("Classification loss: {}".format(classification_loss / n_samples))
        else:
            return metric_evaluator.get()

    def calibrate_classifier(self, trn_loader):
        model = self.model
        model.freeze_backbone()
        model.eval()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([p for p in model.heads.parameters() if p.requires_grad],
                                    lr=self.ca_lr, weight_decay=5e-4, momentum=0.9)
        for _ in range(self.ca_epochs):
            current_features, current_targets = self.get_feature(trn_loader)
            features_dataset = TensorDataset(current_features, current_targets)
            prototypes, labels = self.generate_sample_from_proto()
            prototypes_dataset = TensorDataset(prototypes, labels)
            complete_dataset = ConcatDataset([features_dataset, prototypes_dataset])
            epoch_loader = DataLoader(complete_dataset, batch_size=self.ca_batch_size, shuffle=True)
            for features, targets in epoch_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = torch.cat([head(features) for head in model.heads], dim=1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.unfreeze_backbone()
        model.freeze_bn()

    def post_train(self, task_id, trn_loader):
        with torch.no_grad():
            if task_id > 0:
                new_features, old_features = get_old_new_features(self.model, self.old_model, trn_loader, self.device)
                W, b, _ = fit_sgpa(old_features, new_features, self.sgpa_ridge)
                sgpa(self.proto_generator, W, b, self.hessian_basis)
        if task_id > 0:
            print("Classifier calibration")
            self.calibrate_classifier(trn_loader)
        computer = FeatureHessian(self.device)
        self.hessian_basis, self.hessian_values = computer.compute(self.model, deepcopy(trn_loader))
        self.proto_generator.compute(self.model, deepcopy(trn_loader), task_id)

    @torch.no_grad()
    def generate_sample_from_proto(self):
        samples, labels = [], []
        for label, gaussian in self.proto_generator.gaussians.items():
            count = self.proto_generator.class_stats[label]
            current = gaussian.sample((count,)).to(self.device)
            samples.append(current.cpu())
            labels.append(torch.tensor([label] * count).cpu())
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    @torch.no_grad()
    def get_feature(self, loader):
        features, labels = [], []
        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            _, current_features = self.model(images)
            features.append(current_features.cpu())
            labels.append(targets.cpu())
        return torch.cat(features, dim=0), torch.cat(labels)
