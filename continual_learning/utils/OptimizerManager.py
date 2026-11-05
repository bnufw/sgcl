import torch


class OptimizerManager:
    def __init__(self, dataset, rotation=True):
        self.dataset = dataset
        self.rotation = rotation

    def get_optimizer(self, task_id, model, auxiliary_classifier):
        if task_id == 0:
            params = ([p for p in model.backbone.parameters() if p.requires_grad]
                      + [p for p in model.heads.parameters() if p.requires_grad])
            if self.rotation:
                params += [p for p in auxiliary_classifier.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
            milestones = [80, 120, 150] if self.dataset == "imagenet-subset" else [60, 120, 160]
        else:
            model.freeze_bn()
            head_params = [p for p in model.heads[-1].parameters() if p.requires_grad]
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            if self.dataset == "imagenet-subset":
                optimizer = torch.optim.Adam(
                    [{"params": head_params, "lr": 1e-4}, {"params": backbone_params}],
                    lr=1e-5, weight_decay=5e-4)
                milestones = [1000, 2000]
            else:
                optimizer = torch.optim.SGD(backbone_params + head_params, lr=0.005,
                                            momentum=0.9, weight_decay=2e-4)
                milestones = [45, 90]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print("Optimizer: {}, lr={}, weight_decay={}, milestones={}".format(
            type(optimizer).__name__, [group["lr"] for group in optimizer.param_groups],
            [group["weight_decay"] for group in optimizer.param_groups], milestones))
        return optimizer, scheduler
