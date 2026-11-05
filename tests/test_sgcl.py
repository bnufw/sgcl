import importlib
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continual_learning.SGCL import SGCL
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.OptimizerManager import OptimizerManager
from continual_learning.utils.sgr import FeatureHessian
from utilities.parse_utils import get_args


class SmallModel(BaseModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = nn.Linear(6, 6, bias=False)
        self.backbone.feature_space_size = 6
        self.heads = nn.ModuleList()


@pytest.mark.parametrize("dataset,epochs,weights", [
    ("cifar100", 100, (5, 0.03)), ("tiny-imagenet", 100, (10, 0.03)),
    ("imagenet-subset", 160, (20, 0.1)),
])
def test_dataset_configuration(dataset, epochs, weights):
    args = get_args(["--dataset", dataset])
    assert args.approach == "sgcl" and args.backbone == "resnet18"
    assert (args.epochs_first_task, args.epochs_next_task) == (epochs, 100)
    assert (args.lambda_s, args.lambda_p) == weights
    assert (args.batch_size, args.ca_batch_size, args.ca_epochs, args.ca_lr) == (64, 256, 30, 1e-3)
    args = get_args(["--dataset", dataset, "--lambda_s", "0", "--lambda_p", "0", "--ca_epochs", "1"])
    assert args.lambda_s == args.lambda_p == 0 and args.ca_epochs == 1


@pytest.mark.parametrize("dataset", ["cifar100", "tiny-imagenet", "imagenet-subset"])
def test_optimizer_parameters_and_heads(dataset):
    model = SmallModel()
    model.add_classification_head(2)
    manager = OptimizerManager(dataset, rotation=False)
    first, scheduler = manager.get_optimizer(0, model, None)
    assert isinstance(first, torch.optim.SGD)
    assert (first.param_groups[0]["lr"], first.param_groups[0]["momentum"], first.param_groups[0]["weight_decay"]) == (0.1, 0.9, 5e-4)
    assert sorted(scheduler.milestones) == ([80, 120, 150] if dataset == "imagenet-subset" else [60, 120, 160])
    model.add_classification_head(2)
    later, scheduler = manager.get_optimizer(1, model, None)
    optimized = {id(p) for group in later.param_groups for p in group["params"]}
    assert all(id(p) not in optimized for p in model.heads[0].parameters())
    assert all(id(p) in optimized for p in model.heads[1].parameters())
    if dataset == "imagenet-subset":
        assert isinstance(later, torch.optim.Adam)
        assert [group["lr"] for group in later.param_groups] == [1e-4, 1e-5]
        assert all(group["weight_decay"] == 5e-4 for group in later.param_groups)
        assert min(scheduler.milestones) > 100
    else:
        assert isinstance(later, torch.optim.SGD)
        assert later.param_groups[0]["lr"] == 0.005
        assert later.param_groups[0]["weight_decay"] == 2e-4
        assert sorted(scheduler.milestones) == [45, 90]
        for _ in range(45):
            later.step()
            scheduler.step()
        assert later.param_groups[0]["lr"] == pytest.approx(0.0005)


def test_two_task_sgcl(monkeypatch, tmp_path):
    module = importlib.import_module("continual_learning.SGCL")
    monkeypatch.setattr(module, "BaseModel", lambda **kwargs: SmallModel())
    args = get_args(["--ca_epochs", "1", "--batch_size", "8", "--device", "cpu"])
    args.n_class_first_task = 2
    torch.manual_seed(7)
    approach = SGCL(args, "cpu", str(tmp_path), 2, {0: [0, 1], 1: [2, 3]})
    approach.rotation = approach.optimizer_manager.rotation = False
    events = []
    align, calibrate, compute = module.sgpa, approach.calibrate_classifier, FeatureHessian.compute
    old_basis = None

    def record_alignment(prototypes, W, b, basis):
        torch.testing.assert_close(basis, old_basis)
        events.append("sgpa")
        return align(prototypes, W, b, basis)

    def record_calibration(*args):
        events.append("ca")
        return calibrate(*args)

    def record_hessian(*args):
        events.append("hessian")
        return compute(*args)

    monkeypatch.setattr(module, "sgpa", record_alignment)
    monkeypatch.setattr(approach, "calibrate_classifier", record_calibration)
    monkeypatch.setattr(FeatureHessian, "compute", record_hessian)
    for task in range(2):
        loader = DataLoader(TensorDataset(torch.randn(16, 6), torch.arange(16) % 2 + task * 2), batch_size=8)
        approach.pre_train(task)
        approach.train(task, loader, 0, 1)
        approach.post_train(task, loader)
        old_basis = approach.hessian_basis.clone()
        assert approach.hessian_basis.shape == (6, approach.hessian_values.numel())
        assert all(p.grad is None for p in approach.old_model.parameters())
        assert all(0 <= value <= 1 for value in approach.eval(task, task, loader, verbose=False))
    assert events == ["hessian", "sgpa", "ca", "hessian"]
    assert len(approach.proto_generator.prototype) == 4
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("options", [["--lambda_p", "nan"], ["--sgpa_ridge", "0"],
    ["--n_class_first_task", "50"], ["--max_tasks", "0"]])
def test_invalid_configuration(options):
    with pytest.raises(SystemExit):
        get_args(options)


def test_main_writes_six_results_and_stdout(monkeypatch, tmp_path, capsys):
    import main as entry

    class ImageModel(SmallModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 6, bias=False))
            self.backbone.feature_space_size = 6
            self.heads = nn.ModuleList()

    class ImageData(torch.utils.data.Dataset):
        def __init__(self):
            self.targets = [label for label in range(100) for _ in range(4)]
            self.images = torch.randn(400, 3, 32, 32) * 0.05
            self.class_to_idx = {str(label): label for label in range(100)}

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return self.images[index], self.targets[index]

    module = importlib.import_module("continual_learning.SGCL")
    monkeypatch.setattr(module, "BaseModel", lambda **kwargs: ImageModel())
    monkeypatch.setattr(entry, "get_dataset", lambda *args: (ImageData(), ImageData(), 100))
    monkeypatch.chdir(tmp_path)
    args = get_args(["--device", "cpu", "--nw", "0",
                     "--epochs_first_task", "1", "--epochs_next_task", "1", "--ca_epochs", "1",
                     "--batch_size", "8", "--max_tasks", "2"])
    original_stdout = sys.stdout
    entry.main(args)
    assert sys.stdout is original_stdout
    experiments = list((tmp_path / "experiments").glob("sgcl_cifar100_10step_*"))
    assert len(experiments) == 1
    filenames = {"acc_tag.out", "acc_taw.out", "acc_tag_perstep.out", "acc_taw_perstep.out",
                 "forg_tag.out", "forg_taw.out"}
    expected = {experiments[0] / "logger" / filename for filename in filenames}
    stdout_logs = list(experiments[0].glob("stdout-*.txt"))
    assert len(stdout_logs) == 1
    expected.add(stdout_logs[0])
    assert {path for path in (tmp_path / "experiments").rglob("*") if path.is_file()} == expected
    assert {path.name for path in experiments[0].iterdir()} == {"logger", stdout_logs[0].name}
    log = stdout_logs[0].read_text(encoding="utf-8")
    assert "SGCL:" in log and "Task 1, epoch 0/1" in log
    assert "TAw acc=" in log and "TAg acc" in log
    assert "sgr" not in log.lower() and "sgpa" not in log.lower()
    for mode in ("tag", "taw"):
        accuracy = np.loadtxt(experiments[0] / "logger" / f"acc_{mode}.out", delimiter=",")
        per_step = np.loadtxt(experiments[0] / "logger" / f"acc_{mode}_perstep.out", delimiter=",")
        forgetting = np.loadtxt(experiments[0] / "logger" / f"forg_{mode}.out", delimiter=",")
        assert accuracy.shape == forgetting.shape == (2, 2)
        assert np.all((0 <= accuracy) & (accuracy <= 100))
        np.testing.assert_allclose(per_step, [accuracy[0, 0], accuracy[1].mean()], atol=1e-3)
        np.testing.assert_allclose(forgetting[1, 0], accuracy[0, 0] - accuracy[1, 0], atol=1e-3)
    captured = capsys.readouterr()
    assert log in captured.out
    output = (captured.out + captured.err).lower()
    assert "sgr" not in output and "sgpa" not in output
