import argparse
import math


DATASET_DEFAULTS = {
    "cifar100": (100, 100, 5.0, 0.03),
    "tiny-imagenet": (200, 100, 10.0, 0.03),
    "imagenet-subset": (100, 160, 20.0, 0.1),
}


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Subspace-Guided Continual Learning (SGCL)")
    parser.add_argument("--approach", choices=["sgcl"], default="sgcl")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS, default="cifar100")
    parser.add_argument("--data_path", default="./cl_data")
    parser.add_argument("--outpath", "-op", default="./experiments")
    parser.add_argument("--n_task", type=int, choices=[10, 20], default=10)
    parser.add_argument("--n_class_first_task", type=int, default=None)
    parser.add_argument("--epochs_first_task", type=int, default=None)
    parser.add_argument("--epochs_next_task", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ca_epochs", type=int, default=30)
    parser.add_argument("--ca_batch_size", type=int, default=256)
    parser.add_argument("--ca_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_s", type=float, default=None, help="SGR stable loss weight")
    parser.add_argument("--lambda_p", type=float, default=None, help="SGR plastic loss weight")
    parser.add_argument("--sgpa_ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--nw", type=int, default=12)
    parser.add_argument("--max_tasks", type=int, default=-1, help="Limit tasks for a short check")
    args = parser.parse_args(argv)
    classes, epochs, lambda_s, lambda_p = DATASET_DEFAULTS[args.dataset]
    defaults = dict(n_class_first_task=classes // args.n_task, epochs_first_task=epochs,
                    lambda_s=lambda_s, lambda_p=lambda_p)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.n_class_first_task != classes // args.n_task:
        parser.error("n_class_first_task must equal the number of classes per task")
    if any(not math.isfinite(value) or value < 0 for value in (args.lambda_s, args.lambda_p)):
        parser.error("lambda_s and lambda_p must be finite and non-negative")
    if not math.isfinite(args.sgpa_ridge) or args.sgpa_ridge <= 0:
        parser.error("sgpa_ridge must be finite and positive")
    if args.max_tasks != -1 and args.max_tasks < 1:
        parser.error("max_tasks must be -1 or positive")
    args.backbone = "resnet18"
    return args
