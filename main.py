import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

from torch.utils.data import DataLoader

from continual_learning.SGCL import SGCL
from dataset.continual_learning_dataset import ContinualLearningDataset
from dataset.dataset_utils import get_dataset
from utilities.generic_utils import experiment_folder, get_task_dict, remap_targets, seed_everything
from utilities.matrix_logger import FileOutputDuplicator, Logger
from utilities.parse_utils import get_args


def train_task(approach, task_id, train_loader, eval_loader):
    approach.pre_train(task_id)
    for epoch in range(approach.total_epochs):
        approach.train(task_id, train_loader, epoch, approach.total_epochs)
        approach.eval(task_id, task_id, eval_loader, verbose=True)
        approach.scheduler.step()


def finish_task(approach, logger, task_id, train_loader, test_loaders):
    approach.post_train(task_id, train_loader)
    for test_id in range(task_id + 1):
        taw, tag = approach.eval(task_id, test_id, test_loaders[test_id], verbose=False)
        logger.update_accuracy(task_id, test_id, taw, tag)
        if test_id < task_id:
            logger.update_forgetting(task_id, test_id)
        logger.print_latest(task_id, test_id)
    logger.compute_average()
    logger.print_file()


def main(args=None):
    args = get_args() if args is None else args
    os.makedirs(args.outpath, exist_ok=True)
    out_path, _ = experiment_folder(args.outpath, f"{args.approach}_{args.dataset}_{args.n_task}step")
    device = "cpu" if args.device == "cpu" else "cuda:" + str(args.device)
    seed_everything(args.seed)
    train_set, test_set, total_classes = get_dataset(args.dataset, args.data_path)
    train_set, test_set, _ = remap_targets(train_set, test_set, total_classes)
    class_per_task = total_classes // args.n_task
    task_dict = get_task_dict(args.n_task, total_classes)
    run_n_tasks = args.n_task if args.max_tasks == -1 else min(args.max_tasks, args.n_task)
    active_task_dict = {task: task_dict[task] for task in range(run_n_tasks)}
    train_datasets, _ = ContinualLearningDataset(train_set, task_dict, total_classes, train=True).collect()
    test_datasets, test_sizes = ContinualLearningDataset(test_set, task_dict, total_classes, train=False).collect()
    test_loaders = [DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.nw)
                    for dataset in test_datasets]
    train_loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nw)
                     for dataset in train_datasets]
    train_loaders, test_loaders = train_loaders[:run_n_tasks], test_loaders[:run_n_tasks]
    logger = Logger(out_path, run_n_tasks, active_task_dict, test_sizes[:run_n_tasks])
    os.makedirs(os.path.join(out_path, "logger"), exist_ok=True)
    stdout_path = os.path.join(out_path, "stdout-{}.txt".format(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    with open(stdout_path, "w", encoding="utf-8", buffering=1) as output:
        with redirect_stdout(FileOutputDuplicator(sys.stdout, output)):
            approach = SGCL(args, device, out_path, class_per_task, task_dict)
            for task_id, train_loader in enumerate(train_loaders):
                train_task(approach, task_id, train_loader, test_loaders[task_id])
                finish_task(approach, logger, task_id, train_loader, test_loaders)


if __name__ == "__main__":
    main()
