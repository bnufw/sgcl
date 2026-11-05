# SGCL

**Subspace-Guided Continual Learning: Hessian Based Stable–Plastic Decomposition for Exemplar-Free Class-Incremental Learning**

SGCL combines **SGR** regularization (stability and plasticity losses) with **SGPA** prototype alignment.

## Installation

Use Python 3.12 and a CUDA-enabled PyTorch environment:

```bash
python -m pip install -r requirements.txt
```

## Data

Prepare the datasets locally with the following directory structure:

```text
DATA_ROOT/
├── cifar-100-python/{train,test,meta}
├── tiny-imagenet-200/
│   ├── train/
│   └── val/{images/,val_annotations.txt}
└── imagenet_subset/
    ├── train/
    ├── val/
    └── val_100.txt
```

## Run the Six Experiments

```bash
bash run_experiments.sh /path/to/DATA_ROOT
```

The script runs the following six experiments sequentially, using seed 0 and GPU 0 by default.

| Dataset | Tasks | Classes in the First Task | Epochs (First/Subsequent Tasks) | `(λs, λp)` |
| --- | ---: | ---: | --- | --- |
| CIFAR-100 | 10 | 10 | 100/100 | (5, 0.03) |
| CIFAR-100 | 20 | 5 | 100/100 | (5, 0.03) |
| Tiny-ImageNet | 10 | 20 | 100/100 | (10, 0.03) |
| Tiny-ImageNet | 20 | 10 | 100/100 | (10, 0.03) |
| ImageNet-Subset | 10 | 10 | 160/100 | (20, 0.1) |
| ImageNet-Subset | 20 | 5 | 160/100 | (20, 0.1) |

All experiments use ResNet-18 with a training batch size of 64. Classifier calibration runs for 30 epochs with a batch size of 256 and an SGD learning rate of `1e-3`.

If the datasets are stored separately, set `CIFAR_DATA_PATH`, `TINY_DATA_PATH`, and `IMAGENET_DATA_PATH` to the parent directories of their respective dataset directories. You can also set `PYTHON`, `DEVICE`, `NW`, and `SEED` through environment variables. Add `--dry-run` to print the commands without executing them.

The default output root is `experiments/`; you can override it with the script's second argument. Each experiment creates a directory at `experiments/sgcl_<dataset>_<num_tasks>step_<random_id>/`, containing only `logger/` and `stdout-YYYY-MM-DD-HH-MM.txt`. The `logger/` directory contains six result files:

- `acc_tag.out` and `acc_taw.out`: per-task accuracy matrices.
- `acc_tag_perstep.out` and `acc_taw_perstep.out`: average accuracy at each step.
- `forg_tag.out` and `forg_taw.out`: per-task forgetting matrices.

Each experiment directory also contains `stdout-YYYY-MM-DD-HH-MM.txt`, which records the training progress and evaluation output shown in the terminal.

## Acknowledgements

This project is based on [Elastic Feature Consolidation (EFC)](https://github.com/simomagi/elastic_feature_consolidation). We thank the authors for making their code publicly available.
