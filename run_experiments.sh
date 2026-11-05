#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
NW="${NW:-12}"
SEED="${SEED:-0}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

usage() {
    printf 'Usage: bash run_experiments.sh [DATA_ROOT] [OUTPUT_DIR] [--dry-run]\n'
    printf 'Environment: PYTHON, DEVICE, NW, SEED, CIFAR_DATA_PATH, TINY_DATA_PATH, IMAGENET_DATA_PATH\n'
}

dry_run=0
paths=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        --*) usage >&2; exit 2 ;;
        *) paths+=("$argument") ;;
    esac
done
if ((${#paths[@]} > 2)); then usage >&2; exit 2; fi

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

data_root="${paths[0]:-$PROJECT_ROOT/cl_data}"
output="$(absolute_path "${paths[1]:-$PROJECT_ROOT/experiments}")"
cifar_data="$(absolute_path "${CIFAR_DATA_PATH:-$data_root}")"
tiny_data="$(absolute_path "${TINY_DATA_PATH:-$data_root}")"
imagenet_data="$(absolute_path "${IMAGENET_DATA_PATH:-$data_root}")"

if (( ! dry_run )); then
    for required in "$cifar_data/cifar-100-python/train" "$cifar_data/cifar-100-python/test" "$cifar_data/cifar-100-python/meta" \
                    "$tiny_data/tiny-imagenet-200/train" "$tiny_data/tiny-imagenet-200/val/images" \
                    "$tiny_data/tiny-imagenet-200/val/val_annotations.txt" \
                    "$imagenet_data/imagenet_subset/train" "$imagenet_data/imagenet_subset/val" \
                    "$imagenet_data/imagenet_subset/val_100.txt"; do
        if [[ ! -e "$required" ]]; then
            printf 'Missing dataset path: %s\n' "$required" >&2
            exit 1
        fi
    done
    mkdir -p -- "$output"
fi
cd -- "$PROJECT_ROOT"

run() {
    local dataset="$1" tasks="$2" classes="$3" epochs="$4" lambda_s="$5" lambda_p="$6" data="$7"
    local command=("$PYTHON" -B -u "$PROJECT_ROOT/main.py" --approach sgcl
        --dataset "$dataset" --data_path "$data" --n_task "$tasks" --n_class_first_task "$classes"
        --epochs_first_task "$epochs" --epochs_next_task 100 --batch_size 64
        --ca_epochs 30 --ca_batch_size 256 --ca_lr 1e-3
        --lambda_s "$lambda_s" --lambda_p "$lambda_p" --sgpa_ridge 1e-3
        --seed "$SEED" --device "$DEVICE" --nw "$NW" -op "$output")
    if (( dry_run )); then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf 'Running %s, %s tasks\n' "$dataset" "$tasks"
        "${command[@]}"
    fi
}

run cifar100        10 10 100  5 0.03 "$cifar_data"
run cifar100        20  5 100  5 0.03 "$cifar_data"
run tiny-imagenet   10 20 100 10 0.03 "$tiny_data"
run tiny-imagenet   20 10 100 10 0.03 "$tiny_data"
run imagenet-subset 10 10 160 20 0.1  "$imagenet_data"
run imagenet-subset 20  5 160 20 0.1  "$imagenet_data"
