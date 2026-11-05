import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from utilities.parse_utils import get_args


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_experiments.sh"


@pytest.mark.parametrize("custom_output", [False, True])
def test_six_commands_from_another_directory(tmp_path, custom_output):
    data, output = tmp_path / "data with spaces", tmp_path / "results with spaces"
    command = ["bash", str(SCRIPT), str(data)]
    if custom_output:
        command.append(str(output))
    result = subprocess.run(command + ["--dry-run"],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    expected = [("cifar100", 10, 10, 100, 5, 0.03), ("cifar100", 20, 5, 100, 5, 0.03),
                ("tiny-imagenet", 10, 20, 100, 10, 0.03), ("tiny-imagenet", 20, 10, 100, 10, 0.03),
                ("imagenet-subset", 10, 10, 160, 20, 0.1), ("imagenet-subset", 20, 5, 160, 20, 0.1)]
    for command, setting in zip(commands, expected):
        args = get_args(command[command.index(str(ROOT / "main.py")) + 1:])
        assert (args.dataset, args.n_task, args.n_class_first_task, args.epochs_first_task,
                args.lambda_s, args.lambda_p) == setting
        assert args.data_path == str(data)
        assert args.outpath == str(output if custom_output else ROOT / "experiments")
        assert args.ca_epochs == 30 and args.ca_batch_size == 256 and args.batch_size == 64
    assert not output.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_serial_execution_and_error_stop(tmp_path, fail):
    data = tmp_path / "data"
    for path in ("cifar-100-python/train", "cifar-100-python/test", "cifar-100-python/meta", "tiny-imagenet-200/train/.ready",
                 "tiny-imagenet-200/val/images/.ready", "tiny-imagenet-200/val/val_annotations.txt",
                 "imagenet_subset/train/.ready", "imagenet_subset/val/.ready", "imagenet_subset/val_100.txt"):
        target = data / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    record = tmp_path / "calls.jsonl"
    python = tmp_path / "fake python"
    python.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                      "with open(os.environ['CALLS'], 'a') as output: output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                      "sys.exit(7 if os.environ['FAIL'] == '1' else 0)\n")
    python.chmod(0o755)
    env = dict(os.environ, PYTHON=str(python), CALLS=str(record), FAIL=str(int(fail)))
    output = tmp_path / "results"
    result = subprocess.run(["bash", str(SCRIPT), str(data), str(output)], env=env,
                            capture_output=True, text=True)
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    if fail:
        assert result.returncode == 7 and len(calls) == 1
    else:
        assert result.returncode == 0 and len(calls) == 6
        assert all(str(ROOT / "main.py") in call for call in calls)
    assert list(output.iterdir()) == []
    assert "sgpa" not in result.stdout.lower() and "sgr" not in result.stdout.lower()
    repeated = subprocess.run(["bash", str(SCRIPT), str(data), str(output)], env=env,
                              capture_output=True, text=True)
    assert repeated.returncode == result.returncode
    assert len(record.read_text().splitlines()) == len(calls) * 2
