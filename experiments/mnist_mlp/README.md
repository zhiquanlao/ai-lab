
A minimal MNIST classifier used to validate the experiment harness (configs, runs, logging, checkpoints).

## Table of contents
- [Goal](#goal)
- [Data](#data)
- [How to run](#how-to-run)
- [Outputs](#outputs)
- [Expected result](#expected-result)
- [Notes](#notes)

## Goal
Smoke-test a full training loop + evaluation + logging on a CPU-friendly problem.

## Data
MNIST is downloaded automatically into:
- `${data_root}/raw/` (default: `../../data/raw/` relative to this experiment)

## How to run
From repo root:
```bash
python experiments/mnist_mlp/train.py --config experiments/mnist_mlp/config.yaml
