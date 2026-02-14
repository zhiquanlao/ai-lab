# ai-lab

Reproducible ML experiments (configs + runs + logs + checkpoints).

## Table of contents
- [Setup](#setup)
- [Quick start](#quick-start)
- [Run directories and outputs](#run-directories-and-outputs)
- [Portable paths](#portable-paths)
- [How to add a new experiment](#how-to-add-a-new-experiment)
- [Experiment checklist](#experiment-checklist)
- [Experiments](#experiments)

## Setup
Create and activate a venv:
```bash
python3 -m venv ~/venvs/ai
source ~/venvs/ai/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick start
Run the MNIST MLP experiment:
```bash
python experiments/mnist_mlp/train.py --config experiments/mnist_mlp/config.yaml
```
Every run creates a new directory under runs/:
- `runs/<timestamp>-<run_name>/config.resolved.yaml` (exact config used)
- `runs/<timestamp>-<run_name>/metrics.jsonl` (one JSON record per epoch/step)
- `runs/<timestamp>-<run_name>/best.pt` (checkpoint of best validation metric)

It is recommended to use relative paths in `.yaml`:
```yaml
paths:
  data_root: ../../data
  run_root: ../../runs
```

## How to add a new experiment
```bash
mkdir -p experiments/<exp_name>
```
Minimum contents:
- `experiments/<exp_name>/config.yaml`
- `experiments/<exp_name>/train.py`
- `experiments/<exp_name>/README.md`

Conventions:
- Put reusable code (models, datasets, utils) in `src/ailab/`
- Datasets live under `${data_root}/raw/` and processed outputs under `${data_root}/processed/`
- Never commit `data/` or `runs/`
