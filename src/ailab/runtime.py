from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def resolve_relpath(path: str | Path, base_dir: str | Path) -> Path:
    """
    Resolve path relative to base_dir if it's not absolute.
    This makes configs portable across machines.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(base_dir) / p


def get_env_override(name: str) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    return v if v else None


def resolve_run_root(cfg: Dict[str, Any], base_dir: Path) -> Path:
    # precedence: env -> cfg.paths.run_root -> default "runs"
    env = get_env_override("AILAB_RUN_ROOT")
    if env is not None:
        return Path(env)
    run_root = cfg.get("paths", {}).get("run_root", "runs")
    return resolve_relpath(run_root, base_dir)


def resolve_data_root(cfg: Dict[str, Any], base_dir: Path) -> Path:
    # precedence: env -> cfg.paths.data_root -> default "data"
    env = get_env_override("AILAB_DATA_ROOT")
    if env is not None:
        return Path(env)
    data_root = cfg.get("paths", {}).get("data_root", "data")
    return resolve_relpath(data_root, base_dir)


def resolve_device(device_str: str) -> torch.device:
    """
    device_str: "auto" | "cpu" | "cuda" | "cuda:0" | "mps"
    """
    s = device_str.strip().lower()

    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # mps for mac; harmless on linux
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # explicit device request
    dev = torch.device(s)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available on this machine.")
    if dev.type == "mps":
        ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not ok:
            raise RuntimeError("device=mps requested but MPS is not available on this machine.")
    return dev