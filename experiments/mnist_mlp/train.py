from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ailab.runtime import resolve_data_root, resolve_device, resolve_run_root
from ailab.utils import dump_yaml, get_git_sha, load_yaml, log_jsonl, make_run_dir, set_seed


class MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        n += x.size(0)
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None, help='Override device, e.g. "cpu", "cuda", "cuda:0", "auto"')
    ap.add_argument("--run_root", type=str, default=None, help="Override run root directory")
    ap.add_argument("--data_root", type=str, default=None, help="Override data root directory")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent  # anchor relative paths to the config location
    cfg = load_yaml(cfg_path)

    # CLI overrides (portable across machines)
    if args.device is not None:
        cfg["device"] = args.device
    if args.run_root is not None:
        cfg.setdefault("paths", {})["run_root"] = args.run_root
    if args.data_root is not None:
        cfg.setdefault("paths", {})["data_root"] = args.data_root

    run_root = resolve_run_root(cfg, cfg_dir)
    data_root = resolve_data_root(cfg, cfg_dir)

    run_name = cfg.get("run_name", "run")
    run_dir = make_run_dir(run_root, run_name)

    # Repro metadata
    cfg["_meta"] = {
        "git_sha": get_git_sha(),
        "config_path": str(cfg_path),
        "config_dir": str(cfg_dir),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_yaml(cfg, run_dir / "config.resolved.yaml")

    seed = int(cfg["seed"])
    set_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # dataset location is relative to data_root, not your machine
    mnist_root = data_root / "raw"
    train_ds = datasets.MNIST(root=str(mnist_root), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(mnist_root), train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = MLP(hidden_dim=int(cfg["model"]["hidden_dim"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    metrics_path = run_dir / "metrics.jsonl"
    best_test_acc = -1.0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        total = 0.0
        n = 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total += float(loss) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(train_loss=total / n)

        train_loss = total / n
        test_loss, test_acc = evaluate(model, test_loader, device)

        record: Dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        log_jsonl(metrics_path, record)
        print(record)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, run_dir / "best.pt")

    print(f"device={device}")
    print(f"run_dir={run_dir}")
    print(f"data_root={data_root}")


if __name__ == "__main__":
    main()