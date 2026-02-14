import os
import subprocess
from pathlib import Path

def test_mnist_mlp_smoke(tmp_path: Path):
    # Put all artifacts under tmp_path (portable; no local assumptions)
    env = os.environ.copy()
    env["AILAB_RUN_ROOT"] = str(tmp_path / "runs")
    env["AILAB_DATA_ROOT"] = str(tmp_path / "data")  # unused in FakeData mode but fine

    # Run the experiment in FakeData + tiny limits
    cmd = [
        "python",
        "experiments/mnist_mlp/train.py",
        "--config",
        "experiments/mnist_mlp/config.yaml",
    ]

    # Override via environment + a tiny config edit pattern:
    # We'll rely on the config having dataset=mnist; we override via CLI by setting AILAB_* only,
    # and we flip dataset/limits by passing them through a temporary config file.
    cfg_text = Path("experiments/mnist_mlp/config.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("dataset: mnist", "dataset: fake")
    cfg_text = cfg_text.replace("epochs: 2", "epochs: 1")
    cfg_text = cfg_text.replace("limit_train_batches: null", "limit_train_batches: 2")
    cfg_text = cfg_text.replace("limit_test_batches: null", "limit_test_batches: 2")
    tmp_cfg = tmp_path / "config.yaml"
    tmp_cfg.write_text(cfg_text, encoding="utf-8")

    cmd[-1] = str(tmp_cfg)  # replace config path

    p = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    # Basic assertions: created a run directory and wrote metrics
    runs_root = tmp_path / "runs"
    run_dirs = list(runs_root.glob("*-mnist_mlp"))
    assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"

    run_dir = run_dirs[0]
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "best.pt").exists()