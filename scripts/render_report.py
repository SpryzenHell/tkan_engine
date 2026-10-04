from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description="Render benchmark figures from saved JSON results.")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--inversion", default="results/ci_inversion_3000.json")
    p.add_argument("--arrow", default="results/arrow_benchmark_300m.json")
    p.add_argument("--ddp", default="results/ddp_benchmark_cpu.json")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inv = load(Path(args.inversion))
    models = ["MLP", "T-KAN"]
    rmse = [inv["mlp"]["test_rmse_bps"], inv["tkan"]["test_rmse_bps"]]
    ic = [inv["mlp"]["test_ic"], inv["tkan"]["test_ic"]]

    plt.figure(figsize=(8, 5))
    plt.bar(models, rmse)
    plt.ylabel("Test RMSE (bps)")
    plt.title("Controlled nonlinear inversion benchmark")
    plt.tight_layout()
    plt.savefig(out / "inversion_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(models, ic)
    plt.ylabel("Test information coefficient")
    plt.title("Controlled nonlinear inversion benchmark")
    plt.tight_layout()
    plt.savefig(out / "inversion_ic.png", dpi=180)
    plt.close()

    history = inv.get("tkan_training_history", [])
    if history:
        epochs = [item["epoch"] for item in history]
        loss = [item["loss"] for item in history]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("T-KAN training loss")
        plt.tight_layout()
        plt.savefig(out / "inversion_loss.png", dpi=180)
        plt.close()

    arrow = load(Path(args.arrow))
    plt.figure(figsize=(8, 5))
    plt.bar(["Events", "RecordBatches"], [arrow["events"], arrow["batches"]])
    plt.yscale("log")
    plt.ylabel("Count (log scale)")
    plt.title("Apache Arrow stress run")
    plt.tight_layout()
    plt.savefig(out / "arrow_300m.png", dpi=180)
    plt.close()

    ddp = load(Path(args.ddp))
    plt.figure(figsize=(8, 4.5))
    plt.barh(["2-rank CPU DDP"], [ddp["samples_per_s"]])
    plt.xlabel("Global samples / second")
    plt.title("DistributedDataParallel CPU smoke benchmark")
    plt.tight_layout()
    plt.savefig(out / "ddp_cpu.png", dpi=180)
    plt.close()

    print(f"Rendered figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
