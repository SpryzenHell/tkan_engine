from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tkan_engine.model import MicrostructureForecaster


p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seq-len", type=int, default=64)
p.add_argument("--features", type=int, default=20)
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--steps", type=int, default=100)
args = p.parse_args()

torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MicrostructureForecaster(
    args.features,
    hidden_dim=args.hidden,
    depth=2,
    grid_size=8,
).eval().to(device)

x = torch.randn(
    args.batch,
    args.seq_len,
    args.features,
    device=device,
)

with torch.no_grad():
    for _ in range(args.warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.steps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

result = {
    "device": str(device),
    "batch": args.batch,
    "seq_len": args.seq_len,
    "features": args.features,
    "steps": args.steps,
    "samples_per_s": args.batch * args.steps / elapsed,
    "ms_per_batch": elapsed / args.steps * 1000,
    "params": sum(p.numel() for p in model.parameters()),
    "theoretical_gemm_note": "B-spline basis is flattened once per layer and projected with F.linear.",
}

Path("results").mkdir(exist_ok=True)
Path("results/model_benchmark.json").write_text(
    json.dumps(result, indent=2)
)
print(json.dumps(result, indent=2))
