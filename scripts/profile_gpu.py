from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tkan_engine.model import MicrostructureForecaster


p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=64)
p.add_argument("--seq-len", type=int, default=64)
p.add_argument("--features", type=int, default=15)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--out", default="results/gpu_profile.txt")
args = p.parse_args()

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable on this host. "
        "No GPU performance number is claimed."
    )

device = torch.device("cuda")
model = MicrostructureForecaster(
    args.features,
    hidden_dim=args.hidden,
    depth=3,
    grid_size=12,
).to(device)

x = torch.randn(
    args.batch,
    args.seq_len,
    args.features,
    device=device,
)

for _ in range(5):
    model(x)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(args.steps):
        with record_function("tkan_forward"):
            model(x)
        torch.cuda.synchronize()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(
    prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
    )
)
print(Path(args.out).read_text())
