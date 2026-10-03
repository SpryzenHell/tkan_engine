from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tkan_engine.data import SequenceDataset
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import TrainConfig, cleanup_ddp, setup_ddp


def make_fixed_dataset(n: int, seq_len: int, features: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, seq_len, features)).astype(np.float32)
    y = (
        0.5 * x[:, -1, 0]
        - 0.2 * x[:, -1, 1]
        + 0.1 * np.sin(x[:, -1, 2])
    ).astype(np.float32)
    return SequenceDataset(x, y)


p = argparse.ArgumentParser()
p.add_argument("--events", type=int, default=200_000)
p.add_argument("--seq-len", type=int, default=64)
p.add_argument("--features", type=int, default=15)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--grid-size", type=int, default=12)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--distributed", action="store_true")
p.add_argument("--out", default="results/ddp_benchmark.json")
a = p.parse_args()

rank, world, local_rank = setup_ddp(a.distributed)
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

dataset = make_fixed_dataset(a.events, a.seq_len, a.features)
if world > 1:
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True
    )
else:
    sampler = None

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=a.batch_size,
    sampler=sampler,
    shuffle=sampler is None,
    pin_memory=device.type == "cuda",
)
model = MicrostructureForecaster(
    a.features,
    hidden_dim=a.hidden,
    depth=a.depth,
    grid_size=a.grid_size,
).to(device)

if world > 1:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.SmoothL1Loss()
core = model.module if hasattr(model, "module") else model

it = iter(loader)
for _ in range(a.warmup):
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader)
        xb, yb = next(it)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    pred = model(xb)
    loss = loss_fn(pred, yb) + core.regularization_loss()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

if device.type == "cuda":
    torch.cuda.synchronize()

if world > 1:
    torch.distributed.barrier()

start = time.perf_counter()
samples = 0
for step in range(a.steps):
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader)
        xb, yb = next(it)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    pred = model(xb)
    loss = loss_fn(pred, yb) + core.regularization_loss()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    samples += len(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
if world > 1:
    torch.distributed.barrier()

elapsed = time.perf_counter() - start
result = {
    "rank": rank,
    "world_size": world,
    "device": str(device),
    "global_batch_size": a.batch_size * world,
    "steps": a.steps,
    "global_samples": samples * world,
    "wall_seconds": elapsed,
    "samples_per_s": samples * world / elapsed,
    "step_ms": elapsed / a.steps * 1000,
    "parameters": sum(p.numel() for p in core.parameters()),
}

if rank == 0:
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
cleanup_ddp()
