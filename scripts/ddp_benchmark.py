from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from tkan_engine.data import SequenceDataset
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import cleanup_ddp, setup_ddp


def make_fixed_dataset(
    n: int,
    seq_len: int,
    features: int,
    seed: int = 7,
) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, seq_len, features)).astype(np.float32)
    y = (
        0.5 * x[:, -1, 0]
        - 0.2 * x[:, -1, 1]
        + 0.1 * np.sin(x[:, -1, 2])
    ).astype(np.float32)
    return SequenceDataset(x, y)


parser = argparse.ArgumentParser(
    description="Measure T-KAN training throughput on CPU or one or more GPUs."
)
parser.add_argument("--events", type=int, default=200_000)
parser.add_argument("--seq-len", type=int, default=64)
parser.add_argument("--features", type=int, default=15)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--grid-size", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--out", default="results/ddp_benchmark.json")
args = parser.parse_args()

for name, value in (
    ("events", args.events),
    ("seq-len", args.seq_len),
    ("features", args.features),
    ("hidden", args.hidden),
    ("depth", args.depth),
    ("grid-size", args.grid_size),
    ("batch-size", args.batch_size),
    ("steps", args.steps),
):
    if value <= 0:
        parser.error(f"--{name} must be positive")
if args.warmup < 0:
    parser.error("--warmup must be non-negative")

rank, world, local_rank = setup_ddp(args.distributed)
device = torch.device(
    f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
)
if device.type == "cuda":
    torch.cuda.set_device(local_rank)

dataset = make_fixed_dataset(
    args.events,
    args.seq_len,
    args.features,
    seed=args.seed,
)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
sampler = (
    torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    if world > 1
    else None
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    shuffle=sampler is None,
    pin_memory=device.type == "cuda",
    drop_last=True,
)
if len(loader) == 0:
    raise SystemExit(
        "The data set is too small for this batch size and world size; "
        "increase --events or lower --batch-size."
    )

model = MicrostructureForecaster(
    args.features,
    hidden_dim=args.hidden,
    depth=args.depth,
    grid_size=args.grid_size,
).to(device)

if world > 1:
    from torch.nn.parallel import DistributedDataParallel

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )

core = model.module if hasattr(model, "module") else model
optimizer_fused = False
optimizer_args = {"lr": 1e-3}
if device.type == "cuda":
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            fused=True,
            **optimizer_args,
        )
        optimizer_fused = True
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
else:
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)

amp_enabled = device.type == "cuda" and not args.no_amp
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
loss_fn = torch.nn.SmoothL1Loss()


def step(xb: torch.Tensor, yb: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(xb)
            loss = loss_fn(prediction, yb) + core.regularization_loss()
    else:
        prediction = model(xb)
        loss = loss_fn(prediction, yb) + core.regularization_loss()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


iterator = iter(loader)
for _ in range(args.warmup):
    try:
        xb, yb = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        xb, yb = next(iterator)
    step(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))

if device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
if world > 1:
    torch.distributed.barrier()

start = time.perf_counter()
local_samples = 0
for _ in range(args.steps):
    try:
        xb, yb = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        xb, yb = next(iterator)
    step(xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
    local_samples += len(xb)

# Synchronize only outside the timed loop; synchronizing every step can lower
# throughput and makes this a poor measure of queued CUDA execution.
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

if world > 1:
    torch.distributed.barrier()
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(
        elapsed_tensor,
        op=torch.distributed.ReduceOp.MAX,
    )
    elapsed = float(elapsed_tensor.item())
    sample_tensor = torch.tensor(
        [local_samples],
        dtype=torch.long,
        device=device,
    )
    torch.distributed.all_reduce(
        sample_tensor,
        op=torch.distributed.ReduceOp.SUM,
    )
    global_samples = int(sample_tensor.item())
else:
    global_samples = local_samples

result = {
    "rank": rank,
    "world_size": world,
    "device": str(device),
    "device_name": (
        torch.cuda.get_device_name(local_rank)
        if device.type == "cuda"
        else "cpu"
    ),
    "visible_gpu_count": torch.cuda.device_count(),
    "amp_enabled": amp_enabled,
    "optimizer_fused": optimizer_fused,
    "events": args.events,
    "seed": args.seed,
    "seq_len": args.seq_len,
    "features": args.features,
    "hidden": args.hidden,
    "depth": args.depth,
    "grid_size": args.grid_size,
    "local_batch_size": args.batch_size,
    "global_batch_size": args.batch_size * world,
    "warmup_steps": args.warmup,
    "measured_steps": args.steps,
    "global_samples": global_samples,
    "wall_seconds": elapsed,
    "samples_per_s": global_samples / elapsed if elapsed > 0 else 0.0,
    "step_ms": elapsed / args.steps * 1000,
    "parameters": sum(parameter.numel() for parameter in core.parameters()),
    "max_memory_allocated_bytes": (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    ),
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
    },
}

if rank == 0:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

cleanup_ddp()
