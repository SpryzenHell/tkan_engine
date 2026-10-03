from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grid_update_interval: int = 25
    num_workers: int = 0
    amp: bool = True
    compile: bool = False


def setup_ddp(distributed: bool = False):
    if not distributed:
        return 0, 1, 0

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(model, dataset, cfg: TrainConfig, distributed: bool = False, out_dir="results"):
    rank, world, local_rank = setup_ddp(distributed)

    device = torch.device(
        f"cuda:{local_rank}"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
        )
        if distributed and world > 1
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    if distributed and world > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            static_graph=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.amp and device.type == "cuda"),
    )
    criterion = nn.SmoothL1Loss()
    history = []
    step = 0

    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        running = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=scaler.is_enabled(),
            ):
                pred = model(xb)
                core = model.module if isinstance(model, DDP) else model
                loss = criterion(pred, yb) + core.regularization_loss()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            step += 1

            if (
                cfg.grid_update_interval
                and step % cfg.grid_update_interval == 0
            ):
                core = model.module if isinstance(model, DDP) else model
                with torch.no_grad():
                    sample_x = xb[: min(2048, len(xb))]
                    core.encoder.update_grids(sample_x)

                if distributed and world > 1:
                    for block in core.encoder.blocks:
                        dist.broadcast(
                            block.kan.grid.data,
                            src=0,
                        )
                        dist.broadcast(
                            block.kan.spline_weight.data,
                            src=0,
                        )

        avg = running / max(1, len(loader))

        if world > 1:
            tensor = torch.tensor([avg], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            avg = float(tensor.item() / world)

        history.append({"epoch": epoch + 1, "loss": avg})

    result = {
        "device": str(device),
        "world_size": world,
        "history": history,
        "config": asdict(cfg),
    }

    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "train_result.json").write_text(
            json.dumps(result, indent=2)
        )

    if distributed:
        cleanup_ddp()

    return result
