from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tkan_engine.data import SequenceDataset, SyntheticLOBStream, make_sequences
from tkan_engine.features import LOBFeatureExtractor
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import TrainConfig, train


p = argparse.ArgumentParser()
p.add_argument("--events", type=int, default=30_000)
p.add_argument("--seq-len", type=int, default=64)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--hidden", type=int, default=48)
p.add_argument("--depth", type=int, default=2)
p.add_argument("--grid-size", type=int, default=8)
p.add_argument("--ddp", action="store_true")
args = p.parse_args()

frames = [
    LOBFeatureExtractor.transform(frame, levels=10)
    for frame in SyntheticLOBStream(
        args.events,
        batch_size=10_000,
    ).batches()
]
df = pd.concat(frames, ignore_index=True)
df["mid_return"] = (
    df["mid"].pct_change().fillna(0.0).astype(np.float32)
)

features = df.drop(columns=["mid"]).to_numpy(np.float32)
target = (
    df["mid_return"].shift(-1).fillna(0.0).to_numpy(np.float32)
    * 1e4
)

cut = int(len(features) * 0.8)
mu = features[:cut].mean(axis=0)
sd = features[:cut].std(axis=0) + 1e-6
features = (features - mu) / sd

X, y = make_sequences(
    features,
    target,
    length=args.seq_len,
)
train_n = int(len(X) * 0.8)

model = MicrostructureForecaster(
    X.shape[-1],
    hidden_dim=args.hidden,
    depth=args.depth,
    grid_size=args.grid_size,
)

train_result = train(
    model,
    SequenceDataset(X[:train_n], y[:train_n]),
    TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        grid_update_interval=100,
    ),
    distributed=args.ddp,
)

if args.ddp and int(os.environ.get("RANK", "0")) != 0:
    raise SystemExit(0)

model.eval()
with torch.no_grad():
    pred = (
        model.cpu()(torch.from_numpy(X[train_n:]))
        .squeeze(-1)
        .numpy()
    )

true = y[train_n:]
mse = float(np.mean((pred - true) ** 2))
ic = (
    float(np.corrcoef(pred, true)[0, 1])
    if pred.std() > 0 and true.std() > 0
    else 0.0
)

metrics = {
    "events": args.events,
    "sequences": len(X),
    "train_sequences": train_n,
    "test_sequences": len(X) - train_n,
    "target_units": "basis points per event",
    "test_mse_bps2": mse,
    "test_rmse_bps": float(np.sqrt(mse)),
    "test_information_coefficient": ic,
    "device": train_result["device"],
    "world_size": train_result["world_size"],
    "features": X.shape[-1],
}

Path("results").mkdir(exist_ok=True)
Path("results/model_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)
print(json.dumps(metrics, indent=2))