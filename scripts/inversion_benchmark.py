from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tkan_engine.data import SequenceDataset, make_sequences
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import TrainConfig, train


def make_dataset(events: int, seq_len: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    ofi = np.tanh(rng.normal(size=events))
    previous = np.r_[0.0, ofi[:-1]]
    moving = np.convolve(ofi, np.ones(5) / 5, mode="same")

    clean = (
        0.9 * np.sin(2.6 * ofi)
        + 0.55 * ofi**3
        + 0.25 * (ofi - previous)
    )
    noise = np.random.default_rng(seed + 1).normal(size=events)
    target = clean + 0.08 * noise

    features = np.column_stack(
        [ofi, ofi - previous, moving]
    ).astype(np.float32)

    X, y = make_sequences(
        features,
        target.astype(np.float32),
        length=seq_len,
    )
    split = int(0.8 * len(X))
    return X, y, split


class MLP(nn.Module):
    def __init__(self, features: int, seq_len: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features * seq_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def score(model, X, y, split):
    model.eval()
    with torch.no_grad():
        pred = (
            model(torch.from_numpy(X[split:]))
            .squeeze(-1)
            .numpy()
        )
    true = y[split:]
    mse = float(np.mean((pred - true) ** 2))
    ic = (
        float(np.corrcoef(pred, true)[0, 1])
        if pred.std() > 0 and true.std() > 0
        else 0.0
    )
    return {
        "test_mse_bps2": mse,
        "test_rmse_bps": float(np.sqrt(mse)),
        "test_ic": ic,
    }


def fit_mlp(X, y, split, hidden, epochs, batch_size, lr):
    model = MLP(X.shape[-1], X.shape[1], hidden)
    loader = torch.utils.data.DataLoader(
        SequenceDataset(X[:split], y[:split]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


p = argparse.ArgumentParser()
p.add_argument("--events", type=int, default=3_000)
p.add_argument("--seq-len", type=int, default=16)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--hidden", type=int, default=16)
p.add_argument("--grid-size", type=int, default=6)
p.add_argument("--out", default="results/inversion_benchmark.json")
a = p.parse_args()

X, y, split = make_dataset(a.events, a.seq_len)

torch.manual_seed(7)
start = time.perf_counter()
mlp = fit_mlp(
    X, y, split,
    a.hidden, a.epochs, a.batch_size, 3e-3
)
mlp_seconds = time.perf_counter() - start

torch.manual_seed(123)
start = time.perf_counter()
tkan = MicrostructureForecaster(
    X.shape[-1],
    hidden_dim=a.hidden,
    depth=1,
    grid_size=a.grid_size,
)
tkan_training = train(
    tkan,
    SequenceDataset(X[:split], y[:split]),
    TrainConfig(
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=3e-3,
        grid_update_interval=50,
    ),
)
tkan_seconds = time.perf_counter() - start

result = {
    "events": a.events,
    "sequence_length": a.seq_len,
    "train_sequences": split,
    "test_sequences": len(X) - split,
    "features": int(X.shape[-1]),
    "target_units": "synthetic impact basis points",
    "generator": "0.9*sin(2.6*OFI) + 0.55*OFI^3 + 0.25*dOFI + N(0,0.08^2)",
    "mlp_parameters": sum(p.numel() for p in mlp.parameters()),
    "tkan_parameters": sum(p.numel() for p in tkan.parameters()),
    "mlp_seconds": mlp_seconds,
    "tkan_seconds": tkan_seconds,
    "mlp": score(mlp, X, y, split),
    "tkan": score(tkan, X, y, split),
    "tkan_training_history": tkan_training["history"],
    "notes": "Controlled synthetic inversion benchmark; not exchange data and not a trading-alpha claim.",
}

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
