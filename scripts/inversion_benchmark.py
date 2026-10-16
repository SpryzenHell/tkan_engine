from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tkan_engine.data import SequenceDataset, make_sequences
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import TrainConfig, train


def causal_moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Trailing average using only the current and earlier observations."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window <= 0:
        raise ValueError("window must be positive")
    if values.size == 0:
        return values.copy()

    cumulative = np.cumsum(values, dtype=np.float64)
    sums = cumulative.copy()
    if window < len(values):
        sums[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return sums / counts


def make_dataset(
    events: int,
    seq_len: int,
    seed: int = 123,
    noise_std: float = 0.08,
    persistence: float = 0.85,
):
    if events <= seq_len + 1:
        raise ValueError("events must exceed seq_len by at least two")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must be in [0, 1)")

    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=events)
    latent = np.empty(events, dtype=np.float64)
    latent[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - persistence**2)
    for index in range(1, events):
        latent[index] = (
            persistence * latent[index - 1]
            + innovation_scale * innovations[index]
        )

    ofi = np.tanh(latent)
    previous = np.r_[0.0, ofi[:-1]]
    moving = causal_moving_average(ofi, window=5)

    clean = (
        0.9 * np.sin(2.6 * ofi)
        + 0.55 * ofi**3
        + 0.25 * (ofi - previous)
    )
    noise = np.random.default_rng(seed + 1).normal(size=events)
    target = clean + noise_std * noise

    features = np.column_stack([ofi, ofi - previous, moving]).astype(np.float32)
    X, y = make_sequences(
        features,
        target.astype(np.float32),
        length=seq_len,
    )
    split = int(0.8 * len(X))
    if split <= 0 or split >= len(X):
        raise ValueError("not enough sequences for a non-empty chronological split")
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
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = (
            model(torch.from_numpy(X[split:]).to(device))
            .squeeze(-1)
            .detach()
            .cpu()
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


def run_benchmark(
    events: int = 3_000,
    seq_len: int = 16,
    epochs: int = 5,
    batch_size: int = 128,
    hidden: int = 16,
    grid_size: int = 6,
    seed: int = 123,
    noise_std: float = 0.08,
    out: str | Path | None = None,
) -> dict:
    if epochs <= 0 or batch_size <= 0 or hidden <= 0 or grid_size <= 0:
        raise ValueError("epochs, batch_size, hidden, and grid_size must be positive")

    X, y, split = make_dataset(
        events,
        seq_len,
        seed=seed,
        noise_std=noise_std,
    )

    torch.manual_seed(seed + 7)
    start = time.perf_counter()
    mlp = fit_mlp(X, y, split, hidden, epochs, batch_size, 3e-3)
    mlp_seconds = time.perf_counter() - start

    torch.manual_seed(seed + 123)
    start = time.perf_counter()
    tkan = MicrostructureForecaster(
        X.shape[-1],
        hidden_dim=hidden,
        depth=1,
        grid_size=grid_size,
    )
    tkan_training = train(
        tkan,
        SequenceDataset(X[:split], y[:split]),
        TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=3e-3,
            grid_update_interval=50,
        ),
    )
    tkan_seconds = time.perf_counter() - start

    mlp_metrics = score(mlp, X, y, split)
    tkan_metrics = score(tkan, X, y, split)
    mlp_parameters = sum(parameter.numel() for parameter in mlp.parameters())
    tkan_parameters = sum(parameter.numel() for parameter in tkan.parameters())
    mlp_metrics["parameters"] = mlp_parameters
    tkan_metrics["parameters"] = tkan_parameters

    result = {
        "events": events,
        "sequence_length": seq_len,
        "train_sequences": split,
        "test_sequences": len(X) - split,
        "features": int(X.shape[-1]),
        "seed": seed,
        "noise_std": noise_std,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "mlp_device": str(next(mlp.parameters()).device),
            "tkan_device": str(next(tkan.parameters()).device),
        },
        "ofi_persistence": 0.85,
        "target_units": "synthetic impact basis points",
        "generator": "OFI is tanh of a stationary AR(1) process (phi=0.85); target = 0.9*sin(2.6*OFI) + 0.55*OFI^3 + 0.25*dOFI + N(0, noise_std^2); features use trailing, causal windows.",
        "mlp_parameters": mlp_parameters,
        "tkan_parameters": tkan_parameters,
        "mlp_seconds": mlp_seconds,
        "tkan_seconds": tkan_seconds,
        "mlp": mlp_metrics,
        "tkan": tkan_metrics,
        "tkan_training_history": tkan_training["history"],
        "notes": "Controlled synthetic next-step prediction with a chronological split and no future-looking input features. Not exchange data and not a trading-alpha claim.",
    }

    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=3_000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--out", default="results/inversion_benchmark.json")
    args = parser.parse_args(argv)

    result = run_benchmark(
        events=args.events,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        grid_size=args.grid_size,
        seed=args.seed,
        noise_std=args.noise_std,
        out=args.out,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
