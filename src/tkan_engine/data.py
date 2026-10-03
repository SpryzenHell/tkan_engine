from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LOBConfig:
    levels: int = 10
    seed: int = 7


class SyntheticLOBStream:
    """Deterministic L2-like event generator for stress tests and examples."""

    def __init__(
        self,
        events: int,
        config: LOBConfig = LOBConfig(),
        batch_size: int = 100_000,
    ) -> None:
        self.events = int(events)
        self.config = config
        self.batch_size = int(batch_size)

    def batches(self) -> Iterator[pd.DataFrame]:
        rng = np.random.default_rng(self.config.seed)
        levels = self.config.levels
        remaining = self.events
        emitted = 0
        timestamp = 1_700_000_000_000_000_000
        mid_level = 100.0

        while remaining:
            n = min(self.batch_size, remaining)
            global_t = emitted + np.arange(n)

            # Synthetic microstructure process with a controlled nonlinear
            # order-flow response. This is not exchange replay data.
            latent_imb = rng.normal(0.0, 1.0, n) + 1.2 * np.sin(global_t / 71.0)
            imbalance = np.tanh(latent_imb)
            nonlinear_signal = (
                1.6e-5 * np.sin(3.0 * imbalance)
                + 1.0e-5 * imbalance**3
            )
            ret = (
                rng.normal(0.0, 1.8e-5, n) + nonlinear_signal
            ).astype(np.float64)

            mid = mid_level + np.cumsum(ret)
            mid_level = float(mid[-1])

            spread = np.maximum(
                0.01,
                0.04 + 0.01 * rng.lognormal(0.0, 0.25, n),
            )
            bid1 = mid - spread / 2
            ask1 = mid + spread / 2

            bid_sz = rng.lognormal(
                2.2 + 0.6 * imbalance[:, None],
                0.25,
                (n, levels),
            ).astype(np.float32)
            ask_sz = rng.lognormal(
                2.2 - 0.6 * imbalance[:, None],
                0.25,
                (n, levels),
            ).astype(np.float32)

            bid_px = bid1[:, None] - np.arange(levels)[None, :] * 0.01
            ask_px = ask1[:, None] + np.arange(levels)[None, :] * 0.01
            trade_side = np.where(imbalance > 0, 1, -1).astype(np.int8)

            frame = {
                "ts_ns": np.arange(
                    timestamp,
                    timestamp + n * 1_000,
                    1_000,
                    dtype=np.int64,
                ),
                "mid": mid,
                "bid_px_1": bid1,
                "ask_px_1": ask1,
                "bid_sz_1": bid_sz[:, 0],
                "ask_sz_1": ask_sz[:, 0],
                "trade_side": trade_side,
                "trade_sz": rng.lognormal(
                    1.0, 0.4, n
                ).astype(np.float32),
            }

            for level in range(levels):
                frame[f"bid_px_{level + 1}"] = bid_px[:, level]
                frame[f"ask_px_{level + 1}"] = ask_px[:, level]
                frame[f"bid_sz_{level + 1}"] = bid_sz[:, level]
                frame[f"ask_sz_{level + 1}"] = ask_sz[:, level]

            yield pd.DataFrame(frame)
            timestamp += n * 1_000
            emitted += n
            remaining -= n


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    length: int = 64,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= length:
        raise ValueError("not enough observations for a sequence")

    starts = range(0, len(features) - length, stride)
    X = np.stack(
        [features[i:i + length] for i in starts],
        axis=0,
    ).astype(np.float32)
    y = np.asarray(
        [targets[i + length] for i in starts],
        dtype=np.float32,
    )
    return X, y