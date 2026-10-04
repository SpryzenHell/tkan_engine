from __future__ import annotations

import numpy as np
import pandas as pd

from tkan_engine.data import SequenceDataset, SyntheticLOBStream, make_sequences
from tkan_engine.features import LOBFeatureExtractor
from tkan_engine.model import MicrostructureForecaster
from tkan_engine.training import TrainConfig, train

frames = [
    LOBFeatureExtractor.transform(frame, 10)
    for frame in SyntheticLOBStream(
        5_000,
        batch_size=5_000,
    ).batches()
]
df = pd.concat(frames, ignore_index=True)
df["mid_return"] = df["mid"].pct_change().fillna(0.0)

X, y = make_sequences(
    df.drop(columns=["mid"]).to_numpy(np.float32),
    df["mid_return"].shift(-1).fillna(0.0).to_numpy(np.float32),
    length=16,
)

model = MicrostructureForecaster(
    X.shape[-1],
    hidden_dim=16,
    depth=1,
    grid_size=6,
)

print(
    train(
        model,
        SequenceDataset(X[:2000], y[:2000]),
        TrainConfig(
            epochs=1,
            batch_size=64,
            grid_update_interval=0,
        ),
        distributed=True,
    )
)
