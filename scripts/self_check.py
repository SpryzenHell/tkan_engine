from __future__ import annotations

import importlib.util
import platform
import sys

import torch

from tkan_engine.data import SyntheticLOBStream
from tkan_engine.features import LOBFeatureExtractor
from tkan_engine.model import MicrostructureForecaster


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyArrow installed: {bool(importlib.util.find_spec('pyarrow'))}")

    frame = next(SyntheticLOBStream(32, batch_size=32).batches())
    features = LOBFeatureExtractor.transform(frame, levels=10)
    x = torch.from_numpy(features.drop(columns=["mid"]).to_numpy()).unsqueeze(0)
    model = MicrostructureForecaster(x.shape[-1], hidden_dim=16, depth=1, grid_size=6)
    with torch.no_grad():
        y = model(x)
    print(f"Synthetic LOB rows: {len(frame)}")
    print(f"Feature dimension: {x.shape[-1]}")
    print(f"Model output shape: {tuple(y.shape)}")
    assert y.shape == (1, 1)
    print("Self-check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
