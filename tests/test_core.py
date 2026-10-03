import torch
import numpy as np

from tkan_engine.data import SyntheticLOBStream
from tkan_engine.features import LOBFeatureExtractor
from tkan_engine.kan import KANLinear, TemporalKAN


def test_bsplines_partition():
    layer = KANLinear(3, 4, grid_size=6, spline_order=3)
    x = torch.rand(32, 3) * 2 - 1
    basis = layer.b_splines(x)
    assert basis.shape == (32, 3, 9)
    assert torch.allclose(
        basis.sum(-1),
        torch.ones(32, 3),
        atol=1e-5,
    )


def test_forward_and_backward():
    layer = KANLinear(4, 5)
    x = torch.randn(8, 4, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()
    assert y.shape == (8, 5)
    assert x.grad is not None


def test_adaptive_grid_changes_knots():
    layer = KANLinear(2, 2, grid_size=5)
    old = layer.grid.clone()
    x = torch.randn(200, 2) * 0.2 + 2.0
    layer.update_grid(x)
    assert not torch.equal(old, layer.grid)


def test_temporal_kan_shape():
    model = TemporalKAN(7, hidden_dim=16, depth=2)
    x = torch.randn(4, 12, 7)
    assert model(x).shape == (4, 1)


def test_lob_feature_schema():
    df = next(SyntheticLOBStream(128).batches())
    out = LOBFeatureExtractor.transform(df, 10)
    assert "queue_imbalance" in out
    assert "microprice_dev_bps" in out
    assert len(out) == 128


def test_stream_continuity():
    batches = list(
        SyntheticLOBStream(
            250,
            batch_size=125,
        ).batches()
    )
    assert len(batches) == 2
    assert abs(
        float(batches[1]["mid"].iloc[0])
        - float(batches[0]["mid"].iloc[-1])
    ) < 0.01


def test_optional_arrow_message():
    from tkan_engine import arrow_pipeline

    if arrow_pipeline.pa is None:
        try:
            arrow_pipeline.require_arrow()
        except RuntimeError as exc:
            assert "pyarrow" in str(exc)
