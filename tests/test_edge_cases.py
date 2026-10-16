from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tkan_engine.data import LOBConfig, SequenceDataset, SyntheticLOBStream, make_sequences
from tkan_engine.features import LOBFeatureExtractor
from tkan_engine.kan import KANLinear, TemporalKAN


@pytest.mark.parametrize(
    ("events", "batch_size"),
    [(-1, 8), (10, 0), (10, -1)],
)
def test_synthetic_lob_stream_rejects_invalid_sizes(events, batch_size):
    with pytest.raises(ValueError):
        SyntheticLOBStream(events, batch_size=batch_size)


def test_synthetic_lob_stream_rejects_zero_levels():
    with pytest.raises(ValueError, match="levels"):
        SyntheticLOBStream(10, config=LOBConfig(levels=0))


def test_synthetic_lob_stream_exact_batch_counts_and_reproducibility():
    first = list(SyntheticLOBStream(17, config=LOBConfig(seed=13), batch_size=8).batches())
    second = list(SyntheticLOBStream(17, config=LOBConfig(seed=13), batch_size=8).batches())
    assert [len(batch) for batch in first] == [8, 8, 1]
    assert [len(batch) for batch in second] == [8, 8, 1]
    for lhs, rhs in zip(first, second):
        pd.testing.assert_frame_equal(lhs, rhs)


def test_synthetic_lob_zero_events_is_empty():
    assert list(SyntheticLOBStream(0, batch_size=4).batches()) == []


def test_feature_values_are_finite_and_imbalances_are_bounded():
    frame = next(SyntheticLOBStream(128, batch_size=128).batches())
    features = LOBFeatureExtractor.transform(frame, levels=10)
    assert len(features) == len(frame)
    assert features.shape[1] == 16
    assert np.isfinite(features.to_numpy()).all()
    assert (features["queue_imbalance"].abs() <= 1.0).all()
    assert (features["spread_bps"] > 0).all()


def test_make_sequences_stride_and_target_alignment():
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    targets = np.arange(10, dtype=np.float32) * 2
    x, y = make_sequences(features, targets, length=2, stride=2)
    assert x.shape == (4, 2, 3)
    np.testing.assert_array_equal(y, np.array([4, 8, 12, 16], dtype=np.float32))
    np.testing.assert_array_equal(x[1], features[2:4])


@pytest.mark.parametrize(
    ("length", "stride"),
    [(0, 1), (-1, 1), (2, 0), (2, -1)],
)
def test_make_sequences_rejects_invalid_window(length, stride):
    with pytest.raises(ValueError):
        make_sequences(np.ones((8, 2)), np.ones(8), length=length, stride=stride)


def test_make_sequences_rejects_mismatched_targets():
    with pytest.raises(ValueError, match="same number"):
        make_sequences(np.ones((8, 2)), np.ones(7), length=2)


def test_sequence_dataset_rejects_mismatched_or_empty_data():
    with pytest.raises(ValueError, match="same number"):
        SequenceDataset(np.ones((4, 3, 2)), np.ones(3))
    with pytest.raises(ValueError, match="must not be empty"):
        SequenceDataset(np.empty((0, 3, 2)), np.empty((0,)))


def test_temporal_kan_gradients_remain_finite():
    torch.manual_seed(23)
    model = TemporalKAN(input_dim=5, hidden_dim=8, output_dim=1, depth=2, grid_size=5)
    x = torch.randn(6, 9, 5, requires_grad=True)
    y = model(x)
    (y.square().mean() + model.regularization_loss()).backward()
    assert y.shape == (6, 1)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    parameters_with_grad = [p.grad for p in model.parameters() if p.grad is not None]
    assert parameters_with_grad
    assert all(torch.isfinite(grad).all() for grad in parameters_with_grad)


def test_adaptive_grid_keeps_model_output_finite():
    torch.manual_seed(31)
    layer = KANLinear(3, 4, grid_size=6, spline_order=3)
    inputs = torch.randn(128, 3) * 0.4 + 1.2
    layer.update_grid(inputs)
    output = layer(inputs)
    assert output.shape == (128, 4)
    assert torch.isfinite(output).all()
    assert torch.isfinite(layer.grid).all()


def test_arrow_stream_validation_batching_and_accounting():
    pa = pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    batches = list(ap.synthetic_arrow_stream(101, batch_size=32, seed=9))
    assert [batch.num_rows for batch in batches] == [32, 32, 32, 5]
    first = ap.process_batches(ap.synthetic_arrow_stream(101, batch_size=32, seed=9))
    second = ap.process_batches(ap.synthetic_arrow_stream(101, batch_size=32, seed=9))
    assert first.events == 101
    assert first.batches == 4
    assert first.selected_rows <= first.events
    assert first.elapsed_s > 0
    assert first.events_per_s > 0
    assert np.isfinite(first.checksum)
    assert second.events == first.events
    assert second.selected_rows == first.selected_rows
    assert second.checksum == pytest.approx(first.checksum, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("events", "batch_size"),
    [(-1, 16), (4, 0), (4, -5)],
)
def test_arrow_generator_rejects_invalid_sizes(events, batch_size):
    pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    with pytest.raises(ValueError):
        list(ap.synthetic_arrow_stream(events, batch_size=batch_size))


def test_arrow_empty_stream_has_zero_throughput():
    pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    result = ap.process_batches(ap.synthetic_arrow_stream(0, batch_size=32))
    assert result.events == 0
    assert result.batches == 0
    assert result.selected_rows == 0
    assert result.events_per_s == 0.0
    assert result.checksum == 0.0


def test_arrow_integer_queue_sizes_and_stale_spread_column():
    pa = pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    batch = pa.record_batch({
        "ts_ns": pa.array([1, 2], type=pa.int64()),
        "bid_px_1": pa.array([100.0, 101.0]),
        "ask_px_1": pa.array([100.02, 101.02]),
        "bid_sz_1": pa.array([3, 0], type=pa.int64()),
        "ask_sz_1": pa.array([1, 0], type=pa.int64()),
        "spread_bps": pa.array([999.0, 999.0]),
    })
    result = ap.process_batches([batch])
    assert result.events == 2
    assert result.selected_rows == 2
    assert np.isfinite(result.checksum)
    assert result.checksum < 999.0


def test_arrow_schema_errors_are_readable():
    pa = pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    batch = pa.record_batch({
        "ts_ns": pa.array([1], type=pa.int64()),
        "mid": pa.array([100.0]),
    })
    with pytest.raises(ValueError, match="missing required columns"):
        ap.process_batches([batch])


def test_arrow_within_rejects_reversed_range_and_keeps_edges():
    pa = pytest.importorskip("pyarrow")
    from tkan_engine import arrow_pipeline as ap

    table = pa.table({"x": [0.0, 1.0, 2.0, 3.0]})
    result = ap.q_within(table, "x", 1.0, 2.0)
    assert result.column("x").to_pylist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="lower"):
        ap.q_within(table, "x", 2.0, 1.0)


def test_temporal_kan_is_causal_with_respect_to_future_steps():
    """Appending future LOB observations must not change the last past prediction."""
    torch.manual_seed(20261017)
    model = TemporalKAN(
        input_dim=4, hidden_dim=8, output_dim=1, depth=2, grid_size=5, dropout=0.0
    ).eval()
    prefix = torch.randn(3, 7, 4)
    future = torch.randn(3, 5, 4) * 3.0
    def all_hidden(sequence):
        hidden = model.encoder.input_proj(sequence)
        for block in model.encoder.blocks:
            hidden = block(hidden)
        return hidden

    with torch.no_grad():
        short_prediction = model.encoder.head(all_hidden(prefix)[:, -1])
        extended_prediction_at_same_time = model.encoder.head(
            all_hidden(torch.cat([prefix, future], dim=1))[:, 6]
        )
    assert short_prediction.shape == extended_prediction_at_same_time.shape == (3, 1)
    assert torch.allclose(
        short_prediction, extended_prediction_at_same_time, atol=1e-6, rtol=1e-6
    )


def test_temporal_kan_output_and_gradients_stay_finite_after_grid_update():
    torch.manual_seed(20261018)
    model = TemporalKAN(input_dim=5, hidden_dim=8, output_dim=1, depth=2, grid_size=5)
    inputs = torch.randn(12, 10, 5) * 0.3 + 1.0
    model.update_grids(inputs[:4])
    prediction = model(inputs)
    loss = prediction.square().mean() + model.regularization_loss()
    loss.backward()
    assert prediction.shape == (12, 1)
    assert torch.isfinite(prediction).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
