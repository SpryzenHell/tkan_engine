from __future__ import annotations

import numpy as np
import pytest

from scripts.inversion_benchmark import causal_moving_average, make_dataset
from scripts.render_report import _parameter_count


def test_causal_moving_average_uses_only_past_values():
    values = np.arange(1, 8, dtype=np.float64)
    actual = causal_moving_average(values, window=5)
    np.testing.assert_allclose(actual, [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])


def test_causal_moving_average_is_invariant_to_future_changes():
    rng = np.random.default_rng(17)
    original = rng.normal(size=32)
    changed = original.copy()
    changed[20:] += 1000.0
    base_features = causal_moving_average(original, window=5)
    changed_features = causal_moving_average(changed, window=5)
    np.testing.assert_array_equal(base_features[:20], changed_features[:20])


def test_causal_moving_average_validates_shape_and_window():
    with pytest.raises(ValueError, match="one-dimensional"):
        causal_moving_average(np.ones((2, 3)))
    with pytest.raises(ValueError, match="window"):
        causal_moving_average(np.ones(4), window=0)


def test_inversion_dataset_is_deterministic_and_has_chronological_split():
    first_x, first_y, first_split = make_dataset(400, 8, seed=91, noise_std=0.08)
    second_x, second_y, second_split = make_dataset(400, 8, seed=91, noise_std=0.08)
    assert first_x.shape == (392, 8, 3)
    assert first_y.shape == (392,)
    assert first_split == second_split == int(0.8 * 392)
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


@pytest.mark.parametrize(
    ("events", "seq_len", "noise_std", "persistence"),
    [
        (10, 10, 0.08, 0.85),
        (100, 8, -0.1, 0.85),
        (100, 8, 0.08, 1.0),
    ],
)
def test_inversion_data_rejects_invalid_settings(events, seq_len, noise_std, persistence):
    from scripts.inversion_benchmark import make_dataset as build

    with pytest.raises(ValueError):
        build(events, seq_len, seed=1, noise_std=noise_std, persistence=persistence)


def test_report_accepts_actual_benchmark_and_saved_result_schemas():
    # The benchmark writes parameter counts at the top level; older checked-in
    # result files store them inside each model record. Both formats are valid.
    actual = {"mlp_parameters": 801, "tkan_parameters": 3266}
    saved = {"mlp": {"parameters": 801}, "tkan": {"parameters": 3266}}
    assert _parameter_count(actual, "mlp") == 801
    assert _parameter_count(actual, "tkan") == 3266
    assert _parameter_count(saved, "mlp") == 801
    assert _parameter_count(saved, "tkan") == 3266


def test_report_fails_clearly_if_parameter_count_is_missing():
    with pytest.raises(ValueError, match="parameter count"):
        _parameter_count({"mlp": {}}, "mlp")


def test_sweep_summary_calculates_paired_rmse_gains():
    from scripts.inversion_sweep import summarize

    runs = [
        {"noise_std": 0.08, "seed": 1,
         "mlp": {"test_rmse_bps": 1.0, "test_ic": 0.2, "test_mse_bps2": 1.0, "parameters": 10},
         "tkan": {"test_rmse_bps": 0.8, "test_ic": 0.3, "test_mse_bps2": 0.64, "parameters": 20}},
        {"noise_std": 0.08, "seed": 2,
         "mlp": {"test_rmse_bps": 0.9, "test_ic": 0.25, "test_mse_bps2": 0.81, "parameters": 10},
         "tkan": {"test_rmse_bps": 0.95, "test_ic": 0.24, "test_mse_bps2": 0.9025, "parameters": 20}},
    ]
    summary = summarize(runs, [0.08])
    assert len(summary) == 1
    result = summary[0]
    assert result["n_seeds"] == 2
    assert result["paired_rmse_gain_mlp_minus_tkan"]["tkan_wins"] == 1
    assert result["paired_rmse_gain_mlp_minus_tkan"]["mlp_wins"] == 1
    assert result["paired_rmse_gain_mlp_minus_tkan"]["mean_bps"] == pytest.approx(0.075)
