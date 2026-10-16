from __future__ import annotations

import pytest

from scripts.update_experiment_report import (
    arrow_markdown,
    ddp_markdown,
    inversion_markdown,
    replace_block,
    sweep_markdown,
    validation_markdown,
)


def sample_inversion():
    return {
        "mlp_parameters": 12,
        "tkan_parameters": 24,
        "mlp": {"test_mse_bps2": 1.0, "test_rmse_bps": 1.0, "test_ic": 0.5},
        "tkan": {"test_mse_bps2": 0.64, "test_rmse_bps": 0.8, "test_ic": 0.6},
    }


def sample_sweep():
    return {
        "seeds": [1, 2, 3],
        "noise_levels": [0.08],
        "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
        "summary": [{
            "noise_std": 0.08,
            "n_seeds": 3,
            "mlp": {"mean_rmse_bps": 1.1, "std_rmse_bps": 0.1, "mean_ic": 0.4, "std_ic": 0.02},
            "tkan": {"mean_rmse_bps": 0.9, "std_rmse_bps": 0.05, "mean_ic": 0.5, "std_ic": 0.03},
            "paired_rmse_gain_mlp_minus_tkan": {"mean_bps": 0.2, "std_bps": 0.1, "tkan_wins": 3, "mlp_wins": 0, "ties": 0},
        }],
    }


def sample_arrow(events: int, batch_size: int):
    return {"events": events, "batch_size": batch_size, "batches": events // batch_size,
            "elapsed_s": 1.0, "events_per_s": float(events), "selected_rows": events,
            "checksum": 12.5}


def sample_ddp():
    return {"device": "cpu", "device_name": "cpu", "world_size": 2,
            "global_samples": 256, "wall_seconds": 2.0, "samples_per_s": 128.0,
            "local_batch_size": 64, "warmup_steps": 2, "measured_steps": 2}


def sample_matrix():
    return {"ci_run_id": 123, "source_commit": "0123456789abcdef",
            "python_versions": {version: {"tests": 43, "passed": 43, "skipped": 0,
                                            "failures": 0, "errors": 0}
                                 for version in ("3.10", "3.11", "3.12")}}


def test_inversion_markdown_reports_measured_counts_and_direction():
    report = inversion_markdown(sample_inversion())
    assert "| MLP | 12 | 1.000000 | 1.000000 | 0.500000 |" in report
    assert "| T-KAN | 24 | 0.640000 | 0.800000 | 0.600000 |" in report
    assert "20.0% lower" in report


def test_sweep_markdown_reports_repeat_count_and_plots():
    report = sweep_markdown(sample_sweep())
    assert "3 model-pair runs" in report
    assert "0.2000 ± 0.1000 bps" in report
    assert "docs/figures/inversion_noise_rmse.svg" in report
    assert "confidence intervals" in report


def test_arrow_and_ddp_tables_use_the_current_artifact_values():
    arrow = arrow_markdown(sample_arrow(5_000_000, 250_000), sample_arrow(300_000_000, 1_000_000))
    ddp = ddp_markdown(sample_ddp())
    assert "5,000,000" in arrow and "300,000,000" in arrow
    assert "1.000M" in arrow
    assert "256" in ddp and "128.000" in ddp
    assert "CPU results do not measure GPU speed" in ddp


def test_validation_table_reports_each_python_version_and_run_link():
    report = validation_markdown(sample_matrix(), sample_arrow(5_000_000, 250_000),
                                 sample_arrow(300_000_000, 1_000_000), sample_ddp(), sample_sweep())
    assert "actions/runs/123" in report
    assert "| 3.10 | 43 | 43 | 0 | 0 | 0 |" in report
    assert "9 model-pair runs" in report
    assert "12 SVG charts and 12 PNG copies" in report


def test_replace_block_fails_on_missing_or_duplicate_markers(tmp_path):
    path = tmp_path / "example.md"
    path.write_text("before START old END after", encoding="utf-8")
    replace_block(path, "START", "END", "new content")
    assert path.read_text(encoding="utf-8") == "before START\nnew content\nEND after"
    path.write_text("START x START y END", encoding="utf-8")
    with pytest.raises(ValueError, match="one pair"):
        replace_block(path, "START", "END", "new")
