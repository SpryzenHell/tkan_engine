from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def replace_block(path: Path, start_marker: str, end_marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError(f"Expected one pair of report markers in {path}")
    before, rest = text.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    path.write_text(
        before + start_marker + "\n" + body.strip() + "\n" + end_marker + after,
        encoding="utf-8",
    )


def inversion_markdown(result: dict) -> str:
    rows = []
    for label in ("mlp", "tkan"):
        row = result[label]
        params = int(row.get("parameters", result.get(f"{label}_parameters", 0)))
        display = "MLP" if label == "mlp" else "T-KAN"
        rows.append(
            f"| {display} | {params:,} | {float(row['test_mse_bps2']):.6f} | "
            f"{float(row['test_rmse_bps']):.6f} | {float(row['test_ic']):.6f} |"
        )
    mlp_rmse = float(result["mlp"]["test_rmse_bps"])
    tkan_rmse = float(result["tkan"]["test_rmse_bps"])
    change = (mlp_rmse - tkan_rmse) / mlp_rmse * 100.0 if mlp_rmse > 0 else 0.0
    if change > 0:
        interpretation = (
            f"On this run, T-KAN's test RMSE was {change:.1f}% lower than the MLP's. "
            "This is a result for the selected synthetic generator and seed only."
        )
    elif change < 0:
        interpretation = (
            f"On this run, T-KAN's test RMSE was {abs(change):.1f}% higher than the MLP's. "
            "The comparison does not show a T-KAN error reduction for this seed."
        )
    else:
        interpretation = "Both models had the same test RMSE on this run."
    return "\n".join([
        "| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        interpretation,
        "The split is chronological. The moving-average input is trailing and uses only current or earlier observations.",
    ])


def sweep_markdown(result: dict) -> str:
    summary = result.get("summary", [])
    if not summary:
        raise ValueError("The inversion sweep has no summary entries")
    lines = [
        "| Target noise SD | Seeds | MLP RMSE (mean ± SD) | T-KAN RMSE (mean ± SD) | MLP IC (mean ± SD) | T-KAN IC (mean ± SD) | Mean RMSE gain (MLP − T-KAN) | T-KAN wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        gain = row["paired_rmse_gain_mlp_minus_tkan"]
        lines.append(
            f"| {float(row['noise_std']):.2f} | {int(row['n_seeds'])} | "
            f"{float(row['mlp']['mean_rmse_bps']):.4f} ± {float(row['mlp']['std_rmse_bps']):.4f} | "
            f"{float(row['tkan']['mean_rmse_bps']):.4f} ± {float(row['tkan']['std_rmse_bps']):.4f} | "
            f"{float(row['mlp']['mean_ic']):.4f} ± {float(row['mlp']['std_ic']):.4f} | "
            f"{float(row['tkan']['mean_ic']):.4f} ± {float(row['tkan']['std_ic']):.4f} | "
            f"{float(gain['mean_bps']):+.4f} ± {float(gain['std_bps']):.4f} bps | "
            f"{int(gain['tkan_wins'])}/{int(row['n_seeds'])} |"
        )
    lines.extend([
        "",
        f"This sweep includes {len(result.get('runs', []))} model-pair runs across "
        f"{len(result.get('seeds', []))} seeds and {len(summary)} noise settings. "
        "Reported SD values measure variation across these runs; they are not confidence intervals. "
        "Positive RMSE gain means T-KAN had lower error, while negative gain means the MLP had lower error.",
        "",
        "![RMSE by noise level](docs/figures/inversion_noise_rmse.svg)",
        "",
        "![Information coefficient by noise level](docs/figures/inversion_noise_ic.svg)",
        "",
        "![RMSE by random seed](docs/figures/inversion_seed_rmse.svg)",
        "",
        "![Paired RMSE difference](docs/figures/inversion_rmse_gain.svg)",
        "",
        "Full per-seed values and training times are recorded in results/inversion_sweep.json.",
    ])
    return "\n".join(lines)


def main() -> int:
    root = Path(".")
    inversion = read_json(root / "results/ci_inversion_3000.json")
    sweep = read_json(root / "results/inversion_sweep.json")
    metric_body = inversion_markdown(inversion)
    sweep_body = sweep_markdown(sweep)
    targets = [
        (root / "README.md", "INVERSION_METRICS_START", "INVERSION_METRICS_END", "INVERSION_SWEEP_START", "INVERSION_SWEEP_END"),
        (root / "docs/EXPERIMENTS.md", "INVERSION_METRICS_START", "INVERSION_METRICS_END", "INVERSION_SWEEP_START", "INVERSION_SWEEP_END"),
    ]
    for path, metrics_start, metrics_end, sweep_start, sweep_end in targets:
        replace_block(path, f"<!-- {metrics_start} -->", f"<!-- {metrics_end} -->", metric_body)
        replace_block(path, f"<!-- {sweep_start} -->", f"<!-- {sweep_end} -->", sweep_body)
    print("Updated measured inversion tables and sweep analysis in README.md and docs/EXPERIMENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
