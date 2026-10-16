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


def arrow_markdown(small: dict, large: dict) -> str:
    rows = []
    for result in (small, large):
        rows.append(
            f"| {int(result['events']):,} | {int(result['batch_size']):,} | "
            f"{int(result['batches']):,} | {float(result['elapsed_s']):.3f} | "
            f"{float(result['events_per_s']) / 1_000_000:.3f}M | "
            f"{int(result['selected_rows']):,} | {float(result['checksum']):,.4f} |"
        )
    return "\n".join([
        "| Events | Batch size | Batches | Elapsed (s) | Throughput | Selected rows | Checksum |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "These are fresh runs over generated events, not replayed exchange messages. They ran in separate CI jobs, so rates are host-dependent and are not a controlled scaling comparison. Full environment and source commit are saved in the JSON result files.",
    ])


def ddp_markdown(result: dict) -> str:
    return "\n".join([
        "| Device | Ranks | Global samples | Wall time (s) | Samples/s | Local batch | Warmup steps | Measured steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {result.get('device_name', result.get('device', 'unknown'))} | "
        f"{int(result['world_size'])} | {int(result['global_samples']):,} | "
        f"{float(result['wall_seconds']):.3f} | {float(result['samples_per_s']):.3f} | "
        f"{int(result['local_batch_size']):,} | {int(result.get('warmup_steps', 0))} | "
        f"{int(result.get('measured_steps', result.get('steps', 0)))} |",
        "",
        "This is a distributed smoke test on the device listed above. CPU results do not measure GPU speed or multi-GPU scaling.",
    ])


def validation_markdown(matrix: dict, small: dict, large: dict, ddp: dict, sweep: dict) -> str:
    run_id = int(matrix["ci_run_id"])
    commit = str(matrix["source_commit"])
    run_url = f"https://github.com/SpryzenHell/tkan_engine/actions/runs/{run_id}"
    versions = matrix.get("python_versions", {})
    lines = [
        f"[GitHub Actions run {run_id}]({run_url}) was generated from commit `{commit[:12]}`. "
        "The report job starts only after the Python tests, Arrow stress runs, CPU DDP test, and sensitivity sweep have all succeeded.",
        "",
        "| Python | Tests | Passed | Skipped | Failures | Errors |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for version in ("3.10", "3.11", "3.12"):
        row = versions[version]
        lines.append(
            f"| {version} | {int(row['tests'])} | {int(row['passed'])} | "
            f"{int(row['skipped'])} | {int(row['failures'])} | {int(row['errors'])} |"
        )
    lines.extend([
        "",
        "| Additional check | Measured result |",
        "|---|---|",
        f"| Arrow stream | {int(small['events']):,} and {int(large['events']):,} generated events processed in batches |",
        f"| CPU DDP | {int(ddp['world_size'])} ranks, {int(ddp['global_samples']):,} global samples |",
        f"| Seed/noise experiment | {len(sweep.get('runs', []))} model-pair runs across {len(sweep.get('seeds', []))} seeds and {len(sweep.get('noise_levels', []))} noise settings |",
        "| Charts | 12 SVG charts and 12 PNG copies generated; the same job checks all files |",
        "",
        "All data results, test counts, and plots are saved in this repository. Benchmark data are synthetic; no live-market or A100 performance claim is implied by these checks.",
    ])
    return "\n".join(lines)


def main() -> int:
    root = Path(".")
    inversion = read_json(root / "results/ci_inversion_3000.json")
    sweep = read_json(root / "results/inversion_sweep.json")
    matrix = read_json(root / "results/test_matrix.json")
    arrow_5m = read_json(root / "results/arrow_benchmark_5m.json")
    arrow_300m = read_json(root / "results/arrow_benchmark_300m.json")
    ddp = read_json(root / "results/ddp_benchmark_cpu.json")

    metrics_body = inversion_markdown(inversion)
    sweep_body = sweep_markdown(sweep)
    arrow_body = arrow_markdown(arrow_5m, arrow_300m)
    ddp_body = ddp_markdown(ddp)
    validation_body = validation_markdown(matrix, arrow_5m, arrow_300m, ddp, sweep)

    replace_block(root / "README.md", "<!-- CI_VALIDATION_START -->", "<!-- CI_VALIDATION_END -->", validation_body)
    replace_block(root / "README.md", "<!-- INVERSION_METRICS_START -->", "<!-- INVERSION_METRICS_END -->", metrics_body)
    replace_block(root / "README.md", "<!-- INVERSION_SWEEP_START -->", "<!-- INVERSION_SWEEP_END -->", sweep_body)
    replace_block(root / "README.md", "<!-- ARROW_METRICS_START -->", "<!-- ARROW_METRICS_END -->", arrow_body)
    replace_block(root / "README.md", "<!-- DDP_METRICS_START -->", "<!-- DDP_METRICS_END -->", ddp_body)

    replace_block(root / "docs/EXPERIMENTS.md", "<!-- INVERSION_METRICS_START -->", "<!-- INVERSION_METRICS_END -->", metrics_body)
    replace_block(root / "docs/EXPERIMENTS.md", "<!-- INVERSION_SWEEP_START -->", "<!-- INVERSION_SWEEP_END -->", sweep_body)
    replace_block(root / "docs/EXPERIMENTS.md", "<!-- ARROW_METRICS_START -->", "<!-- ARROW_METRICS_END -->", arrow_body)
    replace_block(root / "docs/EXPERIMENTS.md", "<!-- DDP_METRICS_START -->", "<!-- DDP_METRICS_END -->", ddp_body)

    replace_block(root / "docs/BENCHMARK_STATUS.md", "<!-- CI_VALIDATION_START -->", "<!-- CI_VALIDATION_END -->", validation_body)
    replace_block(root / "docs/BENCHMARK_STATUS.md", "<!-- INVERSION_METRICS_START -->", "<!-- INVERSION_METRICS_END -->", metrics_body)
    replace_block(root / "docs/BENCHMARK_STATUS.md", "<!-- INVERSION_SWEEP_START -->", "<!-- INVERSION_SWEEP_END -->", sweep_body)
    replace_block(root / "docs/BENCHMARK_STATUS.md", "<!-- ARROW_METRICS_START -->", "<!-- ARROW_METRICS_END -->", arrow_body)
    replace_block(root / "docs/BENCHMARK_STATUS.md", "<!-- DDP_METRICS_START -->", "<!-- DDP_METRICS_END -->", ddp_body)
    print("Updated benchmark tables and validation summary using the current CI result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
