from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "svg.fonttype": "none",
    "svg.hashsalt": "tkan-engine",
})


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _parameter_count(result: dict, model_name: str) -> int:
    """Read parameter counts from both the saved report and raw run schemas."""
    model = result.get(model_name, {})
    value = model.get("parameters") if isinstance(model, dict) else None
    if value is None:
        value = result.get(f"{model_name}_parameters")
    if value is None:
        raise ValueError(
            f"Missing parameter count for {model_name!r}; expected "
            f"{model_name}.parameters or {model_name}_parameters"
        )
    return int(value)


def save_figure(fig, name: str, svg_dir: Path, png_dir: Path) -> None:
    # Fixed SVG metadata makes chart regeneration stable and prevents repeated
    # CI commits caused only by a changing timestamp in the file header.
    fig.savefig(
        svg_dir / f"{name}.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "Pirate-Emperor", "Date": None},
    )
    fig.savefig(
        png_dir / f"{name}.png",
        format="png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Pirate-Emperor"},
    )
    plt.close(fig)


def _annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create plots from the checked-in benchmark JSON results."
    )
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--png-out", default="results/figures")
    parser.add_argument("--inversion", default="results/ci_inversion_3000.json")
    parser.add_argument("--arrow-5m", default="results/arrow_benchmark_5m.json")
    parser.add_argument("--arrow-300m", default="results/arrow_benchmark_300m.json")
    parser.add_argument("--ddp", default="results/ddp_benchmark_cpu.json")
    parser.add_argument("--sweep", default="results/inversion_sweep.json")
    args = parser.parse_args()

    svg_dir = Path(args.out)
    png_dir = Path(args.png_out)
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    inversion = load(Path(args.inversion))
    mlp = inversion["mlp"]
    tkan = inversion["tkan"]
    models = ["MLP", "T-KAN"]

    # 1. Test RMSE: lower is better.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    values = [mlp["test_rmse_bps"], tkan["test_rmse_bps"]]
    bars = ax.bar(models, values)
    _annotate_bars(ax, bars, "{:.4f}")
    ax.set_ylabel("Test RMSE (bps)")
    ax.set_title("Controlled nonlinear inversion: test error")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, "inversion_rmse", svg_dir, png_dir)

    # 2. Information coefficient: higher is better.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    values = [mlp["test_ic"], tkan["test_ic"]]
    bars = ax.bar(models, values)
    _annotate_bars(ax, bars, "{:.4f}")
    ax.set_ylabel("Test information coefficient")
    ax.set_title("Controlled nonlinear inversion: rank correlation")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, "inversion_ic", svg_dir, png_dir)

    # 3. Measured T-KAN training loss over the five recorded epochs.
    history = inversion.get("tkan_training_history", [])
    if not history:
        raise ValueError("The inversion result does not contain training history")
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    ax.plot(epochs, losses, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("T-KAN training loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    save_figure(fig, "inversion_loss", svg_dir, png_dir)

    arrow_5m = load(Path(args.arrow_5m))
    arrow_300m = load(Path(args.arrow_300m))
    run_labels = ["5M events", "300M events"]
    throughputs_m = [
        arrow_5m["events_per_s"] / 1_000_000,
        arrow_300m["events_per_s"] / 1_000_000,
    ]

    # 4. Throughput observed in the two CI stress configurations.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    bars = ax.bar(run_labels, throughputs_m)
    _annotate_bars(ax, bars, "{:.2f}M")
    ax.set_ylabel("Events processed per second (millions)")
    ax.set_title("Arrow throughput at two event counts")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, "arrow_300m", svg_dir, png_dir)

    # 5. Runtime by event count; log x-axis keeps both sizes readable.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    event_counts_m = [
        arrow_5m["events"] / 1_000_000,
        arrow_300m["events"] / 1_000_000,
    ]
    elapsed = [arrow_5m["elapsed_s"], arrow_300m["elapsed_s"]]
    ax.plot(event_counts_m, elapsed, marker="o", linewidth=2)
    for x, y in zip(event_counts_m, elapsed):
        ax.annotate(
            f"{y:.3f} s",
            (x, y),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Events processed (millions, log scale)")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title("Observed Arrow runtime")
    ax.grid(alpha=0.25, which="both")
    save_figure(fig, "arrow_runtime", svg_dir, png_dir)

    # 6. Number of batches and configured rows per batch.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    batch_counts = [arrow_5m["batches"], arrow_300m["batches"]]
    batch_sizes = [arrow_5m["batch_size"], arrow_300m["batch_size"]]
    bars = ax.bar(run_labels, batch_counts)
    for bar, batch_size in zip(bars, batch_sizes):
        ax.annotate(
            f'{int(bar.get_height()):,} batches\n{batch_size:,} rows per batch',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("RecordBatch count")
    ax.set_title("Arrow streaming batch configuration")
    ax.set_ylim(0, max(batch_counts) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, "arrow_batches", svg_dir, png_dir)

    # 7. Parameter/error trade-off for the two synthetic inversion models.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    params = [
        _parameter_count(inversion, "mlp"),
        _parameter_count(inversion, "tkan"),
    ]
    rmse_values = [mlp["test_rmse_bps"], tkan["test_rmse_bps"]]
    ax.scatter(params, rmse_values, s=95)
    for label, x, y in zip(models, params, rmse_values):
        ax.annotate(
            f"{label}\n{x:,} parameters",
            (x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Trainable parameter count")
    ax.set_ylabel("Test RMSE (bps, lower is better)")
    ax.set_title("Model size versus test error")
    ax.grid(alpha=0.25)
    save_figure(fig, "model_tradeoff", svg_dir, png_dir)

    # 8. Keep the CPU distributed result separate from GPU benchmarks.
    ddp = load(Path(args.ddp))
    fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    throughput = float(ddp["samples_per_s"])
    ax.barh(["2-rank CPU DDP"], [throughput])
    ax.annotate(
        f"{throughput:.3f} samples/s",
        (throughput, 0),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )
    ax.set_xlabel("Global samples per second")
    ax.set_title("DDP correctness smoke test (CPU only)")
    ax.text(
        0.02, 0.95,
        f'{ddp["global_samples"]:,} samples in {ddp["wall_seconds"]:.3f} s',
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set_xlim(0, throughput * 1.35)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(fig, "ddp_cpu", svg_dir, png_dir)

    # Optional GPU scaling plot appears only when real CUDA benchmark JSON exists.
    results_dir = Path(args.ddp).parent
    gpu_runs = []
    for gpu_count in (1, 2, 4):
        path = results_dir / f"ddp_benchmark_{gpu_count}gpu.json"
        if path.exists():
            run = load(path)
            if str(run.get("device", "")).startswith("cuda"):
                gpu_runs.append(run)
    if gpu_runs:
        gpu_runs.sort(key=lambda item: int(item["world_size"]))
        labels = [f'{int(run["world_size"])} GPU' for run in gpu_runs]
        values = [float(run["samples_per_s"]) for run in gpu_runs]
        baseline = next(
            (float(run["samples_per_s"]) for run in gpu_runs if int(run["world_size"]) == 1),
            None,
        )
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        bars = ax.bar(labels, values)
        for bar, run in zip(bars, gpu_runs):
            speed = float(run["samples_per_s"])
            speedup = speed / baseline if baseline and baseline > 0 else None
            label = f"{speed:,.1f} samples/s"
            if speedup is not None:
                label += f"\n{speedup:.2f}x"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("Global samples per second")
        ax.set_title("Measured CUDA DDP training throughput")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        save_figure(fig, "gpu_scaling", svg_dir, png_dir)

    # Optional robustness analysis: multiple training seeds and target-noise levels.
    sweep_path = Path(args.sweep)
    if sweep_path.exists():
        sweep = load(sweep_path)
        runs = sweep.get("runs", [])
        summaries = sweep.get("summary", [])
        if not runs or not summaries:
            raise ValueError(f"The inversion sweep has no runs or summary rows: {sweep_path}")

        noise_levels = [float(row["noise_std"]) for row in summaries]
        labels = [f"{level:.2f}" for level in noise_levels]
        mlp_rmse = [float(row["mlp"]["mean_rmse_bps"]) for row in summaries]
        tkan_rmse = [float(row["tkan"]["mean_rmse_bps"]) for row in summaries]
        mlp_std = [float(row["mlp"]["std_rmse_bps"]) for row in summaries]
        tkan_std = [float(row["tkan"]["std_rmse_bps"]) for row in summaries]

        # 9. Prediction error as observation noise increases.
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        ax.errorbar(noise_levels, mlp_rmse, yerr=mlp_std, marker="o", capsize=4, label="MLP")
        ax.errorbar(noise_levels, tkan_rmse, yerr=tkan_std, marker="s", capsize=4, label="T-KAN")
        ax.set_xlabel("Target noise standard deviation")
        ax.set_ylabel("Mean test RMSE (bps; error bars show run-to-run SD)")
        ax.set_title("Noise sensitivity of next-step prediction")
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, "inversion_noise_rmse", svg_dir, png_dir)

        # 10. Rank correlation under the same noise sweep.
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        ax.plot(noise_levels, [float(row["mlp"]["mean_ic"]) for row in summaries],
                marker="o", linewidth=2, label="MLP")
        ax.plot(noise_levels, [float(row["tkan"]["mean_ic"]) for row in summaries],
                marker="s", linewidth=2, label="T-KAN")
        ax.set_xlabel("Target noise standard deviation")
        ax.set_ylabel("Mean test information coefficient")
        ax.set_title("Rank-correlation sensitivity to target noise")
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, "inversion_noise_ic", svg_dir, png_dir)

        # 11. Per-seed errors at the noise level nearest to 0.08.
        reference_noise = min(noise_levels, key=lambda value: abs(value - 0.08))
        seed_runs = sorted(
            (row for row in runs if abs(float(row["noise_std"]) - reference_noise) < 1e-12),
            key=lambda row: int(row["seed"]),
        )
        if seed_runs:
            fig, ax = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
            x = list(range(len(seed_runs)))
            width = 0.36
            ax.bar([value - width / 2 for value in x],
                   [float(row["mlp"]["test_rmse_bps"]) for row in seed_runs],
                   width=width, label="MLP")
            ax.bar([value + width / 2 for value in x],
                   [float(row["tkan"]["test_rmse_bps"]) for row in seed_runs],
                   width=width, label="T-KAN")
            ax.set_xticks(x, [str(int(row["seed"])) for row in seed_runs])
            ax.set_xlabel("Random seed")
            ax.set_ylabel("Test RMSE (bps; lower is better)")
            ax.set_title(f"Seed-to-seed variation at noise SD {reference_noise:.2f}")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend()
            save_figure(fig, "inversion_seed_rmse", svg_dir, png_dir)

        # 12. Paired error differences for each noise level; positive favours T-KAN.
        gain_by_noise = []
        for noise in noise_levels:
            group = [row for row in runs if abs(float(row["noise_std"]) - noise) < 1e-12]
            gain_by_noise.append([
                float(row["mlp"]["test_rmse_bps"]) - float(row["tkan"]["test_rmse_bps"])
                for row in group
            ])
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        ax.boxplot(gain_by_noise, tick_labels=labels)
        ax.axhline(0.0, linewidth=1, linestyle="--")
        ax.set_xlabel("Target noise standard deviation")
        ax.set_ylabel("MLP RMSE − T-KAN RMSE (bps; positive favours T-KAN)")
        ax.set_title("Paired prediction-error differences across seeds")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        save_figure(fig, "inversion_rmse_gain", svg_dir, png_dir)

    expected = {
        "inversion_rmse.svg",
        "inversion_ic.svg",
        "inversion_loss.svg",
        "arrow_300m.svg",
        "arrow_runtime.svg",
        "arrow_batches.svg",
        "model_tradeoff.svg",
        "ddp_cpu.svg",
    }
    if sweep_path.exists():
        expected.update({
            "inversion_noise_rmse.svg",
            "inversion_noise_ic.svg",
            "inversion_seed_rmse.svg",
            "inversion_rmse_gain.svg",
        })
    missing = expected - {path.name for path in svg_dir.glob("*.svg")}
    if missing:
        raise RuntimeError(f"Plot generation failed; missing: {sorted(missing)}")

    baseline_count = 8
    sweep_count = 4 if sweep_path.exists() else 0
    print(
        f"Rendered {baseline_count} baseline and {sweep_count} sweep SVG plots "
        f"to {svg_dir}; PNG copies saved to {png_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
