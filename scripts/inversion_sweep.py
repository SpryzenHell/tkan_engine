from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .inversion_benchmark import run_benchmark
except ImportError:  # direct execution as a script
    from inversion_benchmark import run_benchmark


def parse_csv(text: str, cast):
    try:
        values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated values: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return mean, std


def summarize(runs: list[dict], noise_levels: list[float]) -> list[dict]:
    summary = []
    for noise in noise_levels:
        rows = [
            row for row in runs
            if np.isclose(float(row["noise_std"]), float(noise), rtol=0.0, atol=1e-12)
        ]
        if not rows:
            continue

        row_summary = {"noise_std": float(noise), "n_seeds": len(rows)}
        for model_name in ("mlp", "tkan"):
            rmse_mean, rmse_std = _mean_std([
                float(row[model_name]["test_rmse_bps"]) for row in rows
            ])
            ic_mean, ic_std = _mean_std([
                float(row[model_name]["test_ic"]) for row in rows
            ])
            mse_mean, mse_std = _mean_std([
                float(row[model_name]["test_mse_bps2"]) for row in rows
            ])
            row_summary[model_name] = {
                "mean_rmse_bps": rmse_mean,
                "std_rmse_bps": rmse_std,
                "mean_ic": ic_mean,
                "std_ic": ic_std,
                "mean_mse_bps2": mse_mean,
                "std_mse_bps2": mse_std,
                "parameters": int(rows[0][model_name]["parameters"]),
            }

        gains = [
            float(row["mlp"]["test_rmse_bps"]) - float(row["tkan"]["test_rmse_bps"])
            for row in rows
        ]
        gain_mean, gain_std = _mean_std(gains)
        row_summary["paired_rmse_gain_mlp_minus_tkan"] = {
            "mean_bps": gain_mean,
            "std_bps": gain_std,
            "tkan_wins": sum(gain > 0.0 for gain in gains),
            "mlp_wins": sum(gain < 0.0 for gain in gains),
            "ties": sum(gain == 0.0 for gain in gains),
            "values_bps": gains,
        }
        summary.append(row_summary)
    return summary


def run_sweep(
    events: int,
    seq_len: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    grid_size: int,
    seeds: list[int],
    noise_levels: list[float],
) -> dict:
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if len(set(noise_levels)) != len(noise_levels):
        raise ValueError("noise levels must be unique")
    if any(value < 0 for value in noise_levels):
        raise ValueError("noise levels must be non-negative")

    runs = []
    for noise in noise_levels:
        for seed in seeds:
            result = run_benchmark(
                events=events,
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                hidden=hidden,
                grid_size=grid_size,
                seed=seed,
                noise_std=noise,
                out=None,
            )
            runs.append({
                "seed": int(seed),
                "noise_std": float(noise),
                "mlp": result["mlp"],
                "tkan": result["tkan"],
                "mlp_seconds": result["mlp_seconds"],
                "tkan_seconds": result["tkan_seconds"],
                "train_sequences": result["train_sequences"],
                "test_sequences": result["test_sequences"],
            })

    return {
        "experiment": "next-step T-KAN versus MLP sensitivity to seed and target noise",
        "events": events,
        "sequence_length": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden": hidden,
        "grid_size": grid_size,
        "seeds": seeds,
        "noise_levels": noise_levels,
        "target_units": "synthetic impact basis points",
        "input_features": "OFI, OFI change, and causal five-observation trailing mean",
        "target_definition": "0.9*sin(2.6*OFI[t]) + 0.55*OFI[t]^3 + 0.25*(OFI[t]-OFI[t-1]) + Gaussian noise; OFI is tanh of a stationary AR(1) series with phi=0.85",
        "runs": runs,
        "summary": summarize(runs, noise_levels),
        "notes": "Exploratory synthetic-data sensitivity analysis. Means and sample standard deviations describe the selected seeds; they are not confidence intervals, exchange-data results, or trading-performance claims.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repeat the nonlinear inversion benchmark across seeds and noise levels."
    )
    parser.add_argument("--events", type=int, default=3_000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--seeds", default="123,321,456,789,987")
    parser.add_argument("--noise-levels", default="0.04,0.08,0.16")
    parser.add_argument("--out", default="results/inversion_sweep.json")
    args = parser.parse_args(argv)

    seeds = parse_csv(args.seeds, int)
    noise_levels = parse_csv(args.noise_levels, float)
    result = run_sweep(
        events=args.events,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        grid_size=args.grid_size,
        seeds=seeds,
        noise_levels=noise_levels,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
