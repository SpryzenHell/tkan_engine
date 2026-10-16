# Benchmark and test status

## Latest complete CI validation

<!-- CI_VALIDATION_START -->
The current test matrix and full CI run link will be written here after all jobs succeed.
<!-- CI_VALIDATION_END -->

## Controlled nonlinear inversion

The benchmark fits a known nonlinear target from synthetic order-flow features. The test set is chronological; this is not exchange replay or a trading backtest.

<!-- INVERSION_METRICS_START -->
| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | — | — | — | — |
| T-KAN | — | — | — | — |
<!-- INVERSION_METRICS_END -->

The result uses synthetic data and is not exchange replay or a trading-performance claim. See results/ci_inversion_3000.json.

### Seed and noise sensitivity

<!-- INVERSION_SWEEP_START -->
The repeated seed/noise measurements and four additional plots will be recorded here from CI results.
<!-- INVERSION_SWEEP_END -->

## Arrow stream benchmark

Both runs process deterministic synthetic batches and release each batch after processing. They demonstrate streaming at scale, not replay of real exchange messages.

| Events | Batches | Batch size | Elapsed (s) | Events/s | Selected rows |
|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 20 | 250,000 | 0.512705 | 9,752,190.8 | 5,000,000 |
| 300,000,000 | 300 | 1,000,000 | 26.934001 | 11,138,337.6 | 300,000,000 |

The runs were in separate CI jobs; their rates are host-dependent and should not be read as a controlled scaling comparison. Checksums and environment details are saved in results/arrow_benchmark_5m.json and results/arrow_benchmark_300m.json.

## DDP CPU smoke test

| World size | Device | Global samples | Wall time (s) | Throughput (samples/s) |
|---:|---|---:|---:|---:|
| 2 | CPU | 2,560 | 21.004563 | 121.8783 |

This checks distributed execution on CPU. It is not a GPU throughput or scaling result. Full details are saved in results/ddp_benchmark_cpu.json.

## GPU DDP and profiler

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured yet. The integrator-1 runner was offline during the 16 October 2026 validation. Run the manual tkan-engine-a100-validation workflow from the repository where that runner is registered. The workflow checks out this project, validates the visible A100 40GB devices, and uploads measured GPU results, plots, profiler output, and updated benchmark summaries as an artifact. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Current test coverage

The suite checks spline basis shapes and sums, forward/backward gradients, adaptive spline grids, temporal output shapes, finite LOB features, stream continuity and reproducibility, sequence alignment, causal rolling features, invalid parameters, report schema compatibility, seed-sweep statistics, Arrow batch accounting, integer queue sizes, stale spread values, missing columns, empty streams, range filtering, and boundary handling.

Continuous integration runs this suite on Python 3.10, 3.11 and 3.12, repeats the controlled inversion benchmark, runs a multi-seed/noise sensitivity sweep, processes 5M and 300M synthetic events, runs two-rank CPU DDP, generates twelve SVG plots, and uploads PNG copies. The A100 test is a separate manual workflow because its self-hosted runner was offline during this validation.
