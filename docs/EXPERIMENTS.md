# Experiments and measurements

## Controlled nonlinear inversion

The target is generated from the order-flow imbalance feature and its change:

```text
0.9*sin(2.6*OFI) + 0.55*OFI^3 + 0.25*dOFI + N(0, 0.08^2)
```

The latest recorded experiment used 3,000 observations, 16-step sequences, a chronological 80/20 split, five epochs, and a batch size of 128. It was measured in [GitHub Actions run 38927224698](https://github.com/SpryzenHell/tkan_engine/actions/runs/38927224698); all seven jobs in that run passed.

<!-- INVERSION_METRICS_START -->
| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | — | — | — | — |
| T-KAN | — | — | — | — |
<!-- INVERSION_METRICS_END -->

Source: results/ci_inversion_3000.json.

## Seed and noise sensitivity

The companion sensitivity sweep repeats the model comparison across several seeds and target-noise levels. The generated report includes mean and standard deviation for test RMSE and information coefficient, paired per-seed RMSE differences, training times, and all individual run metrics.

<!-- INVERSION_SWEEP_START -->
The sweep table and four plots will be filled from results/inversion_sweep.json by the CI report step.
<!-- INVERSION_SWEEP_END -->

## Arrow stream sizes

The 5M-event and 300M-event jobs both run a deterministic synthetic Arrow stream. They use different batch sizes, and the measurements were taken in separate CI jobs.

| Events | Batch size | Batches | Elapsed (s) | Events/s | Selected rows | Checksum |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 250,000 | 20 | 0.512705 | 9,752,190.8 | 5,000,000 | 25,165,450.6906 |
| 300,000,000 | 1,000,000 | 300 | 26.934001 | 11,138,337.6 | 300,000,000 | 1,509,462,417.5704 |

The process builds and releases one batch at a time. These counts describe generated rows, not a replay of real market messages. They demonstrate the Arrow path's ability to work across a stream larger than a single in-memory batch.

Sources: `results/arrow_benchmark_5m.json` and `results/arrow_benchmark_300m.json`.

## DDP

The saved smoke benchmark used two Gloo ranks on CPU, with 2,560 total samples over ten measured steps:

| Device | Ranks | Wall time (s) | Global samples/s |
|---|---:|---:|---:|
| CPU | 2 | 21.004563 | 121.8783 |

This is a distributed correctness smoke test. Do not use it as a GPU throughput or scaling claim.

Source: `results/ddp_benchmark_cpu.json`.

## A100 experiments

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured in this checkout because the integrator-1 runner was offline during the 16 October 2026 validation. The manual GPU workflow records one-, two-, and four-GPU throughput where the devices are available, along with mixed-precision settings, peak allocated memory, and a PyTorch operator profile. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Data and model choices

All current committed benchmark datasets are generated deterministically. No pretrained weights are needed. No real exchange data is bundled; supply a Parquet file only when its license and access terms allow the test. For a real-data benchmark, save the dataset identifier, date range, symbols, schema, hardware details, library versions and command beside the run result.

## Reproducible commands

```shell
python scripts/inversion_benchmark.py --events 3000 --epochs 5 --batch-size 128 --hidden 16 --grid-size 6
python scripts/benchmark_arrow.py --events 5000000 --batch-size 250000
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
torchrun --standalone --nproc_per_node=2 scripts/ddp_benchmark.py --distributed --events 10000 --steps 10 --batch-size 128
pytest
```
