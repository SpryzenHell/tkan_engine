# Resume claim validation

This document ties each project claim to code paths, reproducible commands, measured evidence, and remaining limits. A result is considered verified only for the workload and hardware actually measured.

## 1. Temporal KAN nonlinear inversion

**Implementation:** `src/tkan_engine/kan.py` contains `KANLinear.b_splines`, learned spline coefficients, adaptive grid updates, and the causal temporal convolution block. `src/tkan_engine/model.py` exposes the `MicrostructureForecaster`.

**Reproducible experiment:**

```bash
python scripts/inversion_benchmark.py --events 3000 --epochs 5 --batch-size 128 --hidden 16 --grid-size 6
python scripts/inversion_sweep.py --events 3000 --epochs 5 --batch-size 128 --hidden 16 --grid-size 6 --seeds 123,321,456,789,987 --noise-levels 0.04,0.08,0.16
pytest
```

The experiment creates a nonlinear target from synthetic order-flow imbalance. The 3,000-row benchmark uses an 80/20 chronological split and trailing, past-only input features.

| Metric | MLP | T-KAN |
|---|---:|---:|
| Parameters | 801 | 3,266 |
| Test MSE | 0.340292 | 0.324264 |
| Test RMSE | 0.583346 | 0.569442 |
| Test information coefficient | 0.781706 | 0.793471 |

On this run the T-KAN test RMSE was 2.4% lower. In the separate five-seed by three-noise sweep, T-KAN had lower RMSE in 12 of 15 paired comparisons. This is evidence for the synthetic task, not proof of generalization to real markets or statistical significance.

The test suite checks B-spline basis shape/partition, forward and backward gradients, adaptive-grid changes, finite outputs/gradients after updates, and a regression that appending future observations does not alter an earlier timestep's prediction.

## 2. Apache Arrow event pipeline and q-like operations

**Implementation:** `src/tkan_engine/arrow_pipeline.py` processes one `RecordBatch` at a time, derives best-level midpoint/spread and queue imbalance as needed, filters with Arrow compute expressions, selects columns, and calculates a checksum. Helpers cover q-like filtering, selection, inclusive range filtering, and grouping; this is a subset of q-like operations, not a KDB+/q interpreter.

**Reproducible commands:**

```bash
python scripts/benchmark_arrow.py --events 5000000 --batch-size 250000
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
```

Recorded CI stress result:

| Metric | Result |
|---|---:|
| Generated events | 300,000,000 |
| Batch size | 1,000,000 |
| Batches processed | 300 |
| Elapsed time | 26.805 s |
| Throughput | 11.192 million events/s |
| Selected rows | 300,000,000 |

The run generates synthetic best-level book snapshots and streams them batch by batch; it does not replay 300 million real exchange messages. The two 5M/300M runs were in separate CI jobs, so their throughput is host-dependent and they are not a controlled scaling comparison. The Arrow helper tests separately exercise filter, select, range and group operations.

For a real-data result, run the benchmark against a permitted Parquet snapshot dataset with the required timestamp, best bid/ask prices and size columns, then record the dataset identifier, symbols, time range, schema, licence, machine and software versions. No real-market replay result is currently claimed.

## 3. DDP training throughput and arithmetic intensity

**Implementation:** `src/tkan_engine/training.py` supports PyTorch DistributedDataParallel, distributed sampling, CUDA automatic mixed precision, optional fused AdamW, gradient clipping, warmup-independent training, and synchronization of adaptive spline-grid state across ranks. `scripts/ddp_benchmark.py` measures throughput after warmup and synchronizes CUDA before timing.

**Existing measured smoke result:** two CPU ranks processed 2,560 global samples in 32.508 seconds (78.749 samples/s), using 10 warmup and 10 measured steps. This validates a distributed CPU path only. It is not GPU throughput evidence.

To measure GPU throughput and scaling, use the self-hosted workflow in the `SpryzenHell/integrator` repository: `.github/workflows/tkan-engine-a100-validation.yml`. Set its `tkan_ref` input to the branch or commit being validated. It checks the installed CUDA/PyTorch environment, runs tests and the Arrow experiments, and benchmarks one, two and four visible A100 devices. The registered runner labels include `integrator-1`; this workflow belongs in the Integrator repository where that runner is registered.

A GPU run must save device/model, driver/runtime/PyTorch versions, warmup and measured steps, global/local batch sizes, samples/s, step latency and peak memory. To support the word **maximized**, compare against a baseline under a stated objective and controlled settings; one throughput number alone does not prove maximization.

Arithmetic intensity is a separate hardware measurement. It is FLOPs divided by bytes moved to device memory for the profiled workload. Use Nsight Compute on the A100 host to collect supported FP16/FP32/FP64 FLOP counters and DRAM-byte counters, record the Nsight Compute version and full command, and calculate the ratio only when the requested counters are available. The PyTorch profiler table and allocated-memory metric do not by themselves establish hardware-counter arithmetic intensity. See the [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).

**Status:** GPU scaling, arithmetic-intensity counters, and a measured baseline comparison remain outstanding until the workflow produces a successful GPU artifact. Do not describe the CPU result as A100 or multi-GPU performance.

## Current conclusion

- **Temporal KAN / nonlinear inversion:** implementation and synthetic experiments verified; real-market generalization not established.
- **Arrow / 300M events:** synthetic streaming scale verified; 300M real exchange events and full KDB+/q equivalence not established.
- **DDP / GPU arithmetic intensity:** DDP code and CPU smoke path verified; A100 throughput, scaling improvement, and hardware-counter arithmetic intensity remain unverified pending GPU evidence.
