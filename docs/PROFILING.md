# Profiling

## CPU

scripts/benchmark_model.py reports inference throughput and batch latency with synchronized timing on CUDA and a high-resolution wall clock on CPU.

## GPU arithmetic intensity

scripts/profile_gpu.py uses torch.profiler and records CUDA time and memory. For actual arithmetic intensity, run Nsight Compute on the target NVIDIA host and collect achieved FLOP/s, DRAM bandwidth, occupancy and Tensor Core utilization.

The script refuses to report a GPU metric when CUDA is not available.

## DDP scaling

Run the same model/data configuration at ranks 1, 2, 4 and 8.

Record:
- global samples per second,
- step time,
- all-reduce contribution,
- peak memory per rank,
- scaling efficiency.

The CPU two-rank script is a correctness smoke test only.

## Recorded Arrow benchmark

GitHub Actions run 37152777795 processed 5,000,000 synthetic events in 20 Arrow RecordBatches of 250,000 rows each.

- elapsed: 0.481237 s
- throughput: 10,389,882 events/s
- selected rows: 5,000,000

This is a streaming CPU Arrow benchmark on a GitHub-hosted runner, not a 300M result.

## DDP scaling benchmark schema

scripts/ddp_benchmark.py records:
- world size
- device
- global batch size
- global samples
- wall-clock seconds
- samples/sec
- step latency
- parameter count

For GPU scaling, run the same configuration at 1, 2, 4 and 8 ranks. Compute:

scaling_efficiency(N) = throughput_N / (N * throughput_1)

and retain the hardware, CUDA, PyTorch, driver and batch configuration in the output JSON.
