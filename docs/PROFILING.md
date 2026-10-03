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
