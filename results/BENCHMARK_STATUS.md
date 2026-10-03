# Verified benchmark record

| Benchmark | Result |
|---|---|
| Nonlinear inversion | 3,000 events, 5 epochs; T-KAN RMSE 0.53744 bps, IC 0.86401 |
| Arrow smoke | 5,000,000 events; 10.39M events/s |
| Arrow full-scale | 300,000,000 events configured in CI; record result after completion |
| DDP CPU | 2-rank correctness/throughput smoke configured in CI |
| GPU DDP | Requires CUDA-capable target hardware |
