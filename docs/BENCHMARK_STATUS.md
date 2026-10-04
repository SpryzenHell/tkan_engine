# Benchmark status

Last verified branch head:

`4b60f719832b38568bb796bc3f674700df49b37b`

GitHub Actions run: **37154074480**

## Test matrix

Python 3.10, 3.11 and 3.12 all completed successfully. Each test job reported 7 passed and 2 skipped.

## Nonlinear inversion

The CI benchmark used 3,000 events, sequence length 16, 5 epochs, batch size 128, hidden dimension 16 and grid size 6.

| Model | Test RMSE (bps) | Test IC |
|---|---:|---:|
| MLP | 0.8717508959 | 0.7501842616 |
| T-KAN | 0.5374401946 | 0.8640131220 |

## Arrow

The full stress job completed successfully:

| Metric | Value |
|---|---:|
| Events | 300,000,000 |
| Batches | 300 |
| Batch size | 1,000,000 |
| Elapsed | 19.174389979 s |
| Events/s | 15,645,869.3251 |
| Selected rows | 300,000,000 |
| Checksum | 1,509,462,417.5703695 |

The 5M smoke job in the same run processed 5,000,000 events in 0.475624442 s at 10,512,495.9074 events/s.

## DDP CPU smoke

The two-rank distributed job completed successfully:

| Metric | Value |
|---|---:|
| World size | 2 |
| Device | CPU |
| Global samples | 2,560 |
| Throughput | 104.8995173 samples/s |
| Wall time | 24.404306757 s |

This is a CPU smoke benchmark. It must not be presented as GPU scaling.

## GPU

No GPU throughput or arithmetic-intensity number is committed. The repository contains the profiler and the DDP benchmark harness for use on an NVIDIA system.
