
Python 3.10+ is required.

~~~text
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
pytest
~~~

For an environment without internet, the optional Arrow/GPU dependencies can be installed separately when available.

## Run the nonlinear microstructure experiment

~~~text
python scripts/inversion_benchmark.py
~~~

The repository's verified CI run uses 3,000 controlled synthetic observations, a chronological 80/20 split, and 5 training epochs.

| Model | Test MSE (bps squared) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|
| MLP | 0.75995 | 0.87175 | 0.75018 |
| T-KAN | 0.28884 | 0.53744 | 0.86401 |

The target generator is a known nonlinear response:
0.9 sin(2.6 OFI) + 0.55 OFI cubed + 0.25 dOFI + Gaussian noise with sigma 0.08.

These numbers demonstrate representation learning on a controlled benchmark. They are not market-alpha claims.

Full output: results/inversion_benchmark.json

The same 3,000-event / 5-epoch configuration was executed by GitHub Actions; the saved branch-verified result is results/ci_inversion_3000.json.

## Run an LOB training smoke test

~~~text
python scripts/train.py --events 30000 --seq-len 64 --epochs 6
~~~

The data generator emits deterministic L2-like events with ten levels and derives queue imbalance, spread, microprice deviation, signed trade, depth imbalances and midpoint return.

The training target is the next midpoint return measured in basis points per event.

## Arrow 300M-event path

~~~text
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
~~~

The benchmark streams one Arrow RecordBatch at a time. It does not allocate a 300M-row DataFrame or a 300M-row Arrow Table.

The local validation environment did not have Apache Arrow installed, so no local 300M throughput number is claimed. The command is ready for the target environment.

## DDP

~~~text
torchrun --standalone --nproc_per_node=2 scripts/train.py --ddp --events 2000000 --epochs 10 --batch-size 1024
~~~

The training loop uses DistributedSampler, DistributedDataParallel, gradient clipping, mixed precision on CUDA and explicit synchronization of the KAN grid state after adaptive updates.

A two-rank CPU Gloo smoke test passed locally. This verifies distributed correctness; it is not a GPU scaling benchmark.

## GPU profiling

~~~text
python scripts/profile_gpu.py --batch 256 --seq-len 64 --features 15 --hidden 128 --steps 100
~~~

The script refuses to report a GPU profile when CUDA is unavailable. For arithmetic-intensity work, use Nsight Compute / Nsight Systems on the target NVIDIA machine and commit the resulting profiler summary.