# T-KAN Microstructure Engine

A focused Temporal Kolmogorov-Arnold Network research engine for high-frequency limit-order-book data.

The project is organized around the three engineering claims that matter for the resume:

1. Dynamic B-spline edge activations inside a temporal model.
2. Batch-streamed Arrow processing with q-like select / where / within / by semantics.
3. PyTorch DistributedDataParallel training with synchronized adaptive spline grids.

## Repository status

The original composite repository is retained on main for provenance, but its copied T-KAN sources contain mechanically prefixed identifiers such as tkanImport and torch.tkanLinspace. They are not the supported runtime surface.

This branch introduces a clean package under src/tkan_engine. The clean KAN layer is based on the implementation pattern in Blealtan/efficient-kan; the time-series design was informed by the KAN integration in sktime/pytorch-forecasting; and QuantResearch was used as a quantitative research/backtesting reference. See THIRD_PARTY_NOTICES.md.

No historical Git timestamps are rewritten.

## Install

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

The current local run uses 3,000 controlled synthetic observations with an 80/20 chronological split.

| Model | Test MSE (bps squared) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|
| MLP | 0.75995 | 0.87175 | 0.75018 |
| T-KAN | 0.44005 | 0.66336 | 0.79187 |

The target generator is a known nonlinear response:
0.9 sin(2.6 OFI) + 0.55 OFI cubed + 0.25 dOFI + Gaussian noise with sigma 0.08.

These numbers demonstrate representation learning on a controlled benchmark. They are not market-alpha claims.

Full output: results/inversion_benchmark.json

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

## Architecture

~~~text
LOB feed / Parquet / IPC
          |
          v
Arrow RecordBatch stream
          |
          +---- q-like select / where / within / by
          |
          v
Pandas-compatible microstructure features
          |
          v
rolling windows: [batch, time, feature]
          |
          v
input projection
          |
          +---- causal depthwise temporal convolution
          |
          +---- efficient KAN B-spline edge functions
          |
          v
residual temporal blocks
          |
          v
next-event return
          |
          +---- MSE / RMSE / IC

torchrun -> DDP -> DistributedSampler -> synchronized grid state
~~~

## Evidence and limitations

The repository distinguishes executable capability from measured performance.

Measured locally:
- 7 unit tests passing.
- 2-rank CPU Gloo DDP smoke test passing.
- T-KAN nonlinear inversion benchmark above.
- CPU model throughput benchmark in results/model_benchmark.json.

Not measured locally:
- Apache Arrow 300M-event throughput because pyarrow was unavailable.
- GPU arithmetic intensity, Tensor Core utilization or GPU DDP scaling because CUDA was unavailable.

The exact local environment is recorded in results/ENVIRONMENT.txt and the validation summary in results/VALIDATION_SUMMARY.md.

## Upstream references

- https://github.com/Blealtan/efficient-kan
- https://github.com/letianzj/QuantResearch
- https://github.com/sktime/pytorch-forecasting

The exact inspected commit SHAs and roles are documented in THIRD_PARTY_NOTICES.md.

## Resume mapping

Resume claim: dynamic B-spline edge activations
Implementation: src/tkan_engine/kan.py
Evidence: results/inversion_benchmark.json and tests/test_core.py

Resume claim: 300M+ LOB events through Apache Arrow q-like processing
Implementation: src/tkan_engine/arrow_pipeline.py and scripts/benchmark_arrow.py
Evidence requirement: execute the 300M event command on the target environment and commit the generated JSON.

Resume claim: GPU arithmetic intensity and DDP throughput
Implementation: src/tkan_engine/training.py, scripts/ddp_smoke.py and scripts/profile_gpu.py
Evidence requirement: target NVIDIA hardware with Nsight and a multi-rank benchmark.

## Date note

The supplied project mapping says 2025-07-01 through 2025-10-31, while the resume text says Jan 2026 through Mar 2026. Resolve that discrepancy before publishing the date range.
