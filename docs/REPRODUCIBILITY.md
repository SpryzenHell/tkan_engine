# Reproducibility

## Recommended environment

Use Python 3.10, 3.11 or 3.12. Those versions are exercised by CI on Ubuntu.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Check the installation

```bash
python scripts/self_check.py
pytest
```

## Reproduce the repository figures

### Inversion

```bash
python scripts/inversion_benchmark.py \
  --events 3000 \
  --epochs 5 \
  --batch-size 128 \
  --hidden 16 \
  --grid-size 6
```

The output JSON is deterministic up to floating-point implementation details, and the CI run is stored in `results/ci_inversion_3000.json`.

### Arrow

```bash
python scripts/benchmark_arrow.py --events 5000000 --batch-size 250000
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
```

Throughput depends on the machine. The committed 300M result is specifically the GitHub Actions measurement documented in `results/arrow_benchmark_300m.json`.

### DDP

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/ddp_benchmark.py \
  --distributed \
  --events 10000 \
  --steps 10 \
  --batch-size 128
```

CPU throughput is expected to vary substantially by host.

## Clean-checkout test

The project does not require a pre-existing `results/` directory. Scripts create it when they write metrics. Synthetic examples do not require external data files.

## GPU note

The repository contains the GPU code path and profiling harness but no committed GPU benchmark. A valid GPU result requires a machine with CUDA-visible hardware and should be accompanied by device, driver/runtime, PyTorch version and exact command details.
