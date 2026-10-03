# Validation summary

Local validation after the clean integration pass:

- 7 unit tests pass.
- 2-rank CPU Gloo DDP smoke test passes.
- Nonlinear inversion experiment passes and is recorded in inversion_benchmark.json.
- CPU model benchmark is recorded in model_benchmark.json.
- Apache Arrow is unavailable locally; no Arrow throughput number is claimed.
- CUDA is unavailable locally; no GPU arithmetic-intensity or GPU DDP number is claimed.

The exact local environment is recorded in ENVIRONMENT.txt.
