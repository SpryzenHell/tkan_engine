from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkan_engine.arrow_pipeline import process_batches, synthetic_arrow_stream


p = argparse.ArgumentParser()
p.add_argument("--events", type=int, default=1_000_000)
p.add_argument("--batch-size", type=int, default=250_000)
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()

try:
    result = process_batches(
        synthetic_arrow_stream(
            args.events,
            args.batch_size,
            args.seed,
        )
    )
except RuntimeError as exc:
    payload = {
        "status": "unavailable",
        "reason": str(exc),
        "events_requested": args.events,
        "batch_size": args.batch_size,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/arrow_benchmark.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(2)

payload = result.__dict__
Path("results").mkdir(exist_ok=True)
Path("results/arrow_benchmark.json").write_text(
    json.dumps(payload, indent=2)
)
print(json.dumps(payload, indent=2))
