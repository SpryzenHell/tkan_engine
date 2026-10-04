from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkan_engine.arrow_pipeline import process_batches, scan_parquet, synthetic_arrow_stream

p=argparse.ArgumentParser()
p.add_argument("--events",type=int,default=1_000_000)
p.add_argument("--batch-size",type=int,default=250_000)
p.add_argument("--seed",type=int,default=7)
p.add_argument("--input",default=None,help="Parquet dataset/file; when set, scan this instead of synthetic data")
a=p.parse_args()

try:
    source=scan_parquet(a.input,batch_size=a.batch_size) if a.input else synthetic_arrow_stream(a.events,a.batch_size,a.seed)
    r=process_batches(source)
except RuntimeError as exc:
    payload={"status":"unavailable","reason":str(exc),"events_requested":a.events,"batch_size":a.batch_size,"input":a.input}
    Path("results").mkdir(exist_ok=True)
    Path("results/arrow_benchmark.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
    raise SystemExit(2)

payload=r.__dict__
Path("results").mkdir(exist_ok=True)
Path("results/arrow_benchmark.json").write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
