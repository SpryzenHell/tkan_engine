# Data scale

## 300M-event path

The full stress command is:

```text
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
```

The synthetic generator creates one Arrow RecordBatch at a time. `process_batches` consumes the current batch before the next one is generated.

The measured GitHub Actions run processed:

| Metric | Value |
|---|---:|
| Events | 300,000,000 |
| RecordBatches | 300 |
| Batch size | 1,000,000 |
| Elapsed | 19.174389979 s |
| Throughput | 15,645,869.325 events/s |
| Selected rows | 300,000,000 |

The run used Ubuntu 24.04, Python 3.12.14 and PyArrow 25.0.1.

The important property is the memory boundary: total event count does not change the size of a single Arrow batch.

## Real replay data

For a real Parquet dataset, the minimum columns required by the raw-L2 path are:

```
ts_ns
bid_px_1
ask_px_1
bid_sz_1
ask_sz_1
```

The pipeline derives midpoint, queue imbalance and spread before filtering and projection.

For any real-data performance result, record the data source, date range, symbol universe, partition layout, machine configuration, software versions and exact command. Real exchange data is intentionally excluded from the repository.
