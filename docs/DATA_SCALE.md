# Data scale

## 300M-event design

The command is:

python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000

synthetic_arrow_stream generates one RecordBatch at a time. process_batches transforms and consumes each batch before the next batch is generated.

Therefore:
- rows are bounded by batch size rather than total event count,
- Arrow compute operates on columns,
- a 300M-event stress run does not require a 300M-row in-memory object.

The local runtime used for this branch did not have pyarrow, so it did not produce a 300M throughput result.

For a real replay:
1. store Parquet or Arrow IPC files by date/symbol,
2. use pyarrow.dataset to scan partitions,
3. keep the same RecordBatch interface,
4. record dataset, date range, symbols, CPU, RAM, pyarrow version and command,
5. commit the resulting benchmark JSON.

The 300M number should only be written on the resume after that target-environment run has actually completed.
