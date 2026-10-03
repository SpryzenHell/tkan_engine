from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError as exc:
    pa = None
    pc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class ArrowRun:
    events: int
    batches: int
    elapsed_s: float
    events_per_s: float
    selected_rows: int
    checksum: float


def require_arrow() -> None:
    if pa is None:
        raise RuntimeError(
            "Apache Arrow is required. Install with "
            "pip install 'pyarrow>=25,<26'."
        ) from _IMPORT_ERROR


def dataframe_to_arrow(df):
    require_arrow()
    return pa.Table.from_pandas(df, preserve_index=False)


def q_where(table, expression):
    require_arrow()
    return table.filter(expression)


def q_select(table, columns: list[str]):
    require_arrow()
    return table.select(columns)


def q_by(table, keys: list[str], aggregations: list[tuple[str, str]]):
    require_arrow()
    return table.group_by(keys).aggregate(aggregations)


def q_within(table, column: str, lower: float, upper: float):
    require_arrow()
    col = table[column]
    mask = pc.and_(
        pc.greater_equal(col, lower),
        pc.less_equal(col, upper),
    )
    return table.filter(mask)


def process_batches(batches: Iterable):
    require_arrow()
    start = time.perf_counter()
    events = 0
    selected = 0
    batch_count = 0
    checksum = 0.0

    for batch in batches:
        table = pa.Table.from_batches([batch])
        spread = pc.divide(
            pc.multiply(
                pc.subtract(table["ask_px_1"], table["bid_px_1"]),
                1e4,
            ),
            table["mid"],
        )
        table = table.append_column("spread_bps", spread)
        table = q_where(
            table,
            pc.and_(
                pc.greater(table["queue_imbalance"], -0.9),
                pc.less(table["queue_imbalance"], 0.9),
            ),
        )
        table = q_select(
            table,
            ["ts_ns", "mid", "spread_bps", "queue_imbalance"],
        )

        batch_count += 1
        events += batch.num_rows
        selected += table.num_rows
        checksum += float(
            pc.sum(table["spread_bps"]).as_py() or 0.0
        )

    elapsed = time.perf_counter() - start
    return ArrowRun(
        events=events,
        batches=batch_count,
        elapsed_s=elapsed,
        events_per_s=events / elapsed if elapsed else 0.0,
        selected_rows=selected,
        checksum=checksum,
    )


def scan_parquet(path: str | Path, batch_size: int = 1_000_000, columns: list[str] | None = None):
    """Stream a Parquet dataset as Arrow RecordBatches."""
    require_arrow()
    import pyarrow.dataset as ds
    dataset = ds.dataset(str(path), format="parquet")
    return dataset.scanner(columns=columns, batch_size=batch_size).to_batches()


def synthetic_arrow_stream(
    events: int,
    batch_size: int = 1_000_000,
    seed: int = 7,
):
    require_arrow()
    rng = np.random.default_rng(seed)
    remaining = int(events)
    timestamp = 1_700_000_000_000_000_000
    mid_level = 100.0

    while remaining:
        n = min(batch_size, remaining)
        mid = mid_level + np.cumsum(
            rng.normal(0.0, 2e-5, n)
        )
        mid_level = float(mid[-1])
        spread = np.maximum(
            0.01,
            0.04 + 0.01 * rng.lognormal(0.0, 0.25, n),
        )
        bid = mid - spread / 2
        ask = mid + spread / 2
        bsz = rng.lognormal(2.2, 0.25, n)
        asz = rng.lognormal(2.2, 0.25, n)
        imbalance = (
            (bsz - asz) / np.maximum(bsz + asz, 1e-8)
        )

        yield pa.record_batch(
            {
                "ts_ns": pa.array(
                    np.arange(
                        timestamp,
                        timestamp + n * 1_000,
                        1_000,
                        dtype=np.int64,
                    )
                ),
                "mid": pa.array(mid),
                "bid_px_1": pa.array(bid),
                "ask_px_1": pa.array(ask),
                "queue_imbalance": pa.array(imbalance),
            }
        )
        timestamp += n * 1_000
        remaining -= n