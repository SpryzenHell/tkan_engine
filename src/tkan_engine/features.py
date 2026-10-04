from __future__ import annotations

import numpy as np
import pandas as pd


class LOBFeatureExtractor:
    """Vectorized LOB features used by the pandas training path."""

    @staticmethod
    def transform(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        bid1 = df["bid_px_1"].to_numpy()
        ask1 = df["ask_px_1"].to_numpy()
        bsz = df["bid_sz_1"].to_numpy()
        asz = df["ask_sz_1"].to_numpy()

        mid = (bid1 + ask1) / 2.0
        denom = np.maximum(bsz + asz, 1e-8)

        out["mid"] = mid
        out["spread_bps"] = (ask1 - bid1) / mid * 1e4
        out["queue_imbalance"] = (bsz - asz) / denom
        out["microprice_dev_bps"] = (
            ((ask1 * bsz + bid1 * asz) / denom - mid) / mid * 1e4
        )
        out["signed_trade"] = (
            df["trade_side"].to_numpy(dtype=np.float32)
            * df["trade_sz"].to_numpy(dtype=np.float32)
        )

        for level in range(1, levels + 1):
            b = df[f"bid_sz_{level}"].to_numpy()
            a = df[f"ask_sz_{level}"].to_numpy()
            out[f"imbalance_{level}"] = (
                (b - a) / np.maximum(b + a, 1e-8)
            )

        out["mid_return"] = (
            pd.Series(mid, index=df.index)
            .pct_change()
            .fillna(0.0)
            .to_numpy()
        )
        return out.astype(np.float32)
