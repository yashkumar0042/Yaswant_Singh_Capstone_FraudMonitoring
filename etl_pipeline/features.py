from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


def discount_pct(gross: pd.Series, discount: pd.Series) -> pd.Series:
    gross = pd.to_numeric(gross, errors="coerce").fillna(0)
    discount = pd.to_numeric(discount, errors="coerce").fillna(0)
    denom = gross.replace(0, np.nan)
    pct = (discount / denom) * 100
    return pct.fillna(0).clip(lower=0)


def week_start(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.date


def zscore_within_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    def _z(v: pd.Series) -> pd.Series:
        v = pd.to_numeric(v, errors="coerce").fillna(0)
        if len(v) < 5:
            return pd.Series([0] * len(v), index=v.index)
        std = v.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series([0] * len(v), index=v.index)
        return (v - v.mean()) / std

    return df.groupby(group_col)[value_col].transform(_z).fillna(0)