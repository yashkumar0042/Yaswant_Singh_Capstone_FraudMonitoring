from __future__ import annotations

from typing import List
import pandas as pd


def to_snake_case(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip().replace(" ", "_").replace("-", "_")
        out.append(c2.lower())
    return out


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = to_snake_case(df.columns)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").str.strip()
    return df


def safe_lower(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def parse_dt(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def dedup_keep_last(df: pd.DataFrame, key_cols: List[str], order_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if order_col and order_col in df.columns:
        df = df.sort_values(order_col)
    return df.drop_duplicates(subset=key_cols, keep="last")