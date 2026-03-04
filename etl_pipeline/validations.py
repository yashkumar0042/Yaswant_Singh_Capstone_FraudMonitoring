from __future__ import annotations
import pandas as pd


def validate_fact(fact: pd.DataFrame) -> None:
    if "order_id" not in fact.columns:
        raise ValueError("fact_orders_enriched must have order_id")
    dup = fact["order_id"].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Join explosion: {dup} duplicate order_id rows in fact_orders_enriched")
    if (fact["risk_score"] < 0).any() or (fact["risk_score"] > 100).any():
        raise ValueError("risk_score out of bounds 0..100")