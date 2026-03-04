from __future__ import annotations
import pandas as pd


def flatten_products(products_json) -> pd.DataFrame:
    if isinstance(products_json, dict) and "products" in products_json:
        products_json = products_json["products"]

    rows = []
    if isinstance(products_json, list):
        for p in products_json:
            if not isinstance(p, dict):
                continue
            rows.append({
                "product_id": p.get("product_id") or p.get("id") or p.get("sku"),
                "category": p.get("category") or p.get("product_category") or "unknown",
            })
    return pd.DataFrame(rows)