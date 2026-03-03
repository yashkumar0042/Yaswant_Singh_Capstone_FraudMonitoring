from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Paths (as per hierarchy)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "raw_layer"
OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers: IO
# =========================
def read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return pd.read_csv(path)


def read_json(name: str):
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"✅ Wrote: {path}")


# =========================
# Helpers: Cleaning
# =========================
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


# =========================
# Helpers: Features
# =========================
def discount_pct(gross: pd.Series, discount: pd.Series) -> pd.Series:
    gross = pd.to_numeric(gross, errors="coerce").fillna(0)
    discount = pd.to_numeric(discount, errors="coerce").fillna(0)
    denom = gross.replace(0, np.nan)
    pct = (discount / denom) * 100
    return pct.fillna(0).clip(lower=0)


def week_start(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    # Monday week start
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


# =========================
# Risk scoring config
# =========================
WEIGHTS = {
    "high_discount_flag": 15,
    "coupon_repeat_user_flag": 10,
    "coupon_device_reuse_flag": 10,
    "payment_failed_attempts_score": 20,
    "new_user_flag": 8,
    "new_user_plus_coupon": 7,
    "cod_flag": 6,
    "high_rto_pincode_flag": 12,
    "pincode_reuse_score": 8,
    "device_reuse_score": 8,
    "value_outlier_flag": 8,
    "refund_history_user_flag": 8,
}

REASON_CODES = {
    "high_discount_flag": "HIGH_DISCOUNT",
    "coupon_repeat_user_flag": "COUPON_REPEAT_USER",
    "coupon_device_reuse_flag": "COUPON_DEVICE_REUSE",
    "payment_failed_attempts_score": "PAYMENT_FAIL_SPIKE",
    "new_user_flag": "NEW_USER",
    "new_user_plus_coupon": "NEW_USER_COUPON",
    "cod_flag": "COD_ORDER",
    "high_rto_pincode_flag": "HIGH_RTO_PINCODE",
    "pincode_reuse_score": "PINCODE_REUSE",
    "device_reuse_score": "DEVICE_REUSE",
    "value_outlier_flag": "VALUE_OUTLIER",
    "refund_history_user_flag": "REFUND_HISTORY",
}


def clamp01(s: pd.Series) -> pd.Series:
    return s.fillna(0).clip(0, 1)


def compute_score_and_reasons(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()

    # ensure all features exist
    for f in WEIGHTS.keys():
        if f not in df.columns:
            df[f] = 0

    contrib_cols = []
    for feat, w in WEIGHTS.items():
        v = df[feat]
        if feat.endswith("_score"):
            v = clamp01(pd.to_numeric(v, errors="coerce"))
        else:
            v = pd.to_numeric(v, errors="coerce").fillna(0).clip(0, 1)
        ccol = f"contrib__{feat}"
        df[ccol] = w * v
        contrib_cols.append(ccol)

    df["risk_score"] = (
        df[contrib_cols].sum(axis=1).round().clip(0, 100).astype(int)
    )

    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-1, 39, 69, 100],
        labels=["Low", "Medium", "High"]
    ).astype("string")

    # top 3 reasons
    feat_from_col = {c: c.replace("contrib__", "") for c in contrib_cols}

    def top3(row) -> List[str]:
        pairs = []
        for c in contrib_cols:
            val = row[c]
            if pd.notna(val) and val > 0:
                feat = feat_from_col[c]
                pairs.append((val, REASON_CODES.get(feat, feat)))
        pairs.sort(reverse=True, key=lambda x: x[0])
        reasons = [p[1] for p in pairs[:3]]
        reasons += [""] * (3 - len(reasons))
        return reasons

    reasons_df = df.apply(lambda r: pd.Series(top3(r)), axis=1)
    reasons_df.columns = ["reason_1", "reason_2", "reason_3"]
    df = pd.concat([df, reasons_df], axis=1)

    # drop contrib columns (optional; keep if you want explainability debugging)
    df.drop(columns=contrib_cols, inplace=True, errors="ignore")

    return df


def recommend_action(row: pd.Series) -> str:
    band = row.get("risk_band", "")
    reasons = {row.get("reason_1", ""), row.get("reason_2", ""), row.get("reason_3", "")}

    if band == "High":
        if ("DEVICE_REUSE" in reasons or "PINCODE_REUSE" in reasons) and ("NEW_USER_COUPON" in reasons or "NEW_USER" in reasons):
            return "HOLD_FOR_MANUAL_REVIEW"
        if "PAYMENT_FAIL_SPIKE" in reasons:
            return "CALL_VERIFICATION"
        if "HIGH_RTO_PINCODE" in reasons and "COD_ORDER" in reasons:
            return "OTP_ADDRESS_CONFIRMATION"
        return "MANUAL_REVIEW"

    if band == "Medium":
        if "HIGH_DISCOUNT" in reasons or "NEW_USER_COUPON" in reasons:
            return "SOFT_FRICTION_OTP"
        return "MONITOR"

    return "ALLOW"


# =========================
# Products flatten
# =========================
def flatten_products(products_json) -> pd.DataFrame:
    """
    Supports:
    - list of product dicts
    - or {"products": [...]}
    """
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


# =========================
# Build Fact: 1 row per order
# =========================
def build_fact_orders_enriched(
    users: pd.DataFrame,
    sessions: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    shipments: pd.DataFrame,
    refunds: pd.DataFrame,
    coupons: pd.DataFrame,
    products: pd.DataFrame
) -> pd.DataFrame:

    # standardize + parse dates (only if those cols exist)
    users = parse_dt(standardize_df(users), ["signup_ts", "created_at"])
    sessions = parse_dt(standardize_df(sessions), ["session_ts", "created_at"])
    orders = parse_dt(standardize_df(orders), ["order_ts", "created_at"])
    order_items = standardize_df(order_items)
    payments = parse_dt(standardize_df(payments), ["payment_ts", "created_at"])
    shipments = parse_dt(standardize_df(shipments), ["shipment_ts", "created_at"])
    refunds = parse_dt(standardize_df(refunds), ["refund_ts", "created_at"])
    coupons = standardize_df(coupons)
    products = standardize_df(products)

    # dedup basics
    if "user_id" in users.columns:
        users = dedup_keep_last(users, ["user_id"], order_col="signup_ts" if "signup_ts" in users.columns else None)

    if "session_id" in sessions.columns:
        sessions = dedup_keep_last(sessions, ["session_id"], order_col="session_ts" if "session_ts" in sessions.columns else None)

    if "order_id" in orders.columns:
        orders = dedup_keep_last(orders, ["order_id"], order_col="order_ts" if "order_ts" in orders.columns else None)

    # enforce core columns (create if missing so pipeline runs)
    for c in ["order_id", "user_id", "session_id", "gross_amount", "discount_amount", "net_amount", "payment_method", "coupon_id"]:
        if c not in orders.columns:
            orders[c] = pd.NA

    # normalize categoricals
    orders["payment_method"] = safe_lower(orders["payment_method"]).fillna("unknown")
    orders["coupon_id"] = safe_lower(orders["coupon_id"]).fillna("none")

    if "shipping_city" in orders.columns:
        orders["shipping_city"] = safe_lower(orders["shipping_city"]).fillna("unknown")
    if "shipping_pincode" in orders.columns:
        orders["shipping_pincode"] = orders["shipping_pincode"].astype("string").str.strip().fillna("unknown")
    else:
        orders["shipping_pincode"] = "unknown"

    # numeric amounts
    for c in ["gross_amount", "discount_amount", "net_amount"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce").fillna(0)

    # discount pct
    orders["discount_pct"] = discount_pct(orders["gross_amount"], orders["discount_amount"])

    # products mapping (product_id -> category)
    if "product_id" in products.columns:
        products["product_id"] = products["product_id"].astype("string").str.strip()
    if "product_id" in order_items.columns:
        order_items["product_id"] = order_items["product_id"].astype("string").str.strip()

    if "product_id" in order_items.columns and "product_id" in products.columns:
        order_items = order_items.merge(
            products[["product_id", "category"]].drop_duplicates(),
            on="product_id",
            how="left"
        )

    # order_items aggregation
    if "qty" not in order_items.columns:
        order_items["qty"] = 1
    order_items["qty"] = pd.to_numeric(order_items["qty"], errors="coerce").fillna(1)

    items_agg = order_items.groupby("order_id", as_index=False).agg(
        item_count=("product_id", "nunique") if "product_id" in order_items.columns else ("order_id", "size"),
        total_qty=("qty", "sum"),
    )

    if "category" in order_items.columns:
        tmp = order_items.copy()
        tmp["category"] = tmp["category"].astype("string").fillna("unknown").str.lower().str.strip()
        cat_qty = tmp.groupby(["order_id", "category"], as_index=False)["qty"].sum()
        cat_qty = cat_qty.sort_values(["order_id", "qty"], ascending=[True, False])
        top_cat = cat_qty.drop_duplicates("order_id", keep="first").rename(columns={"category": "top_category"})
        items_agg = items_agg.merge(top_cat[["order_id", "top_category"]], on="order_id", how="left")
    else:
        items_agg["top_category"] = "unknown"

    # payments aggregation
    if "status" in payments.columns:
        payments["status"] = safe_lower(payments["status"])
        payments["is_fail"] = payments["status"].str.contains("fail", na=False).astype(int)
        payments["is_success"] = payments["status"].str.contains("success", na=False).astype(int)
    else:
        payments["is_fail"] = 0
        payments["is_success"] = 0

    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_attempts=("order_id", "size"),
        failed_attempts=("is_fail", "sum"),
        success_attempts=("is_success", "sum"),
    )
    pay_agg["payment_failed_attempts_score"] = (pay_agg["failed_attempts"].clip(upper=10) / 10.0)

    # refunds aggregation
    if "refund_amount" not in refunds.columns:
        refunds["refund_amount"] = 0
    refunds["refund_amount"] = pd.to_numeric(refunds["refund_amount"], errors="coerce").fillna(0)

    refund_agg = refunds.groupby("order_id", as_index=False).agg(
        refund_flag=("refund_amount", lambda x: int(x.sum() > 0)),
        refund_amount=("refund_amount", "sum"),
    )

    # shipments aggregation
    if "status" in shipments.columns:
        shipments["status"] = safe_lower(shipments["status"])
        shipments["rto_flag"] = shipments["status"].str.contains("rto", na=False).astype(int)
        shipments["delivered_flag"] = shipments["status"].str.contains("deliver", na=False).astype(int)
    else:
        shipments["rto_flag"] = 0
        shipments["delivered_flag"] = 0

    ship_agg = shipments.groupby("order_id", as_index=False).agg(
        rto_flag=("rto_flag", "max"),
        delivered_flag=("delivered_flag", "max"),
    )

    # coupons reference
    if "coupon_id" in coupons.columns:
        coupons["coupon_id"] = safe_lower(coupons["coupon_id"]).fillna("none")
    if "discount_pct" in coupons.columns:
        coupons["coupon_discount_pct"] = pd.to_numeric(coupons["discount_pct"], errors="coerce")
    elif "coupon_discount_pct" not in coupons.columns:
        coupons["coupon_discount_pct"] = pd.NA

    coupons_ref = coupons[["coupon_id", "coupon_discount_pct"]].drop_duplicates()

    # base fact join
    fact = orders.copy()
    fact = fact.merge(users, on="user_id", how="left", suffixes=("", "_user")) if "user_id" in users.columns else fact
    fact = fact.merge(sessions, on="session_id", how="left", suffixes=("", "_session")) if "session_id" in sessions.columns else fact
    fact = fact.merge(items_agg, on="order_id", how="left")
    fact = fact.merge(pay_agg, on="order_id", how="left")
    fact = fact.merge(refund_agg, on="order_id", how="left")
    fact = fact.merge(ship_agg, on="order_id", how="left")
    fact = fact.merge(coupons_ref, on="coupon_id", how="left")

    # fill
    for c, d in [
        ("item_count", 0), ("total_qty", 0), ("top_category", "unknown"),
        ("payment_attempts", 0), ("failed_attempts", 0), ("success_attempts", 0),
        ("payment_failed_attempts_score", 0),
        ("refund_flag", 0), ("refund_amount", 0),
        ("rto_flag", 0), ("delivered_flag", 0),
    ]:
        if c in fact.columns:
            fact[c] = fact[c].fillna(d)

    # -------------------------
    # Feature engineering (12)
    # -------------------------
    fact["high_discount_flag"] = (fact["discount_pct"] >= 50).astype(int)
    fact["coupon_used_flag"] = (fact["coupon_id"].astype("string") != "none").astype(int)
    fact["cod_flag"] = fact["payment_method"].astype("string").str.contains("cod", na=False).astype(int)

    # new user
    # supports either signup_ts or created_at in users and order_ts or created_at in orders
    signup_col = "signup_ts" if "signup_ts" in fact.columns else ("created_at" if "created_at" in fact.columns else None)
    order_col = "order_ts" if "order_ts" in fact.columns else ("created_at" if "created_at" in fact.columns else None)

    if signup_col and order_col:
        days = (pd.to_datetime(fact[order_col], errors="coerce") - pd.to_datetime(fact[signup_col], errors="coerce")).dt.days
        fact["days_since_signup"] = days.fillna(9999).astype(int)
        fact["new_user_flag"] = (fact["days_since_signup"] <= 7).astype(int)
    else:
        fact["days_since_signup"] = 9999
        fact["new_user_flag"] = 0

    fact["new_user_plus_coupon"] = ((fact["new_user_flag"] == 1) & (fact["coupon_used_flag"] == 1)).astype(int)

    # refund history by user (dataset-wide approximation)
    if "user_id" in fact.columns:
        user_ref = fact.groupby("user_id")["refund_flag"].sum().rename("prior_refunds_count").reset_index()
        fact = fact.merge(user_ref, on="user_id", how="left")
        fact["prior_refunds_count"] = fact["prior_refunds_count"].fillna(0).astype(int)
        fact["refund_history_user_flag"] = (fact["prior_refunds_count"] >= 2).astype(int)
    else:
        fact["prior_refunds_count"] = 0
        fact["refund_history_user_flag"] = 0

    # device reuse / pincode reuse
    if "device_id" in fact.columns:
        fact["device_id"] = fact["device_id"].astype("string").str.strip().fillna("unknown")
        device_cnt = fact.groupby("device_id")["order_id"].count().rename("device_orders_count").reset_index()
        fact = fact.merge(device_cnt, on="device_id", how="left")
        fact["device_orders_count"] = fact["device_orders_count"].fillna(1).astype(int)
        fact["device_reuse_score"] = (fact["device_orders_count"].clip(upper=20) / 20.0)
    else:
        fact["device_orders_count"] = 1
        fact["device_reuse_score"] = 0

    if "shipping_pincode" in fact.columns:
        pin_cnt = fact.groupby("shipping_pincode")["order_id"].count().rename("pincode_orders_count").reset_index()
        fact = fact.merge(pin_cnt, on="shipping_pincode", how="left")
        fact["pincode_orders_count"] = fact["pincode_orders_count"].fillna(1).astype(int)
        fact["pincode_reuse_score"] = (fact["pincode_orders_count"].clip(upper=20) / 20.0)
    else:
        fact["pincode_orders_count"] = 1
        fact["pincode_reuse_score"] = 0

    # coupon repeat user / device
    if "user_id" in fact.columns:
        user_coupon = fact[fact["coupon_used_flag"] == 1].groupby("user_id")["order_id"].count().rename("coupon_orders_user").reset_index()
        fact = fact.merge(user_coupon, on="user_id", how="left")
        fact["coupon_orders_user"] = fact["coupon_orders_user"].fillna(0).astype(int)
        fact["coupon_repeat_user_flag"] = (fact["coupon_orders_user"] >= 3).astype(int)
    else:
        fact["coupon_orders_user"] = 0
        fact["coupon_repeat_user_flag"] = 0

    if "device_id" in fact.columns:
        dev_coupon_users = (
            fact[fact["coupon_used_flag"] == 1]
            .groupby("device_id")["user_id"].nunique()
            .rename("coupon_users_per_device")
            .reset_index()
        )
        fact = fact.merge(dev_coupon_users, on="device_id", how="left")
        fact["coupon_users_per_device"] = fact["coupon_users_per_device"].fillna(0).astype(int)
        fact["coupon_device_reuse_flag"] = (fact["coupon_users_per_device"] >= 3).astype(int)
    else:
        fact["coupon_users_per_device"] = 0
        fact["coupon_device_reuse_flag"] = 0

    # high RTO pincode flag (dataset-derived)
    p = fact.groupby("shipping_pincode").agg(
        pincode_orders=("order_id", "count"),
        pincode_rto=("rto_flag", "sum"),
    ).reset_index()
    p["pincode_rto_rate"] = (p["pincode_rto"] / p["pincode_orders"]).fillna(0)
    fact = fact.merge(p[["shipping_pincode", "pincode_rto_rate"]], on="shipping_pincode", how="left")
    fact["pincode_rto_rate"] = fact["pincode_rto_rate"].fillna(0)
    fact["high_rto_pincode_flag"] = (fact["pincode_rto_rate"] >= 0.25).astype(int)

    # value outlier within category
    fact["top_category"] = fact["top_category"].astype("string").fillna("unknown")
    fact["net_amount"] = pd.to_numeric(fact["net_amount"], errors="coerce").fillna(0)
    fact["net_amount_z"] = zscore_within_group(fact, "top_category", "net_amount")
    fact["value_outlier_flag"] = (fact["net_amount_z"].abs() >= 2.5).astype(int)

    return fact


# =========================
# Weekly user table
# =========================
def build_user_weekly(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    if "order_ts" not in df.columns:
        df["order_ts"] = pd.NaT

    df["week_start"] = week_start(df["order_ts"]).astype("string")

    out = df.groupby(["user_id", "week_start"], as_index=False).agg(
        orders_count=("order_id", "count"),
        net_revenue=("net_amount", "sum"),
        refunds_count=("refund_flag", "sum"),
        refund_amount=("refund_amount", "sum"),
        coupon_orders_count=("coupon_used_flag", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        cod_orders_count=("cod_flag", "sum"),
        rto_count=("rto_flag", "sum"),
        payment_failures_count=("failed_attempts", "sum"),
        risk_score_avg=("risk_score", "mean"),
    )

    out["avg_discount_pct"] = out["avg_discount_pct"].round(2)
    out["risk_score_avg"] = out["risk_score_avg"].round(1)
    return out


# =========================
# Investigation queue
# =========================
def build_queue(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    df["recommended_action"] = df.apply(recommend_action, axis=1)

    # ensure evidence fields exist
    defaults = {
        "discount_pct": 0,
        "failed_attempts": 0,
        "device_orders_count": 1,
        "pincode_orders_count": 1,
        "days_since_signup": 9999,
        "prior_refunds_count": 0,
        "shipping_pincode": "unknown",
        "coupon_id": "none",
        "payment_method": "unknown",
        "net_amount": 0,
        "order_ts": pd.NaT,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    queue = df[[
        "order_id", "user_id", "order_ts", "net_amount", "payment_method", "coupon_id",
        "shipping_pincode",
        "risk_score", "risk_band",
        "reason_1", "reason_2", "reason_3",
        "recommended_action",
        "discount_pct", "failed_attempts", "device_orders_count", "pincode_orders_count",
        "days_since_signup", "prior_refunds_count",
    ]].copy()

    queue = queue.sort_values(["risk_score", "net_amount"], ascending=[False, False]).reset_index(drop=True)
    queue.insert(0, "rank", range(1, len(queue) + 1))
    return queue


# =========================
# Validations
# =========================
def validate_fact(fact: pd.DataFrame) -> None:
    if "order_id" not in fact.columns:
        raise ValueError("fact_orders_enriched must have order_id")

    dup = fact["order_id"].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Join explosion: {dup} duplicate order_id rows in fact_orders_enriched")

    if (fact["risk_score"] < 0).any() or (fact["risk_score"] > 100).any():
        raise ValueError("risk_score out of bounds 0..100")


# =========================
# Main pipeline
# =========================
def main():
    print("📦 Loading raw files from:", RAW_DIR)

    users = read_csv("users.csv")
    sessions = read_csv("sessions.csv")
    orders = read_csv("orders.csv")
    order_items = read_csv("order_items.csv")
    payments = read_csv("payments.csv")
    shipments = read_csv("shipments.csv")
    refunds = read_csv("refunds.csv")
    coupons = read_csv("coupons.csv")
    products_json = read_json("products.json")
    products = flatten_products(products_json)

    fact = build_fact_orders_enriched(
        users=users,
        sessions=sessions,
        orders=orders,
        order_items=order_items,
        payments=payments,
        shipments=shipments,
        refunds=refunds,
        coupons=coupons,
        products=products,
    )

    # scoring + reasons
    fact = compute_score_and_reasons(fact)

    # weekly + queue
    weekly = build_user_weekly(fact)
    queue = build_queue(fact)

    # validations
    validate_fact(fact)

    # outputs
    write_csv(fact, "fact_orders_enriched.csv")
    write_csv(weekly, "fact_user_risk_weekly.csv")
    write_csv(queue, "investigation_queue.csv")

    print("\n🎉 Pipeline complete. Outputs are in /data (generated outputs only).")


if __name__ == "__main__":
    main()