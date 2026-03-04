from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# For dashboard exports
import matplotlib.pyplot as plt

# For Excel dashboard
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

# For final memo PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# =========================
# Paths (as per hierarchy)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "raw_layer"

DATA_DIR = PROJECT_ROOT / "data"                  # generated outputs only
ANALYSIS_DIR = PROJECT_ROOT / "analysis"           # generated analysis artifacts
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"         # dashboard file
EXPORTS_DIR = DASHBOARD_DIR / "exports"            # charts/images/pdf exports
FINAL_STORY_DIR = PROJECT_ROOT / "final_story"     # final memo/deck

for d in [DATA_DIR, ANALYSIS_DIR, DASHBOARD_DIR, EXPORTS_DIR, FINAL_STORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
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

    df["risk_score"] = df[contrib_cols].sum(axis=1).round().clip(0, 100).astype(int)
    df["risk_band"] = pd.cut(df["risk_score"], bins=[-1, 39, 69, 100], labels=["Low", "Medium", "High"]).astype("string")

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

    users = parse_dt(standardize_df(users), ["signup_ts", "created_at"])
    sessions = parse_dt(standardize_df(sessions), ["session_ts", "created_at"])
    orders = parse_dt(standardize_df(orders), ["order_ts", "created_at"])
    order_items = standardize_df(order_items)
    payments = parse_dt(standardize_df(payments), ["payment_ts", "created_at"])
    shipments = parse_dt(standardize_df(shipments), ["shipment_ts", "created_at"])
    refunds = parse_dt(standardize_df(refunds), ["refund_ts", "created_at"])
    coupons = standardize_df(coupons)
    products = standardize_df(products)

    if "user_id" in users.columns:
        users = dedup_keep_last(users, ["user_id"], order_col="signup_ts" if "signup_ts" in users.columns else None)

    if "session_id" in sessions.columns:
        sessions = dedup_keep_last(sessions, ["session_id"], order_col="session_ts" if "session_ts" in sessions.columns else None)

    if "order_id" in orders.columns:
        orders = dedup_keep_last(orders, ["order_id"], order_col="order_ts" if "order_ts" in orders.columns else None)

    for c in ["order_id", "user_id", "session_id", "gross_amount", "discount_amount", "net_amount", "payment_method", "coupon_id"]:
        if c not in orders.columns:
            orders[c] = pd.NA

    orders["payment_method"] = safe_lower(orders["payment_method"]).fillna("unknown")
    orders["coupon_id"] = safe_lower(orders["coupon_id"]).fillna("none")

    if "shipping_city" in orders.columns:
        orders["shipping_city"] = safe_lower(orders["shipping_city"]).fillna("unknown")

    if "shipping_pincode" in orders.columns:
        orders["shipping_pincode"] = orders["shipping_pincode"].astype("string").str.strip().fillna("unknown")
    else:
        orders["shipping_pincode"] = "unknown"

    for c in ["gross_amount", "discount_amount", "net_amount"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce").fillna(0)

    orders["discount_pct"] = discount_pct(orders["gross_amount"], orders["discount_amount"])

    if "product_id" in products.columns:
        products["product_id"] = products["product_id"].astype("string").str.strip()
    if "product_id" in order_items.columns:
        order_items["product_id"] = order_items["product_id"].astype("string").str.strip()

    if "product_id" in order_items.columns and "product_id" in products.columns:
        order_items = order_items.merge(products[["product_id", "category"]].drop_duplicates(), on="product_id", how="left")

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

    if "refund_amount" not in refunds.columns:
        refunds["refund_amount"] = 0
    refunds["refund_amount"] = pd.to_numeric(refunds["refund_amount"], errors="coerce").fillna(0)

    refund_agg = refunds.groupby("order_id", as_index=False).agg(
        refund_flag=("refund_amount", lambda x: int(x.sum() > 0)),
        refund_amount=("refund_amount", "sum"),
    )

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

    if "coupon_id" in coupons.columns:
        coupons["coupon_id"] = safe_lower(coupons["coupon_id"]).fillna("none")

    if "discount_pct" in coupons.columns:
        coupons["coupon_discount_pct"] = pd.to_numeric(coupons["discount_pct"], errors="coerce")
    elif "coupon_discount_pct" not in coupons.columns:
        coupons["coupon_discount_pct"] = pd.NA

    coupons_ref = coupons[["coupon_id", "coupon_discount_pct"]].drop_duplicates()

    fact = orders.copy()
    if "user_id" in users.columns:
        fact = fact.merge(users, on="user_id", how="left", suffixes=("", "_user"))
    if "session_id" in sessions.columns:
        fact = fact.merge(sessions, on="session_id", how="left", suffixes=("", "_session"))

    fact = fact.merge(items_agg, on="order_id", how="left")
    fact = fact.merge(pay_agg, on="order_id", how="left")
    fact = fact.merge(refund_agg, on="order_id", how="left")
    fact = fact.merge(ship_agg, on="order_id", how="left")
    fact = fact.merge(coupons_ref, on="coupon_id", how="left")

    for c, d in [
        ("item_count", 0), ("total_qty", 0), ("top_category", "unknown"),
        ("payment_attempts", 0), ("failed_attempts", 0), ("success_attempts", 0),
        ("payment_failed_attempts_score", 0),
        ("refund_flag", 0), ("refund_amount", 0),
        ("rto_flag", 0), ("delivered_flag", 0),
    ]:
        if c in fact.columns:
            fact[c] = fact[c].fillna(d)

    # ---- engineered signals ----
    fact["high_discount_flag"] = (fact["discount_pct"] >= 50).astype(int)
    fact["coupon_used_flag"] = (fact["coupon_id"].astype("string") != "none").astype(int)
    fact["cod_flag"] = fact["payment_method"].astype("string").str.contains("cod", na=False).astype(int)

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

    if "user_id" in fact.columns:
        user_ref = fact.groupby("user_id")["refund_flag"].sum().rename("prior_refunds_count").reset_index()
        fact = fact.merge(user_ref, on="user_id", how="left")
        fact["prior_refunds_count"] = fact["prior_refunds_count"].fillna(0).astype(int)
        fact["refund_history_user_flag"] = (fact["prior_refunds_count"] >= 2).astype(int)
    else:
        fact["prior_refunds_count"] = 0
        fact["refund_history_user_flag"] = 0

    if "device_id" in fact.columns:
        fact["device_id"] = fact["device_id"].astype("string").str.strip().fillna("unknown")
        device_cnt = fact.groupby("device_id")["order_id"].count().rename("device_orders_count").reset_index()
        fact = fact.merge(device_cnt, on="device_id", how="left")
        fact["device_orders_count"] = fact["device_orders_count"].fillna(1).astype(int)
        fact["device_reuse_score"] = (fact["device_orders_count"].clip(upper=20) / 20.0)
    else:
        fact["device_orders_count"] = 1
        fact["device_reuse_score"] = 0

    pin_cnt = fact.groupby("shipping_pincode")["order_id"].count().rename("pincode_orders_count").reset_index()
    fact = fact.merge(pin_cnt, on="shipping_pincode", how="left")
    fact["pincode_orders_count"] = fact["pincode_orders_count"].fillna(1).astype(int)
    fact["pincode_reuse_score"] = (fact["pincode_orders_count"].clip(upper=20) / 20.0)

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

    p = fact.groupby("shipping_pincode").agg(
        pincode_orders=("order_id", "count"),
        pincode_rto=("rto_flag", "sum"),
    ).reset_index()
    p["pincode_rto_rate"] = (p["pincode_rto"] / p["pincode_orders"]).fillna(0)
    fact = fact.merge(p[["shipping_pincode", "pincode_rto_rate"]], on="shipping_pincode", how="left")
    fact["pincode_rto_rate"] = fact["pincode_rto_rate"].fillna(0)
    fact["high_rto_pincode_flag"] = (fact["pincode_rto_rate"] >= 0.25).astype(int)

    fact["top_category"] = fact["top_category"].astype("string").fillna("unknown")
    fact["net_amount"] = pd.to_numeric(fact["net_amount"], errors="coerce").fillna(0)
    fact["net_amount_z"] = zscore_within_group(fact, "top_category", "net_amount")
    fact["value_outlier_flag"] = (fact["net_amount_z"].abs() >= 2.5).astype(int)

    return fact


# =========================
# Weekly + Queue
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


def build_queue(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    df["recommended_action"] = df.apply(recommend_action, axis=1)

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
# Analysis artifacts
# =========================
def build_analysis_artifacts(fact: pd.DataFrame, weekly: pd.DataFrame, queue: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Weekly KPI table (for dashboard + report)
    kpi_weekly = weekly.groupby("week_start", as_index=False).agg(
        orders=("orders_count", "sum"),
        net_revenue=("net_revenue", "sum"),
        refunds=("refunds_count", "sum"),
        refund_amount=("refund_amount", "sum"),
        rto=("rto_count", "sum"),
        coupon_orders=("coupon_orders_count", "sum"),
        payment_failures=("payment_failures_count", "sum"),
        avg_risk=("risk_score_avg", "mean"),
    )
    kpi_weekly["refund_rate"] = (kpi_weekly["refunds"] / kpi_weekly["orders"]).replace([np.inf, np.nan], 0).round(4)
    kpi_weekly["rto_rate"] = (kpi_weekly["rto"] / kpi_weekly["orders"]).replace([np.inf, np.nan], 0).round(4)

    # Top patterns (simple but strong)
    # Pattern 1: coupon_id
    by_coupon = fact.groupby("coupon_id", as_index=False).agg(
        orders=("order_id", "count"),
        refund_amount=("refund_amount", "sum"),
        rto=("rto_flag", "sum"),
        avg_risk=("risk_score", "mean"),
        avg_discount=("discount_pct", "mean"),
    ).sort_values(["avg_risk", "orders"], ascending=[False, False])

    # Pattern 2: pincode
    by_pincode = fact.groupby("shipping_pincode", as_index=False).agg(
        orders=("order_id", "count"),
        refund_amount=("refund_amount", "sum"),
        rto=("rto_flag", "sum"),
        avg_risk=("risk_score", "mean"),
        rto_rate=("pincode_rto_rate", "mean"),
    ).sort_values(["avg_risk", "orders"], ascending=[False, False])

    # Pattern 3: device_id (if present)
    if "device_id" in fact.columns:
        by_device = fact.groupby("device_id", as_index=False).agg(
            orders=("order_id", "count"),
            users=("user_id", "nunique"),
            refund_amount=("refund_amount", "sum"),
            avg_risk=("risk_score", "mean"),
        ).sort_values(["users", "avg_risk"], ascending=[False, False])
    else:
        by_device = pd.DataFrame(columns=["device_id", "orders", "users", "refund_amount", "avg_risk"])

    # Summarize patterns (top 10 each)
    patterns_summary = pd.concat([
        by_coupon.head(10).assign(pattern_type="coupon"),
        by_pincode.head(10).assign(pattern_type="pincode"),
        by_device.head(10).assign(pattern_type="device"),
    ], ignore_index=True)

    return {
        "kpi_weekly": kpi_weekly,
        "patterns_summary": patterns_summary,
        "by_coupon": by_coupon,
        "by_pincode": by_pincode,
        "by_device": by_device,
    }


def export_analysis_report_html(kpi_weekly: pd.DataFrame, patterns_summary: pd.DataFrame, out_path: Path) -> None:
    html = f"""
    <html><head><meta charset="utf-8"><title>Fraud Monitoring Analysis Report</title></head>
    <body style="font-family: Arial, sans-serif; margin: 24px;">
      <h1>Fraud Monitoring – Analysis Report</h1>

      <h2>Weekly KPI Summary</h2>
      {kpi_weekly.to_html(index=False)}

      <h2>Top Patterns (Top 10 each)</h2>
      <p><b>pattern_type</b> indicates whether the pattern is coupon / pincode / device.</p>
      {patterns_summary.to_html(index=False)}

      <h2>Notes</h2>
      <ul>
        <li>Refund & RTO are treated as proxy outcomes for risk impact.</li>
        <li>Risk score is explainable: reasons_1..3 are derived from weighted signals.</li>
      </ul>
    </body></html>
    """
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Wrote: {out_path}")


# =========================
# Dashboard generation (Excel + images)
# =========================
def export_charts(kpi_weekly: pd.DataFrame, fact: pd.DataFrame) -> None:
    # 1) Refund amount trend
    if not kpi_weekly.empty and "week_start" in kpi_weekly.columns:
        plt.figure()
        plt.plot(kpi_weekly["week_start"], kpi_weekly["refund_amount"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Weekly Refund Amount Trend")
        plt.tight_layout()
        path1 = EXPORTS_DIR / "weekly_refunds_trend.png"
        plt.savefig(path1, dpi=160)
        plt.close()
        print(f"✅ Wrote: {path1}")

    # 2) Risk band split
    if "risk_band" in fact.columns:
        band_counts = fact["risk_band"].value_counts(dropna=False)
        plt.figure()
        plt.pie(band_counts.values, labels=band_counts.index.astype(str), autopct="%1.1f%%")
        plt.title("Risk Band Split (All Orders)")
        plt.tight_layout()
        path2 = EXPORTS_DIR / "risk_band_split.png"
        plt.savefig(path2, dpi=160)
        plt.close()
        print(f"✅ Wrote: {path2}")


def autosize_excel_columns(ws) -> None:
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(v))
        ws.column_dimensions[col_letter].width = min(45, max(10, max_len + 2))


def export_dashboard_xlsx(
    fact: pd.DataFrame,
    weekly: pd.DataFrame,
    queue: pd.DataFrame,
    kpi_weekly: pd.DataFrame,
    patterns_summary: pd.DataFrame
) -> None:
    xlsx_path = DASHBOARD_DIR / "dashboard.xlsx"

    # Write sheets with pandas
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Keep dashboards simple + useful
        kpi_weekly.to_excel(writer, sheet_name="KPI_Weekly", index=False)
        patterns_summary.to_excel(writer, sheet_name="Patterns", index=False)
        queue.head(500).to_excel(writer, sheet_name="Investigation_Queue", index=False)  # keep file lighter
        # Optional: store a smaller subset of fact for debugging
        fact.head(2000).to_excel(writer, sheet_name="Sample_Fact", index=False)

    # Post-format using openpyxl
    wb = load_workbook(xlsx_path)

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        # header style
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        autosize_excel_columns(ws)

    wb.save(xlsx_path)
    print(f"✅ Wrote: {xlsx_path}")


# =========================
# Final story: PDF memo
# =========================
def export_final_memo_pdf(
    fact: pd.DataFrame,
    kpi_weekly: pd.DataFrame,
    patterns_summary: pd.DataFrame,
    queue: pd.DataFrame,
    out_path: Path
) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    # ---------- helpers ----------
    def wrap_text(c: canvas.Canvas, text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
        """
        Wraps text based on actual rendered width in ReportLab.
        """
        c.setFont(font_name, font_size)
        words = (text or "").split()
        if not words:
            return [""]

        lines = []
        cur = words[0]
        for w in words[1:]:
            trial = cur + " " + w
            if c.stringWidth(trial, font_name, font_size) <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def ensure_space(c: canvas.Canvas, y: float, needed: float, page_w: float, page_h: float):
        """
        Starts a new page if not enough vertical space.
        Returns updated y.
        """
        bottom_margin = 0.75 * inch
        if y - needed < bottom_margin:
            c.showPage()
            return page_h - top_margin
        return y

    def draw_paragraph(c: canvas.Canvas, x: float, y: float, text: str,
                       font_name: str = "Helvetica", font_size: int = 11,
                       line_gap: int = 4) -> float:
        """
        Draw a wrapped paragraph and return new y.
        """
        max_width = page_w - left_margin - right_margin
        lines = wrap_text(c, text, font_name, font_size, max_width)
        line_height = font_size + line_gap

        y = ensure_space(c, y, line_height * len(lines), page_w, page_h)

        c.setFont(font_name, font_size)
        for line in lines:
            c.drawString(x, y, line)
            y -= line_height
        return y

    def draw_bullets(c: canvas.Canvas, x: float, y: float, bullets: List[str],
                     font_name: str = "Helvetica", font_size: int = 11,
                     bullet_indent: float = 12, line_gap: int = 3) -> float:
        """
        Draw bullet list with wrapping.
        """
        max_width = page_w - left_margin - right_margin - bullet_indent
        line_height = font_size + line_gap
        c.setFont(font_name, font_size)

        for b in bullets:
            # Wrap bullet text and draw with bullet symbol only on first line
            lines = wrap_text(c, b, font_name, font_size, max_width)
            needed = line_height * max(1, len(lines))
            y = ensure_space(c, y, needed + 6, page_w, page_h)

            # First line: bullet
            c.drawString(x, y, "•")
            c.drawString(x + bullet_indent, y, lines[0])
            y -= line_height

            # Remaining lines: indent aligned with text
            for line in lines[1:]:
                c.drawString(x + bullet_indent, y, line)
                y -= line_height

        return y

    # ---------- compute summary stats ----------
    total_orders = len(fact)
    total_refund = float(pd.to_numeric(fact.get("refund_amount", 0), errors="coerce").fillna(0).sum())
    total_rto = int(pd.to_numeric(fact.get("rto_flag", 0), errors="coerce").fillna(0).sum())
    high_risk = int((fact.get("risk_band", "") == "High").sum()) if "risk_band" in fact.columns else 0

    top_queue = queue.head(5)[["order_id", "risk_score", "risk_band", "reason_1", "reason_2", "reason_3", "recommended_action"]] \
        if not queue.empty else pd.DataFrame()

    top_patterns = patterns_summary.head(5).to_dict("records") if not patterns_summary.empty else []

    # ---------- PDF layout settings ----------
    c = canvas.Canvas(str(out_path), pagesize=letter)
    page_w, page_h = letter

    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top_margin = 0.75 * inch

    x = left_margin
    y = page_h - top_margin

    # ---------- title ----------
    y = draw_paragraph(c, x, y, "Final Memo — Fraud Monitoring & Investigation Dashboard",
                       font_name="Helvetica-Bold", font_size=16, line_gap=6)
    y -= 6
    y = draw_paragraph(c, x, y, "Generated automatically from ETL outputs (data, analysis, dashboard, and queue).",
                       font_name="Helvetica", font_size=10, line_gap=4)
    y -= 10

    # ---------- executive summary ----------
    y = draw_paragraph(c, x, y, "Executive Summary", font_name="Helvetica-Bold", font_size=12, line_gap=5)
    y -= 2
    bullets = [
        f"Total orders analyzed: {total_orders}",
        f"Total refund amount (proxy loss): {total_refund:,.2f}",
        f"Total RTO count (proxy risk): {total_rto}",
        f"High-risk orders (risk_band=High): {high_risk}",
        "Risk scoring is explainable (0–100) with top 3 reasons captured per order.",
        "Investigation queue ranks suspicious orders and recommends an action (manual review / OTP / call verification)."
    ]
    y = draw_bullets(c, x, y, bullets, font_size=11)

    y -= 10

    # ---------- top patterns ----------
    y = draw_paragraph(c, x, y, "Top Patterns (snapshot)", font_name="Helvetica-Bold", font_size=12, line_gap=5)
    y -= 2

    if not top_patterns:
        y = draw_paragraph(c, x, y, "No patterns available (insufficient data).", font_size=11)
    else:
        pat_lines = []
        for p in top_patterns:
            ptype = p.get("pattern_type", "pattern")
            # Keep it readable in memo: compact representation
            pat_lines.append(f"[{ptype}] " + ", ".join([f"{k}={p[k]}" for k in list(p.keys())[:6] if k in p]))
        y = draw_bullets(c, x, y, pat_lines, font_size=10)

    y -= 10

    # ---------- sample investigation queue ----------
    y = draw_paragraph(c, x, y, "Sample Investigation Queue (Top 5)", font_name="Helvetica-Bold", font_size=12, line_gap=5)
    y -= 2

    if top_queue.empty:
        y = draw_paragraph(c, x, y, "Queue not available (empty).", font_size=11)
    else:
        # Render as wrapped bullet-like lines (simple table in text)
        q_bullets = []
        for _, r in top_queue.iterrows():
            q_bullets.append(
                f"order_id={r['order_id']}, score={r['risk_score']}, band={r['risk_band']}, "
                f"reasons=({r['reason_1']}, {r['reason_2']}, {r['reason_3']}), action={r['recommended_action']}"
            )
        y = draw_bullets(c, x, y, q_bullets, font_size=10)

    y -= 10

    # ---------- next actions ----------
    y = draw_paragraph(c, x, y, "Next Actions (recommended)", font_name="Helvetica-Bold", font_size=12, line_gap=5)
    y -= 2
    actions = [
        "Apply manual review / OTP friction for High-risk orders.",
        "Limit coupon usage by device/pincode for repeated abuse patterns.",
        "Track weekly KPIs (refund_amount, refund_rate, rto_rate, flagged orders) as guardrails after controls."
    ]
    y = draw_bullets(c, x, y, actions, font_size=11)

    c.save()
    print(f"✅ Wrote: {out_path}")

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

    fact = compute_score_and_reasons(fact)
    weekly = build_user_weekly(fact)
    queue = build_queue(fact)

    validate_fact(fact)

    # 1) Core ETL outputs
    write_csv(fact, DATA_DIR / "fact_orders_enriched.csv")
    write_csv(weekly, DATA_DIR / "fact_user_risk_weekly.csv")
    write_csv(queue, DATA_DIR / "investigation_queue.csv")

    # 2) Analysis outputs
    artifacts = build_analysis_artifacts(fact, weekly, queue)
    kpi_weekly = artifacts["kpi_weekly"]
    patterns_summary = artifacts["patterns_summary"]

    write_csv(kpi_weekly, ANALYSIS_DIR / "kpi_weekly.csv")
    write_csv(patterns_summary, ANALYSIS_DIR / "patterns_summary.csv")
    export_analysis_report_html(kpi_weekly, patterns_summary, ANALYSIS_DIR / "analysis_report.html")

    # 3) Dashboard outputs
    export_charts(kpi_weekly, fact)
    export_dashboard_xlsx(fact, weekly, queue, kpi_weekly, patterns_summary)

    # 4) Final story outputs
    export_final_memo_pdf(
        fact=fact,
        kpi_weekly=kpi_weekly,
        patterns_summary=patterns_summary,
        queue=queue,
        out_path=FINAL_STORY_DIR / "final_memo.pdf"
    )

    print("\n🎉 All done.")
    print("Core CSVs in /data")
    print("Analysis report in /analysis")
    print("Dashboard in /dashboard")
    print("Final memo in /final_story")


if __name__ == "__main__":
    main()