from __future__ import annotations

import pandas as pd
from .features import week_start
from .risk_scoring import recommend_action


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