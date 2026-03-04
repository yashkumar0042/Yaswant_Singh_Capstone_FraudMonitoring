from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


def build_analysis_artifacts(fact: pd.DataFrame, weekly: pd.DataFrame, queue: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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

    by_coupon = fact.groupby("coupon_id", as_index=False).agg(
        orders=("order_id", "count"),
        refund_amount=("refund_amount", "sum"),
        rto=("rto_flag", "sum"),
        avg_risk=("risk_score", "mean"),
        avg_discount=("discount_pct", "mean"),
    ).sort_values(["avg_risk", "orders"], ascending=[False, False])

    by_pincode = fact.groupby("shipping_pincode", as_index=False).agg(
        orders=("order_id", "count"),
        refund_amount=("refund_amount", "sum"),
        rto=("rto_flag", "sum"),
        avg_risk=("risk_score", "mean"),
        rto_rate=("pincode_rto_rate", "mean"),
    ).sort_values(["avg_risk", "orders"], ascending=[False, False])

    if "device_id" in fact.columns:
        by_device = fact.groupby("device_id", as_index=False).agg(
            orders=("order_id", "count"),
            users=("user_id", "nunique"),
            refund_amount=("refund_amount", "sum"),
            avg_risk=("risk_score", "mean"),
        ).sort_values(["users", "avg_risk"], ascending=[False, False])
    else:
        by_device = pd.DataFrame(columns=["device_id", "orders", "users", "refund_amount", "avg_risk"])

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