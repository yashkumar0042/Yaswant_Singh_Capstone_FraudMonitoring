from __future__ import annotations

from typing import List
import pandas as pd

from .risk_config import WEIGHTS, REASON_CODES


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