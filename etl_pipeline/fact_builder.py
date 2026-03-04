from __future__ import annotations

import pandas as pd

from .cleaning import standardize_df, parse_dt, dedup_keep_last, safe_lower
from .features import discount_pct, zscore_within_group


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