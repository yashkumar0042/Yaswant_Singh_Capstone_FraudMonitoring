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