from __future__ import annotations

from .paths import ensure_dirs, RAW_DIR, DATA_DIR, ANALYSIS_DIR, FINAL_STORY_DIR
from .io_utils import read_csv, read_json, write_csv
from .products import flatten_products
from .fact_builder import build_fact_orders_enriched
from .risk_scoring import compute_score_and_reasons
from .aggregations import build_user_weekly, build_queue
from .validations import validate_fact
from .analysis_artifacts import build_analysis_artifacts, export_analysis_report_html
from .dashboard_exports import export_charts, export_dashboard_xlsx
from .final_memo import export_final_memo_pdf


def main() -> None:
    ensure_dirs()
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
        out_path=FINAL_STORY_DIR / "final_memo.pdf",
    )

    print("\n🎉 All done.")
    print("Core CSVs in /data")
    print("Analysis report in /analysis")
    print("Dashboard in /dashboard")
    print("Final memo in /final_story")


if __name__ == "__main__":
    main()