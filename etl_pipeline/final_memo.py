from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def export_final_memo_pdf(
    fact: pd.DataFrame,
    kpi_weekly: pd.DataFrame,
    patterns_summary: pd.DataFrame,
    queue: pd.DataFrame,
    out_path: Path
) -> None:
    def wrap_text(c: canvas.Canvas, text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
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

    def ensure_space(c: canvas.Canvas, y: float, needed: float, page_h: float, top_margin: float):
        bottom_margin = 0.75 * inch
        if y - needed < bottom_margin:
            c.showPage()
            return page_h - top_margin
        return y

    def draw_paragraph(c: canvas.Canvas, x: float, y: float, text: str,
                       font_name: str = "Helvetica", font_size: int = 11, line_gap: int = 4,
                       page_w: float = 0, page_h: float = 0, lm: float = 0, rm: float = 0, tm: float = 0) -> float:
        max_width = page_w - lm - rm
        lines = wrap_text(c, text, font_name, font_size, max_width)
        line_height = font_size + line_gap
        y = ensure_space(c, y, line_height * len(lines), page_h, tm)
        c.setFont(font_name, font_size)
        for line in lines:
            c.drawString(x, y, line)
            y -= line_height
        return y

    def draw_bullets(c: canvas.Canvas, x: float, y: float, bullets: List[str],
                     font_name: str = "Helvetica", font_size: int = 11,
                     bullet_indent: float = 12, line_gap: int = 3,
                     page_w: float = 0, page_h: float = 0, lm: float = 0, rm: float = 0, tm: float = 0) -> float:
        max_width = page_w - lm - rm - bullet_indent
        line_height = font_size + line_gap
        c.setFont(font_name, font_size)

        for b in bullets:
            lines = wrap_text(c, b, font_name, font_size, max_width)
            needed = line_height * max(1, len(lines))
            y = ensure_space(c, y, needed + 6, page_h, tm)

            c.drawString(x, y, "•")
            c.drawString(x + bullet_indent, y, lines[0])
            y -= line_height

            for line in lines[1:]:
                c.drawString(x + bullet_indent, y, line)
                y -= line_height

        return y

    total_orders = len(fact)
    total_refund = float(pd.to_numeric(fact.get("refund_amount", 0), errors="coerce").fillna(0).sum())
    total_rto = int(pd.to_numeric(fact.get("rto_flag", 0), errors="coerce").fillna(0).sum())
    high_risk = int((fact.get("risk_band", "") == "High").sum()) if "risk_band" in fact.columns else 0

    top_queue = (
        queue.head(5)[["order_id", "risk_score", "risk_band", "reason_1", "reason_2", "reason_3", "recommended_action"]]
        if not queue.empty else pd.DataFrame()
    )
    top_patterns = patterns_summary.head(5).to_dict("records") if not patterns_summary.empty else []

    c = canvas.Canvas(str(out_path), pagesize=letter)
    page_w, page_h = letter

    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top_margin = 0.75 * inch

    x = left_margin
    y = page_h - top_margin

    y = draw_paragraph(c, x, y, "Final Memo — Fraud Monitoring & Investigation Dashboard",
                       font_name="Helvetica-Bold", font_size=16, line_gap=6,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 6

    y = draw_paragraph(c, x, y, "Generated automatically from ETL outputs (data, analysis, dashboard, and queue).",
                       font_name="Helvetica", font_size=10, line_gap=4,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 10

    y = draw_paragraph(c, x, y, "Executive Summary", font_name="Helvetica-Bold", font_size=12, line_gap=5,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 2

    bullets = [
        f"Total orders analyzed: {total_orders}",
        f"Total refund amount (proxy loss): {total_refund:,.2f}",
        f"Total RTO count (proxy risk): {total_rto}",
        f"High-risk orders (risk_band=High): {high_risk}",
        "Risk scoring is explainable (0–100) with top 3 reasons captured per order.",
        "Investigation queue ranks suspicious orders and recommends an action (manual review / OTP / call verification).",
    ]
    y = draw_bullets(c, x, y, bullets, font_size=11,
                     page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 10

    y = draw_paragraph(c, x, y, "Top Patterns (snapshot)", font_name="Helvetica-Bold", font_size=12, line_gap=5,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 2

    if not top_patterns:
        y = draw_paragraph(c, x, y, "No patterns available (insufficient data).", font_size=11,
                           page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    else:
        pat_lines = []
        for p in top_patterns:
            ptype = p.get("pattern_type", "pattern")
            pat_lines.append(f"[{ptype}] " + ", ".join([f"{k}={p[k]}" for k in list(p.keys())[:6] if k in p]))
        y = draw_bullets(c, x, y, pat_lines, font_size=10,
                         page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 10

    y = draw_paragraph(c, x, y, "Sample Investigation Queue (Top 5)", font_name="Helvetica-Bold", font_size=12, line_gap=5,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 2

    if top_queue.empty:
        y = draw_paragraph(c, x, y, "Queue not available (empty).", font_size=11,
                           page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    else:
        q_bullets = []
        for _, r in top_queue.iterrows():
            q_bullets.append(
                f"order_id={r['order_id']}, score={r['risk_score']}, band={r['risk_band']}, "
                f"reasons=({r['reason_1']}, {r['reason_2']}, {r['reason_3']}), action={r['recommended_action']}"
            )
        y = draw_bullets(c, x, y, q_bullets, font_size=10,
                         page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 10

    y = draw_paragraph(c, x, y, "Next Actions (recommended)", font_name="Helvetica-Bold", font_size=12, line_gap=5,
                       page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)
    y -= 2
    actions = [
        "Apply manual review / OTP friction for High-risk orders.",
        "Limit coupon usage by device/pincode for repeated abuse patterns.",
        "Track weekly KPIs (refund_amount, refund_rate, rto_rate, flagged orders) as guardrails after controls.",
    ]
    y = draw_bullets(c, x, y, actions, font_size=11,
                     page_w=page_w, page_h=page_h, lm=left_margin, rm=right_margin, tm=top_margin)

    c.save()
    print(f"✅ Wrote: {out_path}")