"""
Evidence visualization renderers for Jupyter notebooks.

Provides functions that load evidence state from a workspace directory
and display styled HTML tables/cards using IPython.display. Designed to
be called from within Jupyter notebook code cells.

Usage (inside a notebook cell):
    from pkevolve.verification.renderers import render_papers
    render_papers("/path/to/workspace")
"""

import json
from pathlib import Path

from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------


def _stance_color(stance: str) -> str:
    """Map an evidence stance to a display color."""
    return {
        "SUPPORT": "#22c55e",
        "REFUTE": "#ef4444",
        "NEUTRAL": "#9ca3af",
    }.get(stance, "#9ca3af")


def _stance_icon(stance: str) -> str:
    """Map an evidence stance to an HTML icon entity."""
    return {
        "SUPPORT": "&#9679;",   # filled circle
        "REFUTE": "&#9679;",    # filled circle
        "NEUTRAL": "&#9675;",   # hollow circle
    }.get(stance, "&#9675;")


def _bar_html(
    value: float,
    max_val: float = 1.0,
    color: str = "#3b82f6",
    width: int = 200,
) -> str:
    """Generate an inline HTML progress bar."""
    pct = min(value / max_val, 1.0) * 100
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;width:{width}px;'
        f'display:inline-block">'
        f'<div style="background:{color};width:{pct}%;height:16px;'
        f'border-radius:4px"></div>'
        f"</div> {value:.0%}"
    )


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------


def render_papers(workspace: str) -> None:
    """Render retrieved papers as a styled HTML table.

    Reads ``evidence_state.json`` from *workspace* and displays an HTML
    table with PMID (linked), title, abstract preview, and full-text flag.
    """
    ws = Path(workspace)
    state = json.loads((ws / "evidence_state.json").read_text())
    papers = state.get("papers", {})

    if not papers:
        display(HTML("<p><em>No papers retrieved yet.</em></p>"))
        return

    rows = []
    for pmid, p in papers.items():
        title = p.get("title", "")[:80]
        abstract_preview = (p.get("abstract", "") or "")[:100]
        has_ft = "Yes" if p.get("full_text") else "No"
        rows.append(
            f"<tr>"
            f"<td><a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/' "
            f"target='_blank'>{pmid}</a></td>"
            f"<td>{title}</td>"
            f"<td>{abstract_preview}...</td>"
            f"<td>{has_ft}</td>"
            f"</tr>"
        )

    html = (
        "<h3>Retrieved Papers</h3>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f1f5f9'>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>PMID</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Title</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Abstract</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Full Text</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
        + f"<p><em>Total: {len(papers)} papers</em></p>"
    )
    display(HTML(html))


def render_facts(workspace: str) -> None:
    """Render extracted facts as a color-coded HTML table.

    Green = SUPPORT, red = REFUTE, gray = NEUTRAL. Reads from
    ``evidence_state.json`` in *workspace*.
    """
    ws = Path(workspace)
    state = json.loads((ws / "evidence_state.json").read_text())
    facts = state.get("facts", [])

    if not facts:
        display(HTML("<p><em>No facts extracted yet.</em></p>"))
        return

    rows = []
    for f in facts:
        stance = f.get("stance", "NEUTRAL")
        color = _stance_color(stance)
        icon = _stance_icon(stance)
        text = f.get("text", "")[:120]
        pmid = f.get("source_pmid", "")
        conf = f.get("confidence", 0)
        rows.append(
            f"<tr>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1;"
            f"color:{color};font-weight:bold'>{icon} {stance}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{text}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{pmid}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{conf:.2f}</td>"
            f"</tr>"
        )

    n_sup = sum(1 for f in facts if f.get("stance") == "SUPPORT")
    n_ref = sum(1 for f in facts if f.get("stance") == "REFUTE")
    n_neu = sum(1 for f in facts if f.get("stance") == "NEUTRAL")

    html = (
        "<h3>Extracted Facts</h3>"
        f"<p>Total: {len(facts)} &mdash; "
        f"<span style='color:#22c55e'>&#9679; Support: {n_sup}</span> &nbsp; "
        f"<span style='color:#ef4444'>&#9679; Refute: {n_ref}</span> &nbsp; "
        f"<span style='color:#9ca3af'>&#9675; Neutral: {n_neu}</span></p>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f1f5f9'>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Stance</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Fact</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>PMID</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Conf</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )
    display(HTML(html))


def render_sufficiency(workspace: str) -> None:
    """Render the latest sufficiency check with progress bars.

    Shows overall confidence, per-subclaim coverage bars, and identified
    gaps. Reads from ``evidence_state.json`` in *workspace*.
    """
    ws = Path(workspace)
    state = json.loads((ws / "evidence_state.json").read_text())
    hist = state.get("sufficiency_history", [])
    iteration = state.get("iteration", 0)

    if not hist:
        display(HTML("<p><em>No sufficiency checks performed yet.</em></p>"))
        return

    latest = hist[-1]
    label = latest.get("label", "INSUFFICIENT")
    conf = latest.get("confidence", 0)
    gaps = latest.get("gaps", [])

    # Status color
    if label != "INSUFFICIENT" and conf >= 0.80:
        status_color = "#22c55e"
        status_text = "SUFFICIENT"
    else:
        status_color = "#f59e0b"
        status_text = "INSUFFICIENT"

    conf_bar = _bar_html(conf, 1.0, status_color, 300)

    # Coverage bars per subclaim — disabled for now; subclaim decomposition
    # is not yet actively used, so the coverage dict is always 0.
    # coverage = state.get("coverage", {})
    # subclaims = state.get("subclaims", [])
    # cov_rows = ""
    # for sc in subclaims:
    #     cov = coverage.get(sc, 0)
    #     cov_bar = _bar_html(cov, 1.0, "#3b82f6", 200)
    #     sc_short = sc[:60] + "..." if len(sc) > 60 else sc
    #     cov_rows += (
    #         f"<tr><td style='padding:4px'>{sc_short}</td>"
    #         f"<td>{cov_bar}</td></tr>"
    #     )
    cov_rows = ""

    # Gaps list
    gap_html = ""
    if gaps:
        gap_items = ""
        for g in gaps:
            gt = g.get("gap_type", "unknown")
            if isinstance(gt, dict):
                gt = gt.get("value", str(gt))
            desc = g.get("description", "")
            pri = g.get("priority", 0)
            # Handle priority as either string or number
            pri_str = f"{pri:.1f}" if isinstance(pri, (int, float)) else str(pri)
            gap_items += (
                f"<li><span style='color:#f59e0b'>&#9888;</span> "
                f"<strong>{gt}</strong>: {desc} "
                f"<em>(priority: {pri_str})</em></li>"
            )
        gap_html = (
            f"<h4>Gaps Identified ({len(gaps)})</h4><ul>{gap_items}</ul>"
        )

    html = (
        f"<div style='border:2px solid {status_color};border-radius:8px;"
        f"padding:16px;margin:8px 0'>"
        f"<h3>Sufficiency Check (Iteration {iteration}/8)</h3>"
        f"<p><strong>Label:</strong> {label} &nbsp; "
        f"<strong>Status:</strong> "
        f"<span style='color:{status_color};font-weight:bold'>"
        f"{status_text}</span></p>"
        f"<p><strong>Confidence:</strong> {conf_bar}</p>"
        f"<table style='margin:8px 0'>{cov_rows}</table>"
        f"{gap_html}"
        f"</div>"
    )
    display(HTML(html))


def render_papers_from_state(state) -> None:
    """Render papers from an in-memory EvidenceState object.

    Accepts either an ``EvidenceState`` instance or a plain dict
    (from ``state.model_dump()``).
    """
    if hasattr(state, "model_dump"):
        d = state.model_dump()
    else:
        d = state
    papers = d.get("papers", {})

    if not papers:
        display(HTML("<p><em>No papers retrieved yet.</em></p>"))
        return

    rows = []
    for pmid, p in papers.items():
        title = p.get("title", "")[:80]
        abstract_preview = (p.get("abstract", "") or "")[:100]
        has_ft = "Yes" if p.get("full_text") else "No"
        rows.append(
            f"<tr>"
            f"<td><a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/' "
            f"target='_blank'>{pmid}</a></td>"
            f"<td>{title}</td>"
            f"<td>{abstract_preview}...</td>"
            f"<td>{has_ft}</td>"
            f"</tr>"
        )

    html = (
        "<h3>Retrieved Papers</h3>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f1f5f9'>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>PMID</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Title</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Abstract</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Full Text</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
        + f"<p><em>Total: {len(papers)} papers</em></p>"
    )
    display(HTML(html))


def render_facts_from_state(state) -> None:
    """Render facts from an in-memory EvidenceState object."""
    if hasattr(state, "model_dump"):
        d = state.model_dump()
    else:
        d = state
    facts = d.get("facts", [])

    if not facts:
        display(HTML("<p><em>No facts extracted yet.</em></p>"))
        return

    rows = []
    for f in facts:
        stance = f.get("stance", "NEUTRAL")
        if isinstance(stance, str):
            pass
        else:
            stance = str(stance)
        color = _stance_color(stance)
        icon = _stance_icon(stance)
        text = f.get("text", "")[:120]
        pmid = f.get("source_pmid", "")
        conf = f.get("confidence", 0)
        rows.append(
            f"<tr>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1;"
            f"color:{color};font-weight:bold'>{icon} {stance}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{text}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{pmid}</td>"
            f"<td style='padding:4px 8px;border:1px solid #cbd5e1'>{conf:.2f}</td>"
            f"</tr>"
        )

    n_sup = sum(1 for f in facts if str(f.get("stance", "")).upper() == "SUPPORT")
    n_ref = sum(1 for f in facts if str(f.get("stance", "")).upper() == "REFUTE")
    n_neu = sum(1 for f in facts if str(f.get("stance", "")).upper() == "NEUTRAL")

    html = (
        "<h3>Extracted Facts</h3>"
        f"<p>Total: {len(facts)} &mdash; "
        f"<span style='color:#22c55e'>&#9679; Support: {n_sup}</span> &nbsp; "
        f"<span style='color:#ef4444'>&#9679; Refute: {n_ref}</span> &nbsp; "
        f"<span style='color:#9ca3af'>&#9675; Neutral: {n_neu}</span></p>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f1f5f9'>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Stance</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Fact</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>PMID</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1;text-align:left'>Conf</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )
    display(HTML(html))


def render_sufficiency_from_state(state) -> None:
    """Render sufficiency check from an in-memory EvidenceState object."""
    if hasattr(state, "model_dump"):
        d = state.model_dump()
    else:
        d = state
    hist = d.get("sufficiency_history", [])
    iteration = d.get("iteration", 0)

    if not hist:
        display(HTML("<p><em>No sufficiency checks performed yet.</em></p>"))
        return

    latest = hist[-1]
    label = latest.get("label", "INSUFFICIENT")
    conf = latest.get("confidence", 0)
    gaps = latest.get("gaps", [])

    if label != "INSUFFICIENT" and conf >= 0.80:
        status_color = "#22c55e"
        status_text = "SUFFICIENT"
    else:
        status_color = "#f59e0b"
        status_text = "INSUFFICIENT"

    conf_bar = _bar_html(conf, 1.0, status_color, 300)
    cov_rows = ""

    gap_html = ""
    if gaps:
        gap_items = ""
        for g in gaps:
            gt = g.get("gap_type", "unknown")
            if isinstance(gt, dict):
                gt = gt.get("value", str(gt))
            desc = g.get("description", "")
            pri = g.get("priority", 0)
            # Handle priority as either string or number
            pri_str = f"{pri:.1f}" if isinstance(pri, (int, float)) else str(pri)
            gap_items += (
                f"<li><span style='color:#f59e0b'>&#9888;</span> "
                f"<strong>{gt}</strong>: {desc} "
                f"<em>(priority: {pri_str})</em></li>"
            )
        gap_html = (
            f"<h4>Gaps Identified ({len(gaps)})</h4><ul>{gap_items}</ul>"
        )

    html = (
        f"<div style='border:2px solid {status_color};border-radius:8px;"
        f"padding:16px;margin:8px 0'>"
        f"<h3>Sufficiency Check (Iteration {iteration}/8)</h3>"
        f"<p><strong>Label:</strong> {label} &nbsp; "
        f"<strong>Status:</strong> "
        f"<span style='color:{status_color};font-weight:bold'>"
        f"{status_text}</span></p>"
        f"<p><strong>Confidence:</strong> {conf_bar}</p>"
        f"<table style='margin:8px 0'>{cov_rows}</table>"
        f"{gap_html}"
        f"</div>"
    )
    display(HTML(html))


def render_verdict(workspace: str) -> None:
    """Render the final verdict as a styled card.

    Reads ``verdict.json`` from *workspace* and displays a card with the
    verdict, confidence, reasoning, key evidence, and remaining gaps.
    """
    ws = Path(workspace)
    verdict_path = ws / "verdict.json"

    if not verdict_path.exists():
        display(HTML("<p><em>No verdict emitted yet.</em></p>"))
        return

    v = json.loads(verdict_path.read_text())
    verdict = v.get("verdict", "INSUFFICIENT")
    conf = v.get("confidence", 0)
    reasoning = v.get("reasoning", "")
    key_ev = v.get("key_evidence", [])
    gaps = v.get("gaps_remaining", [])

    vcolors = {
        "SUPPORT": "#22c55e",
        "REFUTE": "#ef4444",
        "INSUFFICIENT": "#f59e0b",
    }
    vc = vcolors.get(verdict, "#9ca3af")

    ev_items = (
        "".join(f"<li>{e}</li>" for e in key_ev)
        if key_ev
        else "<li><em>None</em></li>"
    )
    gap_items = (
        "".join(f"<li>{g}</li>" for g in gaps)
        if gaps
        else "<li><em>None</em></li>"
    )

    html = (
        f"<div style='border:3px solid {vc};border-radius:12px;padding:20px;"
        f"margin:16px 0;background:linear-gradient(135deg,#f8fafc,#f1f5f9)'>"
        f"<h2 style='margin:0 0 12px 0'>Final Verdict</h2>"
        f"<div style='font-size:2em;font-weight:bold;color:{vc};"
        f"margin:8px 0'>{verdict}</div>"
        f"<p><strong>Confidence:</strong> {conf:.0%}</p>"
        f"<p><strong>Reasoning:</strong> {reasoning}</p>"
        f"<h4>Key Evidence</h4><ul>{ev_items}</ul>"
        f"<h4>Remaining Gaps</h4><ul>{gap_items}</ul>"
        f"</div>"
    )
    display(HTML(html))
