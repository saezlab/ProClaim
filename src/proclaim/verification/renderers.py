"""
Evidence visualization renderers for Jupyter notebooks.

Provides functions that load evidence state from a workspace directory
and display styled HTML tables/cards using IPython.display. Designed to
be called from within Jupyter notebook code cells.

Usage (inside a notebook cell):
    from proclaim.verification.renderers import render_papers
    render_papers("/path/to/workspace")
"""

import json
from pathlib import Path

from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------

# Default color cycle for stance labels (first = positive, second = negative, rest = neutral shades)
_DEFAULT_COLORS = ["#22c55e", "#ef4444", "#9ca3af", "#60a5fa", "#f59e0b", "#a78bfa"]


def _stance_color(stance: str) -> str:
    """Map an evidence stance to a display color.

    Uses the configured label order: first label gets green, second gets red,
    remaining get gray. Falls back to gray for unknown stances.
    """
    from proclaim.verification.config import get_label_config
    label_cfg = get_label_config()
    names = label_cfg.stance_names()
    for i, name in enumerate(names):
        if stance.upper() == name.upper():
            return _DEFAULT_COLORS[i] if i < len(_DEFAULT_COLORS) else "#9ca3af"
    return "#9ca3af"


def _stance_icon(stance: str) -> str:
    """Map an evidence stance to an HTML icon entity.

    First two configured labels get filled circles; rest get hollow circles.
    """
    from proclaim.verification.config import get_label_config
    label_cfg = get_label_config()
    names = label_cfg.stance_names()
    for i, name in enumerate(names):
        if stance.upper() == name.upper():
            return "&#9679;" if i < 2 else "&#9675;"
    return "&#9675;"


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

    from proclaim.verification.config import get_label_config
    _lc = get_label_config()
    stance_spans = " &nbsp; ".join(
        f"<span style='color:{_stance_color(name)}'>{_stance_icon(name)} {name}: "
        f"{sum(1 for f in facts if f.get('stance', '').upper() == name.upper())}</span>"
        for name in _lc.stance_names()
    )

    html = (
        "<h3>Extracted Facts</h3>"
        f"<p>Total: {len(facts)} &mdash; {stance_spans}</p>"
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
    max_iterations = state.get("MAX_ITERATIONS", 8)  # Read from state, default to 8

    if not hist:
        display(HTML("<p><em>No sufficiency checks performed yet.</em></p>"))
        return

    latest = hist[-1]
    label = latest.get("label", "INSUFFICIENT")
    conf = latest.get("confidence", 0)
    gaps = latest.get("gaps", [])

    # Status color
    if label.lower() == "sufficient" and conf >= 0.80:
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
        f"<h3>Sufficiency Check (Iteration {iteration}/{max_iterations})</h3>"
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

    from proclaim.verification.config import get_label_config
    _lc = get_label_config()
    stance_spans = " &nbsp; ".join(
        f"<span style='color:{_stance_color(name)}'>{_stance_icon(name)} {name}: "
        f"{sum(1 for f in facts if str(f.get('stance', '')).upper() == name.upper())}</span>"
        for name in _lc.stance_names()
    )

    html = (
        "<h3>Extracted Facts</h3>"
        f"<p>Total: {len(facts)} &mdash; {stance_spans}</p>"
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
    max_iterations = d.get("MAX_ITERATIONS", 8)  # Read from state, default to 8

    if not hist:
        display(HTML("<p><em>No sufficiency checks performed yet.</em></p>"))
        return

    latest = hist[-1]
    label = latest.get("label", "INSUFFICIENT")
    conf = latest.get("confidence", 0)
    gaps = latest.get("gaps", [])

    if label.lower() == "sufficient" and conf >= 0.80:
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
        f"<h3>Sufficiency Check (Iteration {iteration}/{max_iterations})</h3>"
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

    # Build verdict color map dynamically from configured labels
    from proclaim.verification.config import get_label_config
    _lc = get_label_config()
    vcolors = {}
    verdict_names = _lc.verdict_names()
    verdict_color_cycle = ["#22c55e", "#ef4444", "#f59e0b", "#60a5fa", "#a78bfa"]
    for i, name in enumerate(verdict_names):
        vcolors[name] = verdict_color_cycle[i] if i < len(verdict_color_cycle) else "#9ca3af"
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


def render_filtering_summary(workspace: str) -> None:
    """Render a summary of paper filtering operations from the trace log.

    Reads ``evidence_state.json`` from *workspace* and displays filtering
    operations, showing which papers were removed and why.
    """
    ws = Path(workspace)
    state = json.loads((ws / "evidence_state.json").read_text())
    trace = state.get("trace", [])
    papers = state.get("papers", {})
    facts = state.get("facts", [])

    # Find filtering operations in trace
    filter_ops = [op for op in trace if op.get("operation") == "filter_papers_by_stance"]

    if not filter_ops:
        display(HTML("<p><em>No filtering operations performed yet.</em></p>"))
        return

    # Count facts by paper and stance
    facts_by_paper = {}
    for fact in facts:
        pmid = fact.get("source_pmid")
        if pmid not in facts_by_paper:
            facts_by_paper[pmid] = {"SUPPORT": 0, "REFUTE": 0, "NEUTRAL": 0}
        stance = fact.get("stance", "NEUTRAL")
        facts_by_paper[pmid][stance] = facts_by_paper[pmid].get(stance, 0) + 1

    # Build summary for each filtering operation
    summaries = []
    for i, op in enumerate(filter_ops, 1):
        kept_stances = ", ".join(op.get("keep_stances", []))
        before = op.get("papers_before", 0)
        after = op.get("papers_after", 0)
        removed_count = op.get("papers_removed", 0)
        removed_pmids = op.get("removed_pmids", [])

        summary = (
            f"<div style='border:1px solid #cbd5e1;border-radius:8px;padding:12px;"
            f"margin:12px 0;background:#f8fafc'>"
            f"<h4 style='margin:0 0 8px 0'>Filter Operation {i}</h4>"
            f"<p><strong>Kept stances:</strong> {kept_stances}</p>"
            f"<p><strong>Papers before:</strong> {before} → "
            f"<strong>After:</strong> {after} "
            f"(<span style='color:#ef4444'>-{removed_count}</span>)</p>"
        )

        if removed_pmids:
            summary += "<p><strong>Removed papers:</strong></p><ul>"
            for pmid in removed_pmids[:10]:  # Show first 10
                summary += f"<li>{pmid}</li>"
            if len(removed_pmids) > 10:
                summary += f"<li><em>...and {len(removed_pmids) - 10} more</em></li>"
            summary += "</ul>"

        summary += "</div>"
        summaries.append(summary)

    # Show current paper pool with fact counts
    current_papers = []
    for pmid, paper in papers.items():
        title = paper.get("title", "")[:60]
        fact_counts = facts_by_paper.get(pmid, {"SUPPORT": 0, "REFUTE": 0, "NEUTRAL": 0})
        current_papers.append(
            f"<tr>"
            f"<td><a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/' "
            f"target='_blank'>{pmid}</a></td>"
            f"<td>{title}...</td>"
            f"<td style='color:#22c55e'>{fact_counts['SUPPORT']}</td>"
            f"<td style='color:#ef4444'>{fact_counts['REFUTE']}</td>"
            f"<td style='color:#9ca3af'>{fact_counts['NEUTRAL']}</td>"
            f"</tr>"
        )

    current_table = (
        "<h4>Current Paper Pool ({} papers)</h4>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr style='background:#f1f5f9'>"
        "<th style='padding:6px;border:1px solid #cbd5e1'>PMID</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1'>Title</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1'>SUPPORT</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1'>REFUTE</th>"
        "<th style='padding:6px;border:1px solid #cbd5e1'>NEUTRAL</th>"
        "</tr>"
        "{}"
        "</table>"
    ).format(len(papers), "".join(current_papers))

    html = (
        "<h3>Paper Filtering Summary</h3>"
        + "".join(summaries)
        + current_table
    )

    display(HTML(html))
