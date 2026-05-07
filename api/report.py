"""
Axion AI — HTML Report Generator
==================================

Generates a self-contained, print-ready HTML process report from the
in-memory state of a scenario run.

All public functions accept plain Python dicts/lists so they can be tested
without importing domain models. The server endpoint is responsible for
converting domain objects before calling these functions.

Report sections
---------------
1. Executive summary tiles (4 key numbers)
2. Process KPI table (mean / min / max / std, spec highlights)
3. Recommendations summary (by urgency + by rule)
4. Operator decisions log
5. Analytics event sessions
6. Per-rule performance table
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from profile import ProcessProfile, active_profile


# ─────────────────────────────────────────────────────────────────────────────
# KPI definitions are read from the active profile
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_defs(profile: Optional[ProcessProfile] = None) \
        -> List[Tuple[str, str, str, Optional[float], Optional[float]]]:
    """Profile-aware KPI definitions list.

    Returns tuples (col, label, unit, spec_min, spec_max). Falls back to the
    active env-resolved profile when none is passed.
    """
    p = profile or active_profile()
    return [
        (t.tag, t.label, t.units, t.spec_min, t.spec_max)
        for t in p.kpi_tags
    ]

_URGENCY_ORDER = ["critical", "high", "medium", "low"]


# ─────────────────────────────────────────────────────────────────────────────
# Data summarization functions (pure — no I/O, no domain imports)
# ─────────────────────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame,
                profile: Optional[ProcessProfile] = None) -> Dict[str, Any]:
    """Compute KPI statistics from a process DataFrame.

    Returns
    -------
    dict with keys:
      rows         — list of per-variable dicts (label, unit, mean, min, max, std, spec_violated)
      duration_h   — total run duration in hours
      n_samples    — row count
      purity_below_spec_pct — % of samples below the active profile's purity spec
    """
    p = profile or active_profile()
    rows = []
    for col, label, unit, spec_min, spec_max in _kpi_defs(p):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue

        mean_v = float(s.mean())
        min_v  = float(s.min())
        max_v  = float(s.max())
        std_v  = float(s.std())

        spec_violated = False
        if spec_min is not None and min_v < spec_min:
            spec_violated = True
        if spec_max is not None and max_v > spec_max:
            spec_violated = True

        rows.append({
            "col":           col,
            "label":         label,
            "unit":          unit,
            "mean":          mean_v,
            "min":           min_v,
            "max":           max_v,
            "std":           std_v,
            "spec_min":      spec_min,
            "spec_max":      spec_max,
            "spec_violated": spec_violated,
        })

    # Duration
    duration_h = 0.0
    if "timestamp" in df.columns and len(df) > 1:
        ts = pd.to_datetime(df["timestamp"])
        duration_h = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0

    # Profile-defined "below spec" KPI
    purity_below = 0.0
    if p.purity_kpi and p.purity_kpi in df.columns:
        s = df[p.purity_kpi].dropna()
        if len(s) > 0:
            purity_below = float((s < p.purity_spec_min).sum() / len(s) * 100.0)

    return {
        "rows":                    rows,
        "duration_h":              duration_h,
        "n_samples":               len(df),
        "purity_below_spec_pct":   purity_below,
    }


def recommendations_summary(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize a list of recommendation dicts.

    Expected keys per dict: urgency, rule_fired, status, timestamp, diagnosis.

    Returns
    -------
    dict with:
      total        — total count
      by_urgency   — Counter keyed by urgency value
      by_rule      — Counter keyed by rule_fired
      n_decided    — recs that have a non-pending status
      n_accepted   — recs with status "accepted"
      n_rejected   — recs with status "rejected"
      acceptance_rate — n_accepted / n_decided (None if no decisions yet)
    """
    total      = len(recs)
    by_urgency = Counter(r.get("urgency", "unknown") for r in recs)
    by_rule    = Counter(r.get("rule_fired", "unknown") for r in recs)

    decided  = [r for r in recs if r.get("status", "pending") != "pending"]
    accepted = [r for r in recs if r.get("status") == "accepted"]
    rejected = [r for r in recs if r.get("status") == "rejected"]

    acceptance_rate: Optional[float] = None
    if decided:
        acceptance_rate = len(accepted) / len(decided)

    return {
        "total":           total,
        "by_urgency":      dict(by_urgency),
        "by_rule":         dict(by_rule),
        "n_decided":       len(decided),
        "n_accepted":      len(accepted),
        "n_rejected":      len(rejected),
        "acceptance_rate": acceptance_rate,
    }


def decisions_summary(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize operator decision dicts.

    Expected keys: status, justification, timestamp, urgency, rule_id.

    Returns
    -------
    dict with:
      total, by_status (Counter), log (list sorted by timestamp desc)
    """
    by_status = Counter(d.get("status", "unknown") for d in decisions)
    log = sorted(decisions, key=lambda d: d.get("timestamp", ""), reverse=True)
    return {
        "total":     len(decisions),
        "by_status": dict(by_status),
        "log":       log,
    }


def sessions_summary(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize analytics event session dicts.

    Expected keys: detector, tag, duration_min, peak_severity.

    Returns
    -------
    dict with:
      total, by_severity (Counter), by_detector (Counter), longest (top-5 by duration)
    """
    total       = len(sessions)
    by_severity = Counter(s.get("peak_severity", "unknown") for s in sessions)
    by_detector = Counter(s.get("detector", "unknown") for s in sessions)
    longest     = sorted(sessions,
                         key=lambda s: s.get("duration_min", 0),
                         reverse=True)[:5]
    return {
        "total":       total,
        "by_severity": dict(by_severity),
        "by_detector": dict(by_detector),
        "longest":     longest,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{text.upper()}</span>'


def _urgency_badge(u: str) -> str:
    cls = u if u in ("critical", "high", "medium", "low") else "medium"
    return _badge(u, cls)


def _decision_badge(s: str) -> str:
    cls = s if s in ("accepted", "rejected", "modified") else "medium"
    return _badge(s, cls)


def render_html(
    scenario: str,
    generated_at: str,
    kpi: Dict[str, Any],
    rec_summary: Dict[str, Any],
    dec_summary: Dict[str, Any],
    sess_summary: Dict[str, Any],
    perf_rows: List[Dict[str, Any]],
    rec_log: List[Dict[str, Any]],
    profile: Optional[ProcessProfile] = None,
) -> str:
    """Render the full self-contained HTML report string."""
    p = profile or active_profile()

    # ── Executive tiles ──────────────────────────────────────────────────────
    purity_col  = p.purity_kpi or ""
    purity_rows = [r for r in kpi["rows"] if r["col"] == purity_col]
    avg_purity  = _fmt(purity_rows[0]["mean"]) if purity_rows else "—"
    purity_label = p.tag(purity_col).label if p.tag(purity_col) else "Avg. Product Purity"
    purity_unit  = p.tag(purity_col).units if p.tag(purity_col) else "%"
    acceptance  = (
        f"{rec_summary['acceptance_rate'] * 100:.0f}%"
        if rec_summary["acceptance_rate"] is not None else "—"
    )
    peak_urgency = "—"
    for u in _URGENCY_ORDER:
        if rec_summary["by_urgency"].get(u, 0) > 0:
            peak_urgency = u.upper()
            break

    tiles_html = f"""
    <div class="tile-grid">
      <div class="tile">
        <div class="tile-value">{avg_purity}<span class="tile-unit">{purity_unit}</span></div>
        <div class="tile-label">Avg. {purity_label}</div>
        <div class="tile-sub">{kpi['purity_below_spec_pct']:.1f}% samples below spec</div>
      </div>
      <div class="tile">
        <div class="tile-value">{rec_summary['total']}</div>
        <div class="tile-label">Recommendations Issued</div>
        <div class="tile-sub">Peak urgency: {peak_urgency}</div>
      </div>
      <div class="tile">
        <div class="tile-value">{dec_summary['total']}</div>
        <div class="tile-label">Operator Decisions</div>
        <div class="tile-sub">{dec_summary['by_status'].get('accepted', 0)} accepted · {dec_summary['by_status'].get('rejected', 0)} rejected</div>
      </div>
      <div class="tile">
        <div class="tile-value">{acceptance}</div>
        <div class="tile-label">Acceptance Rate</div>
        <div class="tile-sub">{rec_summary['n_decided']} of {rec_summary['total']} decided</div>
      </div>
    </div>"""

    # ── KPI table ─────────────────────────────────────────────────────────────
    kpi_rows_html = ""
    for r in kpi["rows"]:
        viol_cls = ' class="spec-viol"' if r["spec_violated"] else ""
        spec_str = ""
        if r["spec_min"] is not None:
            spec_str += f"min ≥ {r['spec_min']}"
        if r["spec_max"] is not None:
            spec_str += f" max ≤ {r['spec_max']}"
        kpi_rows_html += f"""
        <tr>
          <td>{r['label']}</td>
          <td class="mono">{r['unit']}</td>
          <td{viol_cls} class="mono">{_fmt(r['mean'])}</td>
          <td{viol_cls} class="mono">{_fmt(r['min'])}</td>
          <td{viol_cls} class="mono">{_fmt(r['max'])}</td>
          <td class="mono dim">{_fmt(r['std'])}</td>
          <td class="dim">{spec_str.strip()}</td>
        </tr>"""

    kpi_table = f"""
    <table>
      <thead><tr>
        <th>Variable</th><th>Unit</th>
        <th>Mean</th><th>Min</th><th>Max</th><th>Std</th><th>Spec</th>
      </tr></thead>
      <tbody>{kpi_rows_html}</tbody>
    </table>"""

    # ── Recommendations by urgency ────────────────────────────────────────────
    urg_html = ""
    for u in _URGENCY_ORDER:
        n = rec_summary["by_urgency"].get(u, 0)
        if n == 0:
            continue
        urg_html += f"<span>{_urgency_badge(u)} {n}</span> "

    # Recommendations by rule
    rule_rows_html = ""
    for rule, cnt in sorted(rec_summary["by_rule"].items(),
                             key=lambda x: -x[1]):
        rule_rows_html += f"<tr><td>{rule}</td><td class='mono'>{cnt}</td></tr>"

    recs_section = f"""
    <div class="two-col">
      <div>
        <p class="sub-label">By Urgency</p>
        <div class="badge-row">{urg_html or '<span class="dim">None</span>'}</div>
      </div>
      <div>
        <p class="sub-label">By Rule (top fired)</p>
        <table><tbody>{rule_rows_html or '<tr><td class="dim">No recommendations</td></tr>'}</tbody></table>
      </div>
    </div>"""

    # Recent recommendations log (up to 20, sorted newest first)
    rec_log_rows = ""
    for r in sorted(rec_log, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]:
        ts  = str(r.get("timestamp", ""))[:16].replace("T", " ")
        rec_log_rows += f"""<tr>
          <td class="mono dim">{ts}</td>
          <td>{_urgency_badge(r.get('urgency',''))}</td>
          <td class="dim">{r.get('rule_fired','—')}</td>
          <td>{r.get('diagnosis','—')}</td>
          <td>{_decision_badge(r.get('status','pending'))}</td>
        </tr>"""
    rec_log_table = f"""
    <table>
      <thead><tr><th>Time</th><th>Urgency</th><th>Rule</th><th>Diagnosis</th><th>Status</th></tr></thead>
      <tbody>{rec_log_rows or '<tr><td colspan="5" class="dim">No recommendations</td></tr>'}</tbody>
    </table>"""

    # ── Decisions log ─────────────────────────────────────────────────────────
    dec_rows_html = ""
    for d in dec_summary["log"][:20]:
        ts  = str(d.get("timestamp", ""))[:16].replace("T", " ")
        dec_rows_html += f"""<tr>
          <td class="mono dim">{ts}</td>
          <td>{_decision_badge(d.get('status',''))}</td>
          <td class="dim">{d.get('urgency','—')}</td>
          <td class="dim">{d.get('rule_id','—')}</td>
          <td>{d.get('justification','—')}</td>
        </tr>"""
    decisions_table = f"""
    <table>
      <thead><tr><th>Time</th><th>Decision</th><th>Urgency</th><th>Rule</th><th>Justification</th></tr></thead>
      <tbody>{dec_rows_html or '<tr><td colspan="5" class="dim">No decisions recorded</td></tr>'}</tbody>
    </table>"""

    # ── Analytics sessions ────────────────────────────────────────────────────
    sev_html = " ".join(
        f'<span class="badge badge-{sev.lower()}">{sev.upper()} {n}</span>'
        for sev, n in sorted(sess_summary["by_severity"].items())
    ) or '<span class="dim">No events</span>'

    sess_rows_html = ""
    for s in sess_summary["longest"]:
        ts  = str(s.get("start_time", ""))[:16].replace("T", " ")
        dur = s.get("duration_min", 0)
        sess_rows_html += f"""<tr>
          <td class="mono dim">{ts}</td>
          <td>{s.get('detector','—')}</td>
          <td class="dim">{s.get('tag') or 'multivariate'}</td>
          <td class="mono">{_fmt(dur, 1)} min</td>
          <td>{_urgency_badge(s.get('peak_severity',''))}</td>
        </tr>"""
    sessions_table = f"""
    <p class="sub-label">Severity distribution — {sev_html}</p>
    <table>
      <thead><tr><th>Start</th><th>Detector</th><th>Tag</th><th>Duration</th><th>Peak Sev.</th></tr></thead>
      <tbody>{sess_rows_html or '<tr><td colspan="5" class="dim">No event sessions</td></tr>'}</tbody>
    </table>
    <p class="dim" style="margin-top:6px;font-size:0.78em">Showing longest {min(5, sess_summary['total'])} of {sess_summary['total']} total sessions.</p>"""

    # ── Rule performance ──────────────────────────────────────────────────────
    perf_rows_html = ""
    for row in perf_rows:
        acc = row.get("acceptance_rate")
        acc_str = f"{acc*100:.0f}%" if acc is not None else "—"
        perf_rows_html += f"""<tr>
          <td>{row.get('rule_id','—')}</td>
          <td class="mono">{row.get('n_issued',0)}</td>
          <td class="mono">{row.get('n_accepted',0)}</td>
          <td class="mono">{row.get('n_rejected',0)}</td>
          <td class="mono">{acc_str}</td>
        </tr>"""
    perf_table = f"""
    <table>
      <thead><tr><th>Rule</th><th>Issued</th><th>Accepted</th><th>Rejected</th><th>Acceptance Rate</th></tr></thead>
      <tbody>{perf_rows_html or '<tr><td colspan="5" class="dim">No data</td></tr>'}</tbody>
    </table>"""

    # ── Assemble ──────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Axion AI · Process Report · {scenario}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif; font-size:14px;
         color:#1a2332; background:#fff; }}
  .header {{ background:#07090d; color:#e6edf3; padding:24px 40px; }}
  .header h1 {{ margin:0; font-size:1.25em; letter-spacing:.06em; font-weight:700; }}
  .header .meta {{ color:#8892a4; font-size:.82em; margin-top:6px; }}
  .header .meta strong {{ color:#b8c2cc; }}
  .content {{ padding:28px 40px; max-width:1100px; }}
  .section {{ margin-bottom:28px; }}
  .section-title {{ font-size:.72em; text-transform:uppercase; letter-spacing:.1em;
                    color:#5a6a7a; border-bottom:2px solid #e2e8f0;
                    padding-bottom:5px; margin:0 0 12px 0; }}
  .tile-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:4px; }}
  .tile {{ background:#f7fafc; border:1px solid #e2e8f0; border-radius:4px;
           padding:14px 16px; }}
  .tile-value {{ font-size:2em; font-weight:700; color:#1a2332; line-height:1; }}
  .tile-unit {{ font-size:.5em; color:#5a6a7a; margin-left:2px; }}
  .tile-label {{ font-size:.72em; text-transform:uppercase; letter-spacing:.05em;
                 color:#5a6a7a; margin-top:5px; }}
  .tile-sub {{ font-size:.75em; color:#8892a4; margin-top:2px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.84em; }}
  th {{ background:#f7fafc; text-align:left; padding:7px 10px; font-size:.75em;
        text-transform:uppercase; letter-spacing:.06em; color:#5a6a7a;
        border-bottom:2px solid #e2e8f0; }}
  td {{ padding:6px 10px; border-bottom:1px solid #f0f4f8; }}
  tr:hover td {{ background:#fafcff; }}
  .mono {{ font-family:'Courier New',monospace; }}
  .dim {{ color:#8892a4; }}
  .spec-viol {{ color:#c53030 !important; font-weight:600; }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:3px;
            font-size:.75em; font-weight:600; letter-spacing:.03em; }}
  .badge-critical {{ background:#fff0f0; color:#c53030; }}
  .badge-high {{ background:#fff7e6; color:#c05621; }}
  .badge-medium {{ background:#fffff0; color:#975a16; }}
  .badge-low {{ background:#f0fff4; color:#276749; }}
  .badge-accepted {{ background:#f0fff4; color:#276749; }}
  .badge-rejected {{ background:#fff0f0; color:#c53030; }}
  .badge-modified {{ background:#fff7e6; color:#c05621; }}
  .badge-pending {{ background:#f7fafc; color:#5a6a7a; }}
  .badge-deferred {{ background:#f7fafc; color:#5a6a7a; }}
  .badge-row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  .sub-label {{ font-size:.75em; text-transform:uppercase; letter-spacing:.07em;
                color:#5a6a7a; margin:0 0 8px 0; }}
  .footer {{ margin-top:40px; padding:14px 40px; border-top:1px solid #e2e8f0;
             font-size:.75em; color:#8892a4; text-align:center; }}
  @media print {{
    body {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
    .tile-grid {{ grid-template-columns:repeat(4,1fr); }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>AXION AI · Process Intelligence Report</h1>
  <div class="meta">
    Scenario: <strong>{scenario}</strong> &nbsp;·&nbsp;
    Duration: <strong>{kpi['duration_h']:.1f} h</strong> &nbsp;·&nbsp;
    Samples: <strong>{kpi['n_samples']:,}</strong> &nbsp;·&nbsp;
    Generated: <strong>{generated_at}</strong>
  </div>
</div>

<div class="content">

  <div class="section">
    <h2 class="section-title">Executive Summary</h2>
    {tiles_html}
  </div>

  <div class="section">
    <h2 class="section-title">Process KPI Summary</h2>
    {kpi_table}
  </div>

  <div class="section">
    <h2 class="section-title">Recommendations</h2>
    {recs_section}
  </div>

  <div class="section">
    <h2 class="section-title">Recommendations Log (latest 20)</h2>
    {rec_log_table}
  </div>

  <div class="section">
    <h2 class="section-title">Operator Decisions Log (latest 20)</h2>
    {decisions_table}
  </div>

  <div class="section">
    <h2 class="section-title">Analytics Event Sessions</h2>
    {sessions_table}
  </div>

  <div class="section">
    <h2 class="section-title">Per-Rule Performance</h2>
    {perf_table}
  </div>

</div>

<div class="footer">
  Generated by <strong>Axion AI</strong> · Process Intelligence Platform
</div>
</body>
</html>"""
