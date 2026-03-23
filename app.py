from pathlib import Path
import sys

import csv
import io
import json
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request, send_file

sys.path.append(str(Path(__file__).resolve().parent))
from predictor import Predictor, WQI_RANGES, HHI_RANGES
from groq_service import GroqResearchService


app = Flask(__name__, template_folder="templates", static_folder="static")
predictor = Predictor(Path(__file__).with_name("config.yaml"))
groq_service = GroqResearchService(
    Path(__file__).with_name("config.yaml"),
    Path(__file__).with_name("outputs") / "eval_metrics.json",
)

LOG_PATH = Path(__file__).with_name("outputs") / "ui_events.csv"


def _label_from_ranges(value: float, ranges: dict) -> str:
    for label, (lo, hi) in ranges.items():
        if lo <= value <= hi:
            return label
    keys = list(ranges.keys())
    return keys[-1] if keys else "Unknown"


def _value_range(value: float, label: str, ranges: dict):
    lo, hi = ranges[label]
    width = (hi - lo) * 0.2
    return [round(max(lo, value - width), 2), round(min(hi, value + width), 2)]


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _risk_index(wqi_value: float, hhi_value: float) -> float:
    wqi_norm = min(1.0, max(0.0, _to_float(wqi_value) / 120.0))
    hhi_norm = min(1.0, max(0.0, _to_float(hhi_value) / 3.0))
    # Give health hazard slightly higher weight for management prioritization.
    score = (0.45 * wqi_norm + 0.55 * hhi_norm) * 100.0
    return round(score, 2)


def _simulate_scenario(base_result: dict, scenario_cfg: dict) -> dict:
    rain = _to_float(scenario_cfg.get("rainfall_change_pct", 0.0), 0.0)
    temp = _to_float(scenario_cfg.get("temperature_change_c", 0.0), 0.0)
    agri = _to_float(scenario_cfg.get("agri_change_pct", 0.0), 0.0)
    pop = _to_float(scenario_cfg.get("population_change_pct", 0.0), 0.0)

    base_wqi = _to_float(base_result["wqi"]["value"], 0.0)
    base_hhi = _to_float(base_result["hhi"]["value"], 0.0)

    # Deterministic scenario shift for dashboard-level what-if analysis.
    wqi_new = base_wqi + (-0.09 * rain) + (2.2 * temp) + (0.06 * agri) + (0.05 * pop)
    hhi_new = base_hhi + (-0.004 * rain) + (0.04 * temp) + (0.0025 * agri) + (0.003 * pop)

    wqi_new = max(0.0, min(120.0, wqi_new))
    hhi_new = max(0.0, min(3.0, hhi_new))

    wqi_label = _label_from_ranges(wqi_new, WQI_RANGES)
    hhi_label = _label_from_ranges(hhi_new, HHI_RANGES)

    return {
        "inputs": {
            "rainfall_change_pct": rain,
            "temperature_change_c": temp,
            "agri_change_pct": agri,
            "population_change_pct": pop,
        },
        "wqi": {
            "value": round(wqi_new, 2),
            "class": wqi_label,
            "range": _value_range(wqi_new, wqi_label, WQI_RANGES),
        },
        "hhi": {
            "value": round(hhi_new, 2),
            "class": hhi_label,
            "range": _value_range(hhi_new, hhi_label, HHI_RANGES),
        },
        "risk_index": _risk_index(wqi_new, hhi_new),
    }


def _build_management_plan(base: dict, scenario: dict, compare_obj: dict) -> dict:
    base_wqi = _to_float((base or {}).get("wqi", {}).get("value"), 0.0)
    base_hhi = _to_float((base or {}).get("hhi", {}).get("value"), 0.0)
    scenario_wqi = _to_float((scenario or {}).get("wqi", {}).get("value"), base_wqi)
    scenario_hhi = _to_float((scenario or {}).get("hhi", {}).get("value"), base_hhi)

    delta_wqi = round(scenario_wqi - base_wqi, 2)
    delta_hhi = round(scenario_hhi - base_hhi, 2)

    risk = _risk_index(base_wqi, base_hhi)

    if risk >= 72:
        severity = "Critical"
    elif risk >= 52:
        severity = "High"
    elif risk >= 32:
        severity = "Moderate"
    else:
        severity = "Low"

    strategy_templates = [
        {
            "id": "S1",
            "action": "Deploy village-level treatment and source blending",
            "objective": "Immediate risk suppression in hotspots",
            "timeline": "0-6 months",
            "owner": "TWAD + District Administration",
            "cost_band": "High",
            "base_priority": 95,
            "kpis": ["Hotspot WQI reduction", "Safe source coverage", "Complaint count"],
        },
        {
            "id": "S2",
            "action": "Groundwater recharge and managed aquifer interventions",
            "objective": "Stabilize seasonal stress and dilution capacity",
            "timeline": "6-24 months",
            "owner": "PWD + Rural Development",
            "cost_band": "Medium",
            "base_priority": 82,
            "kpis": ["Recharge volume", "Post-monsoon water level", "Nitrate trend"],
        },
        {
            "id": "S3",
            "action": "Fertilizer and industrial discharge compliance program",
            "objective": "Reduce contaminant load at source",
            "timeline": "0-18 months",
            "owner": "Agriculture + TNPCB",
            "cost_band": "Medium",
            "base_priority": 88,
            "kpis": ["NO3 exceedance rate", "Compliance score", "Inspection closure rate"],
        },
        {
            "id": "S4",
            "action": "Health surveillance and targeted vulnerable-population outreach",
            "objective": "Reduce near-term health burden",
            "timeline": "0-12 months",
            "owner": "Health Department",
            "cost_band": "Low",
            "base_priority": 76,
            "kpis": ["Screening coverage", "HHI reduction in high-risk blocks", "Referral completion"],
        },
        {
            "id": "S5",
            "action": "District digital monitoring and adaptive policy loop",
            "objective": "Improve forecast-to-action governance",
            "timeline": "3-12 months",
            "owner": "District e-Governance Cell",
            "cost_band": "Low",
            "base_priority": 70,
            "kpis": ["Data latency", "Decision cycle time", "Action closure SLA"],
        },
    ]

    strategies = []
    for tpl in strategy_templates:
        priority_score = tpl["base_priority"] + (risk - 50) * 0.35 + (-delta_wqi * 0.2) + (delta_hhi * 6.0)
        priority_score = max(10.0, min(99.0, round(priority_score, 1)))

        if priority_score >= 85:
            priority = "Critical"
        elif priority_score >= 70:
            priority = "High"
        elif priority_score >= 50:
            priority = "Medium"
        else:
            priority = "Low"

        exp_wqi_gain = max(0.5, round((priority_score / 100.0) * 12.0, 2))
        exp_hhi_cut = max(0.01, round((priority_score / 100.0) * 0.45, 3))

        strategies.append(
            {
                "id": tpl["id"],
                "priority": priority,
                "priority_score": priority_score,
                "action": tpl["action"],
                "objective": tpl["objective"],
                "timeline": tpl["timeline"],
                "owner": tpl["owner"],
                "cost_band": tpl["cost_band"],
                "expected_wqi_gain": exp_wqi_gain,
                "expected_hhi_reduction": exp_hhi_cut,
                "kpis": tpl["kpis"],
            }
        )

    strategies.sort(key=lambda s: s["priority_score"], reverse=True)

    return {
        "severity": severity,
        "base_risk_index": risk,
        "scenario_delta": {
            "wqi_delta": delta_wqi,
            "hhi_delta": delta_hhi,
        },
        "strategies": strategies,
    }


def _build_analytics_pack(base: dict, scenario: dict, compare_obj: dict) -> dict:
    base_wqi = _to_float((base or {}).get("wqi", {}).get("value"), 0.0)
    base_hhi = _to_float((base or {}).get("hhi", {}).get("value"), 0.0)
    scenario_wqi = _to_float((scenario or {}).get("wqi", {}).get("value"), base_wqi)
    scenario_hhi = _to_float((scenario or {}).get("hhi", {}).get("value"), base_hhi)

    compare_result = (compare_obj or {}).get("result", {}) if isinstance(compare_obj, dict) else {}
    compare_wqi = _to_float((compare_result or {}).get("wqi", {}).get("value"), base_wqi)
    compare_hhi = _to_float((compare_result or {}).get("hhi", {}).get("value"), base_hhi)

    base_risk = _risk_index(base_wqi, base_hhi)
    scenario_risk = _risk_index(scenario_wqi, scenario_hhi)
    compare_risk = _risk_index(compare_wqi, compare_hhi)

    trend = (base or {}).get("trend", []) if isinstance(base, dict) else []
    horizon = []
    if trend:
        # Apply part of scenario shift gradually across horizon to mimic adaptation trajectory.
        dw = scenario_wqi - base_wqi
        dh = scenario_hhi - base_hhi
        n = max(1, len(trend) - 1)
        for i, item in enumerate(trend):
            year = item.get("year")
            bw = _to_float(item.get("wqi_value"), base_wqi)
            bh = _to_float(item.get("hhi_value"), base_hhi)
            factor = i / n
            sw = max(0.0, min(120.0, bw + dw * factor))
            sh = max(0.0, min(3.0, bh + dh * factor))
            horizon.append(
                {
                    "year": year,
                    "baseline_risk": _risk_index(bw, bh),
                    "scenario_risk": _risk_index(sw, sh),
                    "baseline_wqi": round(bw, 2),
                    "scenario_wqi": round(sw, 2),
                    "baseline_hhi": round(bh, 2),
                    "scenario_hhi": round(sh, 2),
                }
            )

    risk_matrix = [
        {
            "dimension": "Water Quality",
            "likelihood": round(min(5.0, max(1.0, base_wqi / 24.0)), 2),
            "impact": round(min(5.0, max(1.0, base_wqi / 22.0)), 2),
            "zone": "High" if base_wqi >= 75 else "Moderate" if base_wqi >= 50 else "Low",
        },
        {
            "dimension": "Health Hazard",
            "likelihood": round(min(5.0, max(1.0, base_hhi * 2.0 + 1.0)), 2),
            "impact": round(min(5.0, max(1.0, base_hhi * 2.3 + 1.0)), 2),
            "zone": "High" if base_hhi > 1.2 else "Moderate" if base_hhi >= 0.9 else "Low",
        },
        {
            "dimension": "Climate Stress",
            "likelihood": round(min(5.0, max(1.0, 2.5 + abs(_to_float((scenario or {}).get("inputs", {}).get("rainfall_change_pct"), 0.0)) / 40.0)), 2),
            "impact": round(min(5.0, max(1.0, 2.3 + abs(_to_float((scenario or {}).get("inputs", {}).get("temperature_change_c"), 0.0)) / 2.0)), 2),
            "zone": "Moderate",
        },
        {
            "dimension": "Demand Pressure",
            "likelihood": round(min(5.0, max(1.0, 2.0 + abs(_to_float((scenario or {}).get("inputs", {}).get("population_change_pct"), 0.0)) / 30.0)), 2),
            "impact": round(min(5.0, max(1.0, 2.1 + abs(_to_float((scenario or {}).get("inputs", {}).get("agri_change_pct"), 0.0)) / 35.0)), 2),
            "zone": "Moderate",
        },
    ]

    scorecard = [
        {
            "label": "Base",
            "wqi": round(base_wqi, 2),
            "hhi": round(base_hhi, 2),
            "risk_index": base_risk,
        },
        {
            "label": "Scenario",
            "wqi": round(scenario_wqi, 2),
            "hhi": round(scenario_hhi, 2),
            "risk_index": scenario_risk,
        },
        {
            "label": "Comparison",
            "wqi": round(compare_wqi, 2),
            "hhi": round(compare_hhi, 2),
            "risk_index": compare_risk,
        },
    ]

    return {
        "kpis": {
            "base_risk_index": base_risk,
            "scenario_risk_index": scenario_risk,
            "comparison_risk_index": compare_risk,
            "scenario_risk_delta": round(scenario_risk - base_risk, 2),
            "comparison_risk_delta": round(compare_risk - base_risk, 2),
        },
        "horizon": horizon,
        "risk_matrix": risk_matrix,
        "scorecard": scorecard,
    }


def _local_research_bundle(base: dict, scenario: dict, compare_obj: dict, management_plan: dict, analytics: dict, reason: str) -> dict:
    base_wqi = (base or {}).get("wqi", {})
    base_hhi = (base or {}).get("hhi", {})
    scenario_wqi = (scenario or {}).get("wqi", {})
    scenario_hhi = (scenario or {}).get("hhi", {})

    compare_result = (compare_obj or {}).get("result", {}) if isinstance(compare_obj, dict) else {}
    compare_wqi = compare_result.get("wqi", {}) if isinstance(compare_result, dict) else {}
    compare_hhi = compare_result.get("hhi", {}) if isinstance(compare_result, dict) else {}

    def _delta(a, b):
        try:
            return round(float(b) - float(a), 2)
        except Exception:
            return None

    wqi_delta_scenario = _delta(base_wqi.get("value"), scenario_wqi.get("value"))
    hhi_delta_scenario = _delta(base_hhi.get("value"), scenario_hhi.get("value"))

    top_strategy = (management_plan.get("strategies") or [{}])[0]

    return {
        "insight_panel": {
            "summary": "Local fallback insight generated because Groq is unavailable.",
            "key_point": f"Base condition is WQI {base_wqi.get('class', '--')} and HHI {base_hhi.get('class', '--')}.",
            "top_driver": f"Scenario delta: WQI {wqi_delta_scenario}, HHI {hhi_delta_scenario}.",
        },
        "scenario_analysis": {
            "scenario_inputs": (scenario or {}).get("inputs", {}),
            "scenario_wqi": scenario_wqi,
            "scenario_hhi": scenario_hhi,
            "delta_vs_base": {
                "wqi_delta": wqi_delta_scenario,
                "hhi_delta": hhi_delta_scenario,
            },
        },
        "uncertainty_narrative": {
            "note": "Confidence narrative fallback mode. Use eval metrics panel for quantitative trust signals.",
            "reason": reason,
        },
        "district_comparison": {
            "target": {
                "district": (compare_obj or {}).get("district", ""),
                "year": (compare_obj or {}).get("year", ""),
            },
            "target_wqi": compare_wqi,
            "target_hhi": compare_hhi,
        },
        "management_strategies": {
            "summary": f"Priority action: {top_strategy.get('action', 'N/A')}.",
            "severity": management_plan.get("severity", "Unknown"),
            "risk_index": analytics.get("kpis", {}).get("base_risk_index"),
        },
        "journal_summary": {
            "abstract_like": "This draft was generated in local fallback mode due to upstream Groq unavailability."
            " Base, scenario, management, and comparison components are model-derived or deterministic outputs.",
            "limitation": "LLM synthesis unavailable; rerun after Groq connectivity is restored.",
        },
        "meta": {
            "provider": "local_fallback",
            "model": "rule_based_template",
            "reason": reason,
        },
    }


def _local_chat_response(question: str, context: dict, reason: str = "") -> str:
    q = (question or "").strip().lower()
    base = (context or {}).get("base_forecast", {})
    scenario = (context or {}).get("scenario_forecast", {})
    management = (context or {}).get("management_plan", {})
    analytics = (context or {}).get("analytics", {})

    base_wqi = (base or {}).get("wqi", {})
    base_hhi = (base or {}).get("hhi", {})
    top_strategy = (management.get("strategies") or [{}])[0]

    if "strategy" in q or "management" in q or "mitigation" in q:
        return (
            f"Fallback mode. Top strategy is '{top_strategy.get('action', 'N/A')}' with priority "
            f"{top_strategy.get('priority', 'N/A')}. Expected WQI gain is "
            f"{top_strategy.get('expected_wqi_gain', 'N/A')} and expected HHI reduction is "
            f"{top_strategy.get('expected_hhi_reduction', 'N/A')}."
        )

    if "scenario" in q or "what if" in q:
        swqi = (scenario or {}).get("wqi", {})
        shhi = (scenario or {}).get("hhi", {})
        return (
            f"Fallback mode. Scenario output is WQI {swqi.get('value', '--')} ({swqi.get('class', '--')}) and "
            f"HHI {shhi.get('value', '--')} ({shhi.get('class', '--')})."
        )

    if "risk" in q or "priority" in q:
        risk = (analytics.get("kpis", {}) or {}).get("base_risk_index", "--")
        return f"Fallback mode. Current base risk index is {risk}. Focus first on high-priority contamination controls."

    return (
        f"Fallback mode due to Groq unavailability ({reason or 'upstream error'}). "
        f"Base forecast: WQI {base_wqi.get('value', '--')} ({base_wqi.get('class', '--')}), "
        f"HHI {base_hhi.get('value', '--')} ({base_hhi.get('class', '--')}). "
        "Ask about strategy, scenario, or risk for targeted responses. "
        "Run /api/groq/status?check=1 to verify active key/auth."
    )


def _json_block_text(obj, max_lines: int = 80, max_chars: int = 180) -> str:
    try:
        raw = json.dumps(obj if obj is not None else {}, ensure_ascii=False, indent=2)
    except Exception:
        raw = str(obj)

    lines = []
    for line in raw.splitlines():
        lines.append(line[:max_chars] + ("..." if len(line) > max_chars else ""))

    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]

    return "\n".join(lines)


def _metric_range_text(metric: dict) -> str:
    if not isinstance(metric, dict):
        return "-"
    rng = metric.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return f"{rng[0]} - {rng[1]}"
    return "-"


def _build_research_bundle_pdf(bundle: dict, selection: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise RuntimeError("reportlab is required for PDF export") from exc

    styles = getSampleStyleSheet()
    code_style = ParagraphStyle(
        "code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    section_title = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="ST-WQHRNet Research Bundle",
        author="ST-WQHRNet",
    )

    story = []

    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    state = selection.get("state") or (bundle.get("base_forecast", {}) or {}).get("state") or "Tamil Nadu"
    district = selection.get("district") or (bundle.get("base_forecast", {}) or {}).get("district") or "-"
    block = selection.get("block") or (bundle.get("base_forecast", {}) or {}).get("block") or "-"
    village = selection.get("village") or (bundle.get("base_forecast", {}) or {}).get("village") or "-"
    year = selection.get("year") or (bundle.get("base_forecast", {}) or {}).get("year") or "-"

    story.append(Paragraph("ST-WQHRNet Research Bundle - Journal Appendix", styles["Title"]))
    story.append(Paragraph("Tamil Nadu Groundwater Risk Forecasting System", styles["Heading3"]))
    story.append(Spacer(1, 8))

    meta_rows = [
        ["Generated", gen_time],
        ["State", str(state)],
        ["District", str(district)],
        ["Block", str(block)],
        ["Village", str(village)],
        ["Year", str(year)],
        ["Provider", str((bundle.get("ai", {}) or {}).get("meta", {}).get("provider", "-"))],
        ["Model", str((bundle.get("ai", {}) or {}).get("meta", {}).get("model", "-"))],
    ]
    meta_tbl = Table(meta_rows, colWidths=[80 * mm, 95 * mm])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    base = bundle.get("base_forecast", {}) or {}
    scenario = bundle.get("scenario_forecast", {}) or {}
    compare_obj = bundle.get("comparison_forecast", {}) or {}
    compare = (compare_obj.get("result", {}) if isinstance(compare_obj, dict) else {}) or {}

    story.append(Paragraph("Forecast Summary", section_title))
    summary_rows = [["Dataset", "WQI Value", "WQI Class", "WQI Range", "HHI Value", "HHI Class", "HHI Range"]]

    for label, src in [("Base", base), ("Scenario", scenario), ("Comparison", compare)]:
        wqi = (src.get("wqi", {}) if isinstance(src, dict) else {}) or {}
        hhi = (src.get("hhi", {}) if isinstance(src, dict) else {}) or {}
        summary_rows.append(
            [
                label,
                str(wqi.get("value", "-")),
                str(wqi.get("class", "-")),
                _metric_range_text(wqi),
                str(hhi.get("value", "-")),
                str(hhi.get("class", "-")),
                _metric_range_text(hhi),
            ]
        )

    summary_tbl = Table(summary_rows, colWidths=[22 * mm, 22 * mm, 24 * mm, 28 * mm, 22 * mm, 24 * mm, 28 * mm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Management Strategy Board", section_title))
    strategies = ((bundle.get("management_plan", {}) or {}).get("strategies", []) or [])[:10]
    strat_rows = [["ID", "Priority", "Action", "Timeline", "Owner", "WQI Gain", "HHI Cut", "Cost"]]
    for s in strategies:
        strat_rows.append(
            [
                str(s.get("id", "-")),
                f"{s.get('priority', '-')}",
                Paragraph(str(s.get("action", "-")), styles["BodyText"]),
                str(s.get("timeline", "-")),
                Paragraph(str(s.get("owner", "-")), styles["BodyText"]),
                str(s.get("expected_wqi_gain", "-")),
                str(s.get("expected_hhi_reduction", "-")),
                str(s.get("cost_band", "-")),
            ]
        )

    strat_tbl = Table(strat_rows, colWidths=[10 * mm, 18 * mm, 52 * mm, 22 * mm, 30 * mm, 16 * mm, 16 * mm, 12 * mm])
    strat_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(strat_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Risk Matrix", section_title))
    matrix = ((bundle.get("analytics", {}) or {}).get("risk_matrix", []) or [])
    matrix_rows = [["Dimension", "Likelihood", "Impact", "Zone"]]
    for r in matrix:
        matrix_rows.append([str(r.get("dimension", "-")), str(r.get("likelihood", "-")), str(r.get("impact", "-")), str(r.get("zone", "-"))])

    matrix_tbl = Table(matrix_rows, colWidths=[70 * mm, 26 * mm, 26 * mm, 26 * mm])
    matrix_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(matrix_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("AI Narrative Appendix", section_title))
    ai = bundle.get("ai", {}) or {}
    sections = [
        ("Insight Panel", ai.get("insight_panel", {})),
        ("Scenario Analysis", ai.get("scenario_analysis", {})),
        ("Uncertainty Narrative", ai.get("uncertainty_narrative", {})),
        ("District Comparison", ai.get("district_comparison", {})),
        ("Management Strategies", ai.get("management_strategies", {})),
        ("Journal Summary", ai.get("journal_summary", {})),
    ]

    for title, payload in sections:
        story.append(Paragraph(title, styles["Heading3"]))
        story.append(Preformatted(_json_block_text(payload, max_lines=40, max_chars=160), code_style))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Note: This appendix is auto-generated from the current dashboard payload for reproducible reporting.", styles["Italic"]))

    doc.build(story)
    return buffer.getvalue()


def _append_csv_row(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "timestamp_utc",
        "event",
        "state",
        "district",
        "block",
        "village",
        "year",
        "wqi_value",
        "wqi_class",
        "wqi_range",
        "hhi_value",
        "hhi_class",
        "hhi_range",
        "payload_json",
        "result_json",
    ]
    extra_fields = sorted([k for k in row.keys() if k not in base_fields])
    fieldnames = base_fields + extra_fields

    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
        except Exception:
            existing_header = None

        if existing_header and existing_header != fieldnames:
            with LOG_PATH.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            with LOG_PATH.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)

    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/groq")
def groq_page():
    return render_template("groq.html")


@app.route("/api/locations", methods=["GET"])
def locations():
    return jsonify(
        {
            "states": predictor.get_states(),
            "districts": predictor.get_districts(),
            "map": predictor.get_map_meta(),
        }
    )


@app.route("/api/blocks", methods=["GET"])
def blocks():
    district = request.args.get("district")
    return jsonify({"blocks": predictor.get_blocks(district)})


@app.route("/api/villages", methods=["GET"])
def villages():
    district = request.args.get("district")
    block = request.args.get("block")
    return jsonify({"villages": predictor.get_villages(district, block)})


@app.route("/api/log_event", methods=["POST"])
def log_event():
    data = request.get_json(force=True) or {}
    event = str(data.get("event") or "unknown")
    payload = data.get("data") or {}
    state = payload.get("state") or "Tamil Nadu"
    district = payload.get("district") or ""
    block = payload.get("block") or ""
    village = payload.get("village") or ""
    year = payload.get("year") or ""

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "state": state,
        "district": district,
        "block": block,
        "village": village,
        "year": year,
        "wqi_value": "",
        "wqi_class": "",
        "wqi_range": "",
        "hhi_value": "",
        "hhi_class": "",
        "hhi_range": "",
        "payload_json": json.dumps(payload, ensure_ascii=False),
        "result_json": "",
    }
    _append_csv_row(row)
    return jsonify({"ok": True})


@app.route("/api/logs/download", methods=["GET"])
def download_logs():
    if not LOG_PATH.exists():
        _append_csv_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "init",
                "state": "",
                "district": "",
                "block": "",
                "village": "",
                "year": "",
                "wqi_value": "",
                "wqi_class": "",
                "wqi_range": "",
                "hhi_value": "",
                "hhi_class": "",
                "hhi_range": "",
                "payload_json": "{}",
                "result_json": "{}",
            }
        )
    return send_file(LOG_PATH, as_attachment=True, download_name="ui_events.csv", mimetype="text/csv")


@app.route("/api/groq/status", methods=["GET"])
def groq_status():
    do_check = str(request.args.get("check", "0")).lower() in {"1", "true", "yes"}
    out = {
        "ok": True,
        "enabled": groq_service.enabled,
        "model": groq_service.config.model,
        "key_fingerprint": groq_service.key_fingerprint,
    }
    if do_check:
        out["auth"] = groq_service.auth_check()
    return jsonify(out)


@app.route("/api/ai/research_bundle", methods=["POST"])
def research_bundle():
    data = request.get_json(force=True) or {}
    state = data.get("state") or "Tamil Nadu"
    district = data.get("district")
    block = data.get("block")
    village = data.get("village")
    year = int(data.get("year")) if data.get("year") else None

    if not district or year is None:
        return jsonify({"ok": False, "error": "district and year are required"}), 400

    base = predictor.predict_location(district, block, village, year)

    scenario_cfg = data.get("scenario") or {}
    scenario = _simulate_scenario(base, scenario_cfg)

    compare_cfg = data.get("compare") or {}
    compare_district = compare_cfg.get("district") or district
    compare_year = int(compare_cfg.get("year") or year)
    compare_result = predictor.predict_location(compare_district, None, None, compare_year)

    compare_obj = {
        "district": compare_district,
        "year": compare_year,
        "result": compare_result,
    }

    management_plan = _build_management_plan(base, scenario, compare_obj)
    analytics = _build_analytics_pack(base, scenario, compare_obj)

    bundle_input = {
        "base_forecast": base,
        "scenario_forecast": scenario,
        "comparison_forecast": compare_obj,
        "management_plan": management_plan,
        "analytics": analytics,
    }

    warning = None
    provider_event = "groq_bundle_success"
    try:
        ai = groq_service.generate_research_bundle(bundle_input)
    except Exception as exc:
        ai = _local_research_bundle(base, scenario, compare_obj, management_plan, analytics, str(exc))
        warning = "Groq unavailable; local fallback analysis returned."
        provider_event = "groq_bundle_fallback"

    out = {
        "ok": True,
        "base_forecast": base,
        "scenario_forecast": scenario,
        "comparison_forecast": compare_obj,
        "management_plan": management_plan,
        "analytics": analytics,
        "ai": ai,
    }
    if warning:
        out["warning"] = warning

    _append_csv_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": provider_event,
            "state": state,
            "district": district,
            "block": block or "",
            "village": village or "",
            "year": year,
            "wqi_value": base.get("wqi", {}).get("value", ""),
            "wqi_class": base.get("wqi", {}).get("class", ""),
            "wqi_range": "",
            "hhi_value": base.get("hhi", {}).get("value", ""),
            "hhi_class": base.get("hhi", {}).get("class", ""),
            "hhi_range": "",
            "payload_json": json.dumps(data, ensure_ascii=False),
            "result_json": json.dumps({"meta": ai.get("meta", {}), "warning": warning or ""}, ensure_ascii=False),
        }
    )

    return jsonify(out)


@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    data = request.get_json(force=True) or {}
    question = str(data.get("question") or "").strip()
    context = data.get("context") or {}

    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    provider = "Groq"
    warning = None
    try:
        answer = groq_service.chat_with_context(context, question)
    except Exception as exc:
        provider = "local_fallback"
        warning = str(exc)
        answer = _local_chat_response(question, context, reason=str(exc))

    _append_csv_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "groq_chat" if provider == "Groq" else "groq_chat_fallback",
            "state": (context.get("base_forecast", {}) or {}).get("state", "Tamil Nadu"),
            "district": (context.get("base_forecast", {}) or {}).get("district", ""),
            "block": (context.get("base_forecast", {}) or {}).get("block", ""),
            "village": (context.get("base_forecast", {}) or {}).get("village", ""),
            "year": (context.get("base_forecast", {}) or {}).get("year", ""),
            "wqi_value": (context.get("base_forecast", {}) or {}).get("wqi", {}).get("value", ""),
            "wqi_class": (context.get("base_forecast", {}) or {}).get("wqi", {}).get("class", ""),
            "wqi_range": "",
            "hhi_value": (context.get("base_forecast", {}) or {}).get("hhi", {}).get("value", ""),
            "hhi_class": (context.get("base_forecast", {}) or {}).get("hhi", {}).get("class", ""),
            "hhi_range": "",
            "payload_json": json.dumps({"question": question}, ensure_ascii=False),
            "result_json": json.dumps({"provider": provider, "warning": warning or ""}, ensure_ascii=False),
        }
    )

    out = {"ok": True, "answer": answer, "provider": provider}
    if warning:
        out["warning"] = warning
    return jsonify(out)


@app.route("/api/ai/research_bundle/pdf", methods=["POST"])
def research_bundle_pdf():
    data = request.get_json(force=True) or {}
    bundle = data.get("bundle") or {}
    selection = data.get("selection") or {}

    if not isinstance(bundle, dict) or not bundle:
        return jsonify({"ok": False, "error": "bundle payload is required"}), 400

    try:
        pdf_bytes = _build_research_bundle_pdf(bundle, selection)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    district = selection.get("district") or (bundle.get("base_forecast", {}) or {}).get("district") or "district"
    year = selection.get("year") or (bundle.get("base_forecast", {}) or {}).get("year") or "year"
    safe_district = "".join(ch if str(ch).isalnum() else "_" for ch in str(district)).strip("_") or "district"
    filename = f"research_bundle_{safe_district}_{year}.pdf"

    _append_csv_row(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "groq_bundle_pdf_export",
            "state": selection.get("state") or "Tamil Nadu",
            "district": district,
            "block": selection.get("block") or "",
            "village": selection.get("village") or "",
            "year": year,
            "wqi_value": (bundle.get("base_forecast", {}) or {}).get("wqi", {}).get("value", ""),
            "wqi_class": (bundle.get("base_forecast", {}) or {}).get("wqi", {}).get("class", ""),
            "wqi_range": "",
            "hhi_value": (bundle.get("base_forecast", {}) or {}).get("hhi", {}).get("value", ""),
            "hhi_class": (bundle.get("base_forecast", {}) or {}).get("hhi", {}).get("class", ""),
            "hhi_range": "",
            "payload_json": json.dumps(selection, ensure_ascii=False),
            "result_json": json.dumps({"download": filename}, ensure_ascii=False),
        }
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    state = data.get("state") or "Tamil Nadu"
    district = data.get("district")
    block = data.get("block")
    village = data.get("village")
    year = int(data.get("year"))

    try:
        result = predictor.predict_location(district, block, village, year)
        wqi = result.get("wqi", {}) if isinstance(result, dict) else {}
        hhi = result.get("hhi", {}) if isinstance(result, dict) else {}
        wqi_range = wqi.get("range") if isinstance(wqi, dict) else None
        hhi_range = hhi.get("range") if isinstance(hhi, dict) else None
        wqi_range_text = ""
        hhi_range_text = ""
        if isinstance(wqi_range, (list, tuple)) and len(wqi_range) == 2:
            wqi_range_text = f"{wqi_range[0]}-{wqi_range[1]}"
        if isinstance(hhi_range, (list, tuple)) and len(hhi_range) == 2:
            hhi_range_text = f"{hhi_range[0]}-{hhi_range[1]}"
        _append_csv_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "predict_success",
                "state": state,
                "district": district,
                "block": block or "",
                "village": village or "",
                "year": year,
                "wqi_value": wqi.get("value", ""),
                "wqi_class": wqi.get("class", ""),
                "wqi_range": wqi_range_text,
                "hhi_value": hhi.get("value", ""),
                "hhi_class": hhi.get("class", ""),
                "hhi_range": hhi_range_text,
                "payload_json": json.dumps(data, ensure_ascii=False),
                "result_json": json.dumps(result, ensure_ascii=False),
            }
        )
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        _append_csv_row(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "predict_error",
                "state": state,
                "district": district or "",
                "block": block or "",
                "village": village or "",
                "year": data.get("year") or "",
                "wqi_value": "",
                "wqi_class": "",
                "wqi_range": "",
                "hhi_value": "",
                "hhi_class": "",
                "hhi_range": "",
                "payload_json": json.dumps(data, ensure_ascii=False),
                "result_json": json.dumps({"error": str(exc)}, ensure_ascii=False),
            }
        )
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
