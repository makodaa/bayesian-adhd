"""PDF Report Generation Service for EEG Analysis Results."""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

from ..config import APP_VERSION
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

DEFAULT_NORMATIVE_RANGES: dict[str, dict[str, float]] = {
    # TODO: replace with actual normative percentiles
    "delta": {"p25": 25.0, "p75": 55.0},
    "theta": {"p25": 4.0, "p75": 8.0},
    "alpha": {"p25": 8.0, "p75": 12.0},
    "beta": {"p25": 10.0, "p75": 20.0},
    "gamma": {"p25": 1.0, "p75": 5.0},
}


def _format_date(value) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, datetime):
        return value.strftime("%m/%d/%Y")
    if isinstance(value, date):
        return value.strftime("%m/%d/%Y")
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed.strftime("%m/%d/%Y")
    except ValueError:
        return str(value)


def _format_datetime(value) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, datetime):
        return value.strftime("%m/%d/%Y %H:%M")
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed.strftime("%m/%d/%Y %H:%M")
    except ValueError:
        return str(value)


def _display(value, fallback: str = "Not recorded") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _format_hours(value) -> str:
    if value is None:
        return "Not recorded"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _display(value)
    if numeric.is_integer():
        return f"{int(numeric)} h"
    return f"{numeric:.1f} h"


def _clean_percent(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 100.0))


def _map_class_label(predicted_class: str) -> str:
    label = (predicted_class or "").strip()
    mapping = {
        "Inattentive (ADHD-I)": "ADHD - Inattentive Type",
        "Combined / C (ADHD-C)": "ADHD - Combined Type",
        "Hyperactive-Impulsive (ADHD-H)": "ADHD - Hyperactive/Impulsive Type",
        "Not ADHD": "Non-ADHD",
    }
    if label in mapping:
        return mapping[label]
    if not label:
        return "Inconclusive"
    return label


def _extract_ratio(ratios: list[dict], ratio_name: str) -> float:
    for ratio in ratios:
        if ratio.get("ratio_name") == ratio_name:
            try:
                return float(ratio.get("ratio_value", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def generate_band_findings(
    band_power: dict[str, float],
    normative_ranges: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    alpha = band_power.get("alpha", 0.0)
    beta = band_power.get("beta", 0.0)
    theta = band_power.get("theta", 0.0)
    tbr = theta / beta if beta > 0 else 0.0

    ranges = normative_ranges or DEFAULT_NORMATIVE_RANGES

    def normative_clause(band_key: str, value: float) -> str:
        band_norms = ranges.get(band_key)
        if not band_norms:
            return ""
        p25 = band_norms.get("p25")
        p75 = band_norms.get("p75")
        if p25 is None or p75 is None:
            return ""
        if value < p25:
            state = "Below"
        elif value > p75:
            state = "Above"
        else:
            state = "Within"
        return (
            f"{state} normative {band_key} range of {p25:.1f}-{p75:.1f}%."
        )

    def combine_clause(note: str, clause: str) -> str:
        if clause and note:
            return f"{clause} {note}"
        return clause or note

    if alpha > 12:
        label = f"Elevated alpha power ({alpha:.2f}%)."
        note = (
            "May reflect enhanced relaxed wakefulness, reduced cognitive engagement, "
            "or inattentive state consistent with some ADHD presentations."
        )
    elif alpha < 8:
        label = f"Reduced alpha power ({alpha:.2f}%)."
        note = (
            "Reduced posterior alpha; may indicate heightened cortical arousal "
            "or hyperactive/impulsive state."
        )
    else:
        label = f"Alpha power within expected range ({alpha:.2f}%)."
        note = "Posterior alpha rhythm is appropriate for age and vigilance state."
    findings.append(
        {
            "label": label,
            "properties": combine_clause(note, normative_clause("alpha", alpha)),
        }
    )

    if beta > 20:
        label = f"Elevated beta power ({beta:.2f}%)."
        note = (
            "Elevated fast-wave activity; may reflect heightened cortical arousal, "
            "anxiety, or stimulant medication effect."
        )
    elif beta < 10:
        label = f"Reduced beta power ({beta:.2f}%)."
        note = (
            "Reduced fast-frequency activity; may indicate hypo-arousal or "
            "inattentive cortical state."
        )
    else:
        label = f"Beta power within expected range ({beta:.2f}%)."
        note = "Fast cortical rhythms are within expected range."
    findings.append(
        {
            "label": label,
            "properties": combine_clause(note, normative_clause("beta", beta)),
        }
    )

    if theta > 8:
        label = f"Elevated theta power ({theta:.2f}%)."
        note = (
            "Elevated theta is a common EEG correlate observed in ADHD populations; "
            "diffuse excess theta may indicate cortical under-arousal or inattention."
        )
    elif theta < 4:
        label = f"Reduced theta power ({theta:.2f}%)."
        note = "Low theta power; consider vigilance state and age norms."
    else:
        label = f"Theta power within expected range ({theta:.2f}%)."
        note = "Theta distribution is appropriate."
    findings.append(
        {
            "label": label,
            "properties": combine_clause(note, normative_clause("theta", theta)),
        }
    )

    ab = alpha + beta
    if ab > 25:
        label = f"Cortical arousal pattern present (A+B: {ab:.2f}%)."
        arousal = "EEG reflects an activated, alert state."
    elif ab >= 15:
        label = f"Cortical arousal mildly reduced (A+B: {ab:.2f}%)."
        arousal = (
            "Background reflects a transitional or mildly inattentive state."
        )
    else:
        label = f"Cortical arousal markedly reduced (A+B: {ab:.2f}%)."
        arousal = (
            "Dominant slow-wave activity suggests hypo-arousal or pronounced "
            "inattentive state."
        )
    findings.append(
        {
            "label": label,
            "properties": arousal,
        }
    )

    return findings


def build_band_power_block(
    band_power: dict[str, float],
    styles: dict[str, ParagraphStyle],
    content_width: float,
) -> Table:
    bands = [
        ("Delta", "0.5-4 Hz", "delta"),
        ("Theta", "4-8 Hz", "theta"),
        ("Alpha", "8-13 Hz", "alpha"),
        ("Beta", "13-30 Hz", "beta"),
        ("Gamma", "30-100 Hz", "gamma"),
    ]
    bar_colors = {
        "delta": "#111111",
        "theta": "#444444",
        "alpha": "#777777",
        "beta": "#aaaaaa",
        "gamma": "#cccccc",
    }

    left_content: list[object] = []
    for band_name, freq_range, key in bands:
        val = band_power.get(key, 0.0)
        inline_text = (
            f"{band_name} ({freq_range}) "
            f"<font name=\"Helvetica\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            f"Properties: {val:.2f}%</font>"
        )
        left_content.append(Paragraph(inline_text, styles["SubSubHeader"]))
        left_content.append(Spacer(1, 1))

    right_w = content_width * 0.55
    left_w = content_width * 0.45
    label_w = 32
    pct_w = 35
    bar_h = 10
    bar_spacing = 6
    chart_w = max(right_w - label_w - pct_w, 60)
    total_h = len(bands) * (bar_h + bar_spacing)
    max_val = max(band_power[k] for _, _, k in bands) if band_power else 1

    drawing = Drawing(right_w, total_h)
    labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    for i, ((_, _, key), label) in enumerate(zip(bands, labels)):
        val = band_power.get(key, 0.0)
        bar_w = (val / max_val) * chart_w if max_val > 0 else 0
        y = total_h - (i + 1) * (bar_h + bar_spacing) + bar_spacing

        drawing.add(
            String(0, y + 2, label, fontSize=6, fontName="Helvetica")
        )
        rect = Rect(label_w, y, bar_w, bar_h)
        rect.fillColor = colors.HexColor(bar_colors[key])
        rect.strokeColor = colors.HexColor(bar_colors[key])
        rect.strokeWidth = 0
        drawing.add(rect)
        drawing.add(
            String(
                label_w + bar_w + 3,
                y + 2,
                f"{val:.1f}%",
                fontSize=6.5,
                fontName="Helvetica",
            )
        )

    outer = Table([[left_content, drawing]], colWidths=[left_w, right_w])
    outer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return outer


def build_disclaimer_box(
    styles: dict[str, ParagraphStyle],
    content_width: float,
) -> Table:
    disclaimer_text = (
        "This report is generated by a computational EEG analysis tool and is intended "
        "to support - not replace - clinical evaluation. The findings, classifications, "
        "and comments presented here do not constitute a definitive medical diagnosis. "
        "All results must be interpreted by a qualified healthcare professional in the "
        "context of the patient's full clinical history and presentation."
    )
    return build_shaded_content_box(
        disclaimer_text,
        styles,
        content_width,
        style_key="DisclaimerBody",
    )


def build_shaded_content_box(
    text: str,
    styles: dict[str, ParagraphStyle],
    content_width: float,
    bold: bool = False,
    style_key: str | None = None,
    min_height: float | None = None,
) -> Table:
    resolved_key = style_key or ("ShadedBoxBold" if bold else "ShadedBoxText")
    inner = [Paragraph(text, styles[resolved_key])]
    table = Table([[inner]], colWidths=[content_width])
    style_rules = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f0f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if min_height:
        style_rules.append(("MINROWHEIGHT", (0, 0), (-1, -1), min_height))
    table.setStyle(TableStyle(style_rules))
    return table


def _confidence_bar(confidence: float, width: float) -> Drawing:
    height = 10
    track = colors.HexColor("#c8dff5")
    fill = colors.HexColor("#2a6099")
    drawing = Drawing(width, height)
    track_rect = Rect(0, 0, width, height)
    track_rect.fillColor = track
    track_rect.strokeColor = track
    track_rect.strokeWidth = 0
    drawing.add(track_rect)

    fill_rect = Rect(0, 0, width * confidence, height)
    fill_rect.fillColor = fill
    fill_rect.strokeColor = fill
    fill_rect.strokeWidth = 0
    drawing.add(fill_rect)
    return drawing


def make_footer_canvas(footer_fields: list[str]):
    class FooterCanvas(rl_canvas.Canvas):
        _footer_fields = footer_fields

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()  # type: ignore[attr-defined]

        def save(self):
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_footer(self._pageNumber, total)  # type: ignore[attr-defined]
                rl_canvas.Canvas.showPage(self)
            rl_canvas.Canvas.save(self)

        def _draw_footer(self, page_num: int, total: int):
            self.saveState()
            self.setFont("Helvetica", 7)
            self.setFillColor(colors.HexColor("#555555"))
            margin = 18 * mm
            page_width = A4[0]
            fields = list(self._footer_fields)
            fields.append(f"Page {page_num} / {total}")
            slots = len(fields)
            if slots == 0:
                self.restoreState()
                return
            slot_w = (page_width - (2 * margin)) / slots
            for idx, field in enumerate(fields):
                x_start = margin + (idx * slot_w)
                x_mid = x_start + (slot_w / 2)
                x_end = x_start + slot_w
                if idx == 0:
                    self.drawString(x_start, 3, field)
                elif idx == slots - 1:
                    self.drawRightString(x_end, 3, field)
                else:
                    self.drawCentredString(x_mid, 3, field)
            self.restoreState()

    return FooterCanvas


class PDFReportService:
    """Generate clinical PDF reports for EEG analysis results."""

    def __init__(self):
        self.styles = self._build_styles()

    def _build_styles(self) -> dict[str, ParagraphStyle]:
        styles: dict[str, ParagraphStyle] = {}

        styles["ReportTitle"] = ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.black,
            spaceAfter=2,
        )
        styles["SectionHeader"] = ParagraphStyle(
            name="SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=0,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=6,
            spaceAfter=1,
        )
        styles["SectionHeaderTight"] = ParagraphStyle(
            name="SectionHeaderTight",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=-6,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=6,
            spaceAfter=1,
        )
        styles["SubHeader"] = ParagraphStyle(
            name="SubHeader",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=0,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=2,
            spaceAfter=1,
        )
        styles["SubSubHeader"] = ParagraphStyle(
            name="SubSubHeader",
            fontName="Helvetica-BoldOblique",
            fontSize=7.5,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=2,
        )
        styles["DetailLabel"] = ParagraphStyle(
            name="DetailLabel",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=0,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=0,
            spaceAfter=1,
        )
        styles["DetailLabelTight"] = ParagraphStyle(
            name="DetailLabelTight",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=-6,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=0,
            spaceAfter=1,
        )
        styles["SubSubHeaderItalic"] = ParagraphStyle(
            name="SubSubHeaderItalic",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=2,
        )
        styles["BodyText"] = ParagraphStyle(
            name="BodyText",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            spaceAfter=2,
        )
        styles["HeaderMeta"] = ParagraphStyle(
            name="HeaderMeta",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            spaceAfter=2,
        )
        styles["BodyTextRight"] = ParagraphStyle(
            name="BodyTextRight",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            alignment=2,
            spaceAfter=2,
        )
        styles["HeaderMetaRight"] = ParagraphStyle(
            name="HeaderMetaRight",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            alignment=2,
            spaceAfter=2,
        )
        styles["CenteredBody"] = ParagraphStyle(
            name="CenteredBody",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            alignment=1,
            spaceAfter=2,
        )
        styles["IndentedBody"] = ParagraphStyle(
            name="IndentedBody",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=12,
            leading=7,
            spaceAfter=2,
        )
        styles["FindingsBody"] = ParagraphStyle(
            name="FindingsBody",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=12,
            leading=8,
            spaceAfter=2,
        )
        styles["ShadedBoxText"] = ParagraphStyle(
            name="ShadedBoxText",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            spaceAfter=2,
        )
        styles["ShadedBoxBold"] = ParagraphStyle(
            name="ShadedBoxBold",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            spaceAfter=2,
        )
        styles["NarrativeLabel"] = ParagraphStyle(
            name="NarrativeLabel",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.HexColor("#1a3a5c"),
            spaceAfter=2,
        )
        styles["DisclaimerHeader"] = ParagraphStyle(
            name="DisclaimerHeader",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            spaceAfter=2,
        )
        styles["DisclaimerBody"] = ParagraphStyle(
            name="DisclaimerBody",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=colors.black,
            leading=7,
            spaceAfter=2,
        )
        styles["LogoPlaceholder"] = ParagraphStyle(
            name="LogoPlaceholder",
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=colors.HexColor("#999999"),
            alignment=1,
        )
        return styles

    def generate_report(
        self,
        result_data: dict,
        clinician_data: dict,
        normative_ranges: dict[str, dict[str, float]] | None = None,
    ) -> bytes:
        logger.info("Generating PDF report for result %s", result_data.get("result_id"))

        report_data = self._build_report_data(
            result_data, clinician_data, normative_ranges
        )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=25 * mm,
        )

        story: list = []
        story.extend(self._build_header(report_data))
        story.append(Spacer(1, 0))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 1))
        story.extend(self._build_referral_patient(report_data))
        story.extend(self._build_recording_assessment(report_data))
        story.extend(self._build_findings(report_data))
        story.extend(self._build_model_classification(report_data))
        story.append(self._build_summary_disclaimer_columns(report_data, doc.width))
        story.extend(self._build_signature_block(report_data))

        footer_fields = report_data["footer_fields"]
        doc.multiBuild(story, canvasmaker=make_footer_canvas(footer_fields))

        pdf_content = buffer.getvalue()
        buffer.close()
        logger.info("PDF report generated successfully, size: %s bytes", len(pdf_content))
        return pdf_content

    def _build_report_data(
        self,
        result_data: dict,
        clinician_data: dict,
        normative_ranges: dict[str, dict[str, float]] | None,
    ) -> dict:
        inferred = result_data.get("inferenced_at")
        report_date = _format_date(inferred)

        clinician_name = " ".join(
            filter(
                None,
                [
                    clinician_data.get("first_name", ""),
                    clinician_data.get("middle_name", ""),
                    clinician_data.get("last_name", ""),
                ],
            )
        ).strip()
        clinician_occupation = _display(
            clinician_data.get("occupation"), "Not recorded"
        )

        subject_code = _display(result_data.get("subject_code"))
        recording_id = _display(result_data.get("recording_id"))
        result_id = _display(result_data.get("result_id"))

        band_powers = result_data.get("band_powers", [])
        avg_relative: dict[str, list[float]] = {}
        for row in band_powers:
            band = str(row.get("frequency_band", "")).lower()
            if band not in ("delta", "theta", "alpha", "beta", "gamma"):
                continue
            try:
                avg_relative.setdefault(band, []).append(float(row.get("relative_power", 0)))
            except (TypeError, ValueError):
                avg_relative.setdefault(band, []).append(0.0)

        band_power = {}
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            values = avg_relative.get(band) or [0.0]
            band_power[band] = (sum(values) / len(values)) * 100

        ratios = result_data.get("ratios", [])
        tbr = _extract_ratio(ratios, "theta_beta_ratio")
        if tbr == 0.0 and band_power.get("beta"):
            tbr = band_power.get("theta", 0.0) / band_power.get("beta", 1.0)

        predicted_class = result_data.get("predicted_class", "")
        label = _map_class_label(predicted_class)
        confidence = result_data.get("confidence_score", 0.0) or 0.0
        confidence_pct = confidence * 100 if confidence <= 1 else confidence

        summary_findings = (
            "Relative band power summary: "
            f"Delta {band_power['delta']:.2f}%, Theta {band_power['theta']:.2f}%, "
            f"Alpha {band_power['alpha']:.2f}%, Beta {band_power['beta']:.2f}%, "
            f"Gamma {band_power['gamma']:.2f}%. Theta/Beta Ratio {tbr:.2f}."
        )
        diagnostic_significance = (
            f"Model classification: {label} ({confidence_pct:.1f}% confidence). "
            "Clinical correlation is required."
        )
        clinical_comments = _display(result_data.get("notes"), "Not recorded")

        footer_fields = [
            f"Subject Code: {subject_code}",
            f"Recording ID: {recording_id}",
            report_date,
        ]

        return {
            "unit_name": "EEG Assessment Unit",
            "institution_name": "BayesianADHD",
            "report_date": report_date,
            "logo_path": None,
            "referral_name": "Not recorded",
            "referral_institution": "Not recorded",
            "referral_address": ["Not recorded"],
            "patient_name": None,
            "patient_id": subject_code,
            "patient_address": None,
            "patient_dob": "Not recorded",
            "patient_age_at_study": (
                f"{result_data.get('age')} years"
                if result_data.get("age") is not None
                else "Not recorded"
            ),
            "gender": _display(result_data.get("gender")),
            "study_id": result_id,
            "local_study_id": recording_id,
            "technician": _display(result_data.get("technician_name")),
            "duration_minutes": _display(result_data.get("duration_minutes")),
            "recorded_minutes": _display(result_data.get("recorded_minutes")),
            "medication": _display(result_data.get("medication"), "None"),
            "alertness": _display(result_data.get("alertness"), "Not recorded"),
            "sleep_hours": _display(result_data.get("sleep_hours")),
            "coffee_hours_ago": _display(result_data.get("coffee_hours_ago")),
            "drugs_hours_ago": _display(result_data.get("drugs_hours_ago")),
            "meal_hours_ago": _display(result_data.get("meal_hours_ago")),
            "sensor_group": "10-20 International, 19 channels",
            "band_power": band_power,
            "normative_ranges": normative_ranges or DEFAULT_NORMATIVE_RANGES,
            "model_classification": {
                "label": label,
                "confidence": confidence if confidence <= 1 else confidence / 100,
                "theta_beta_ratio": tbr,
                "model_version": f"ADHDNet v{APP_VERSION}",
                "notes": "",
            },
            "summary_findings": summary_findings,
            "diagnostic_significance": diagnostic_significance,
            "clinical_comments": clinical_comments,
            "post_assessment_notes": _display(
                result_data.get("post_assessment_notes"), "Not recorded"
            ),
            "technician_name": _display(result_data.get("technician_name")),
            "clinician_name": clinician_name or "Not recorded",
            "clinician_occupation": clinician_occupation,
            "supervising_physician": "Not recorded",
            "footer_fields": footer_fields,
        }

    def _build_header(self, report_data: dict) -> list:
        elements: list = []
        left_block = [
            Paragraph(
                "Electroencephalogram (EEG) Report", self.styles["ReportTitle"]
            ),
            Paragraph(
                _display(report_data["institution_name"]),
                self.styles["HeaderMeta"],
            ),
        ]

        right_block = [
            Paragraph(
                f"Date: {report_data['report_date']}",
                self.styles["HeaderMetaRight"],
            )
        ]

        table = Table(
            [[left_block, right_block]],
            colWidths=[doc_width() * 0.7, doc_width() * 0.3],
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        elements.append(table)
        return elements

    def _build_referral_patient(self, report_data: dict) -> list:
        elements: list = []

        referral_lines = [
            Paragraph("REFERRAL FROM", self.styles["SectionHeader"]),
            Paragraph(f"Name: {_display(report_data['referral_name'])}", self.styles["BodyText"]),
            Paragraph(
                f"Institution: {_display(report_data['referral_institution'])}",
                self.styles["BodyText"],
            ),
        ]
        for line in report_data.get("referral_address", []) or ["Not recorded"]:
            referral_lines.append(Paragraph(_display(line), self.styles["BodyText"]))

        patient_lines = [
            Paragraph("SUBJECT INFORMATION", self.styles["SectionHeader"]),
            Paragraph(
                f"Subject Code: {_display(report_data['patient_id'])}",
                self.styles["BodyText"],
            ),
            Paragraph(
                f"Age at Assessment: {_display(report_data['patient_age_at_study'])}",
                self.styles["BodyText"],
            ),
            Paragraph(
                f"Gender: {_display(report_data.get('gender'))}",
                self.styles["BodyText"],
            ),
        ]

        table = Table([[referral_lines, patient_lines]], colWidths=[doc_width() / 2, doc_width() / 2])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (1, 0), (1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LINEBEFORE", (1, 0), (1, 0), 0.5, colors.black),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 1))
        return elements

    def _build_recording_assessment(self, report_data: dict) -> list:
        elements: list = []

        elements.append(
            Paragraph("ASSESSMENT INFORMATION", self.styles["SectionHeaderTight"])
        )
        elements.append(Spacer(1, 1))

        bar_rows = [
            [
                f"Assessment ID: {report_data['study_id']}   "
                f"Recording ID: {report_data['local_study_id']}   "
                f"Technician: {report_data['technician']}"
            ],
            [
                f"Recorded minutes: {report_data['recorded_minutes']}   "
                f"Duration minutes: {report_data['duration_minutes']}"
            ],
        ]

        bar_table = Table(bar_rows, colWidths=[doc_width()])
        bar_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8e8e8")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(bar_table)
        elements.append(Spacer(1, 3))

        left_details = [
            ["Sensor group", _display(report_data["sensor_group"])],
            ["Medication", _display(report_data["medication"])],
            [
                "Time since drug intake (hours)",
                _format_hours(report_data.get("drugs_hours_ago")),
            ],
        ]
        right_details = [
            ["Alertness", _display(report_data["alertness"])],
            ["Sleep hours", _format_hours(report_data.get("sleep_hours"))],
            [
                "Time since caffeine intake (hours)",
                _format_hours(report_data.get("coffee_hours_ago")),
            ],
            [
                "Time since last meal (hours)",
                _format_hours(report_data.get("meal_hours_ago")),
            ],
        ]

        def build_detail_table(rows: list[list[str]]) -> Table:
            table = Table(rows, colWidths=[45 * mm, (doc_width() / 2) - 45 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            return table

        left_table = build_detail_table(left_details)
        right_table = build_detail_table(right_details)
        details_table = Table(
            [[left_table, right_table]],
            colWidths=[doc_width() / 2, doc_width() / 2],
        )
        details_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        elements.append(details_table)

        elements.append(
            Paragraph("Relative Band Powers", self.styles["DetailLabelTight"])
        )
        elements.append(Spacer(1, 1))
        band_block = build_band_power_block(
            report_data["band_power"], self.styles, doc_width()
        )
        band_block.hAlign = "LEFT"
        band_wrapper = Table([[band_block]], colWidths=[doc_width()])
        band_wrapper.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        elements.append(band_wrapper)
        elements.append(Spacer(1, 2))
        return elements

    def _build_findings(self, report_data: dict) -> list:
        elements: list = []
        elements.append(Paragraph("FINDINGS", self.styles["SectionHeaderTight"]))
        elements.append(Spacer(1, 1))

        elements.append(Paragraph("Spectral Findings", self.styles["SubHeader"]))
        for finding in generate_band_findings(
            report_data["band_power"], report_data.get("normative_ranges")
        ):
            elements.append(Paragraph(finding["label"], self.styles["SubSubHeader"]))
            elements.append(
                Paragraph(
                    finding["properties"],
                    self.styles["FindingsBody"],
                )
            )
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 0))

        return elements

    def _build_model_classification(self, report_data: dict) -> list:
        model = report_data["model_classification"]
        confidence = model["confidence"]
        confidence_pct = confidence * 100

        elements: list = []
        elements.append(Paragraph("Model Classification", self.styles["SubHeader"]))
        elements.append(Spacer(1, 1))

        confidence_statement = self._format_confidence_statement(confidence, confidence_pct)
        classification_statement = (
            "The spatiotemporal model found the recording to have features supportive of "
            f"{model['label']}. {confidence_statement}"
        )
        elements.append(
            Paragraph(classification_statement, self.styles["FindingsBody"])
        )
        elements.append(Spacer(1, 2))
        return elements

    @staticmethod
    def _format_confidence_statement(confidence: float, confidence_pct: float) -> str:
        if confidence >= 0.8:
            tier = "High confidence"
        elif confidence >= 0.6:
            tier = "Moderate confidence"
        else:
            tier = "Low confidence"
        return f"{tier} ({confidence_pct:.1f}%)."

    def _build_summary_disclaimer_columns(self, report_data: dict, width: float) -> Table:
        summary_block = [
            Paragraph("SUMMARY OF FINDINGS", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["summary_findings"], self.styles, width / 2 - 6
            ),
        ]
        pre_block = [
            Paragraph("PRE-ASSESSMENT NOTES", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["clinical_comments"],
                self.styles,
                width / 2 - 6,
                min_height=40,
            ),
        ]
        diagnostic_block = [
            Paragraph("DIAGNOSTIC SIGNIFICANCE", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["diagnostic_significance"],
                self.styles,
                width / 2 - 6,
                bold=True,
            ),
        ]
        post_block = [
            Paragraph("POST-ASSESSMENT NOTES", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["post_assessment_notes"],
                self.styles,
                width / 2 - 6,
                min_height=40,
            ),
        ]
        disclaimer_block = [
            Paragraph("IMPORTANT DISCLAIMER", self.styles["SectionHeader"]),
            build_disclaimer_box(self.styles, width / 2 - 6),
        ]

        table = Table(
            [
                [summary_block, pre_block],
                [diagnostic_block, post_block],
                [disclaimer_block, ""],
            ],
            colWidths=[width / 2, width / 2],
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _build_signature_block(self, report_data: dict) -> list:
        elements: list = []
        signature_line = "________________________"
        clinician_name = report_data.get("clinician_name", "Not recorded")
        clinician_occupation = report_data.get("clinician_occupation", "")
        clinician_display = clinician_name
        clinician_role = (
            clinician_occupation
            if clinician_occupation and clinician_occupation != "Not recorded"
            else "Clinician"
        )
        rows = [
            [signature_line, signature_line, signature_line],
            [
                report_data["technician_name"],
                Paragraph(clinician_display, self.styles["CenteredBody"]),
                report_data["supervising_physician"],
            ],
            ["Technician", clinician_role, "Supervising clinician"],
        ]
        table = Table(rows, colWidths=[doc_width() / 3] * 3)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
                    ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, 0), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                    ("TOPPADDING", (0, 1), (-1, 1), 1),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 1),
                    ("TOPPADDING", (0, 2), (-1, 2), 1),
                    ("BOTTOMPADDING", (0, 2), (-1, 2), 1),
                ]
            )
        )
        elements.append(table)
        return elements


def doc_width() -> float:
    return A4[0] - (18 * mm) * 2
