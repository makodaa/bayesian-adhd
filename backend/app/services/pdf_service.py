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


def generate_band_findings(band_power: dict[str, float]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    alpha = band_power.get("alpha", 0.0)
    beta = band_power.get("beta", 0.0)
    theta = band_power.get("theta", 0.0)
    tbr = theta / beta if beta > 0 else 0.0

    if alpha > 12:
        label = f"Increased Alpha activity ({alpha:.2f}%)"
        note = ""
    elif alpha < 8:
        label = f"Decreased Alpha activity ({alpha:.2f}%)"
        note = ""
    else:
        label = f"Alpha activity within normal limits ({alpha:.2f}%)"
        note = ""
    alpha_properties = label if not note else f"{label}. {note}"
    findings.append(
        {
            "label": "Alpha (8-13 Hz)",
            "properties": alpha_properties,
        }
    )

    if beta > 20:
        label = f"Increased Beta activity ({beta:.2f}%)"
        note = (
            "Elevated fast-wave activity; may reflect heightened cortical arousal, "
            "anxiety, or stimulant medication effect."
        )
    elif beta < 10:
        label = f"Decreased Beta activity ({beta:.2f}%)"
        note = (
            "Reduced fast-frequency activity; may indicate hypo-arousal or "
            "inattentive cortical state."
        )
    else:
        label = f"Beta activity within normal limits ({beta:.2f}%)"
        note = "Fast cortical rhythms are within expected range."
    findings.append(
        {
            "label": "Beta (13-30 Hz)",
            "properties": f"{label}. {note}",
        }
    )

    if theta > 8:
        label = f"Increased Theta activity ({theta:.2f}%)"
        note = (
            "Elevated theta is a common EEG correlate observed in ADHD populations; "
            "diffuse excess theta may indicate cortical under-arousal or inattention."
        )
    elif theta < 4:
        label = f"Decreased Theta activity ({theta:.2f}%)"
        note = "Low theta power; consider vigilance state and age norms."
    else:
        label = f"Theta activity within normal limits ({theta:.2f}%)"
        note = "Theta distribution is appropriate."
    findings.append(
        {
            "label": "Theta (4-8 Hz)",
            "properties": f"{label}. {note}",
        }
    )

    ab = alpha + beta
    if ab > 25:
        arousal = "Cortical arousal pattern present; EEG reflects an activated, alert state."
    elif ab >= 15:
        arousal = (
            "Cortical arousal mildly reduced; background reflects a transitional "
            "or mildly inattentive state."
        )
    else:
        arousal = (
            "Cortical arousal markedly reduced; dominant slow-wave activity suggests "
            "hypo-arousal or pronounced inattentive state."
        )
    findings.append(
        {
            "label": "Cortical Arousal",
            "properties": f"{arousal} (Combined Alpha+Beta: {ab:.2f}%)",
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
) -> Table:
    resolved_key = style_key or ("ShadedBoxBold" if bold else "ShadedBoxText")
    inner = [Paragraph(text, styles[resolved_key])]
    table = Table([[inner]], colWidths=[content_width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f0f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
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


def make_footer_canvas(footer_text: str):
    class FooterCanvas(rl_canvas.Canvas):
        _footer_text = footer_text

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
            footer_height = 12
            self.setFillColor(colors.HexColor("#222222"))
            self.rect(0, 0, A4[0], footer_height, stroke=0, fill=1)
            self.setFont("Helvetica", 7)
            self.setFillColor(colors.white)
            text = f"{self._footer_text}    Page {page_num} / {total}"
            self.drawString(18 * mm, 3, text)
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
            fontSize=8,
            textColor=colors.black,
            leftIndent=0,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=6,
            spaceAfter=2,
        )
        styles["SectionHeaderTight"] = ParagraphStyle(
            name="SectionHeaderTight",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.black,
            leftIndent=-6,
            firstLineIndent=0,
            alignment=0,
            spaceBefore=6,
            spaceAfter=2,
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
            spaceAfter=2,
        )
        styles["SubSubHeader"] = ParagraphStyle(
            name="SubSubHeader",
            fontName="Helvetica-BoldOblique",
            fontSize=7,
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
        styles["SubSubHeaderItalic"] = ParagraphStyle(
            name="SubSubHeaderItalic",
            fontName="Helvetica-Oblique",
            fontSize=7,
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
        styles["IndentedBody"] = ParagraphStyle(
            name="IndentedBody",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=12,
            leading=7,
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
            fontSize=8,
            textColor=colors.black,
            spaceAfter=2,
        )
        styles["DisclaimerBody"] = ParagraphStyle(
            name="DisclaimerBody",
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=colors.HexColor("#e67e22"),
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

    def generate_report(self, result_data: dict, clinician_data: dict) -> bytes:
        logger.info("Generating PDF report for result %s", result_data.get("result_id"))

        report_data = self._build_report_data(result_data, clinician_data)

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
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 4))
        story.extend(self._build_referral_patient(report_data))
        story.extend(self._build_recording_assessment(report_data))
        story.extend(self._build_findings(report_data))
        story.extend(self._build_model_classification(report_data))
        story.append(Spacer(1, 1))
        story.append(self._build_summary_disclaimer_columns(report_data, doc.width))
        story.append(Spacer(1, 6))
        story.extend(self._build_signature_block(report_data))

        footer_text = report_data["footer_text"]
        doc.multiBuild(story, canvasmaker=make_footer_canvas(footer_text))

        pdf_content = buffer.getvalue()
        buffer.close()
        logger.info("PDF report generated successfully, size: %s bytes", len(pdf_content))
        return pdf_content

    def _build_report_data(self, result_data: dict, clinician_data: dict) -> dict:
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

        footer_text = (
            f"Subject Code: {subject_code}    Recording ID: {recording_id}    "
            f"{report_date}"
        )

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
            "patient_dob": _format_date(result_data.get("date_of_birth")),
            "patient_age_at_study": (
                f"{result_data.get('age')} years"
                if result_data.get("age") is not None
                else "Not recorded"
            ),
            "gender": _display(result_data.get("gender")),
            "study_id": result_id,
            "local_study_id": recording_id,
            "technician": clinician_name or "Not recorded",
            "start_datetime": "Not recorded",
            "stop_datetime": "Not recorded",
            "duration_minutes": "Not recorded",
            "recorded_minutes": "Not recorded",
            "eeg_type": "Standard EEG",
            "indication": "ADHD assessment",
            "medication": _display(result_data.get("medication_intake"), "None"),
            "alertness": "Not recorded",
            "sensor_group": "10-20 International, 19 channels",
            "band_power": band_power,
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
            "technician_name": "Not recorded",
            "clinician_name": clinician_name or "Not recorded",
            "clinician_occupation": clinician_occupation,
            "supervising_physician": "Not recorded",
            "footer_text": footer_text,
        }

    def _build_header(self, report_data: dict) -> list:
        elements: list = []
        left_block = [
            Paragraph("EEG REPORT", self.styles["ReportTitle"]),
            Paragraph(_display(report_data["unit_name"]), self.styles["BodyText"]),
            Paragraph(_display(report_data["institution_name"]), self.styles["BodyText"]),
        ]

        logo = report_data.get("logo_path")
        if logo:
            try:
                reader = ImageReader(logo)
                iw, ih = reader.getSize()
                max_h = 40
                scale = max_h / ih
                img = Image(logo, width=iw * scale, height=ih * scale)
                center_block = [img]
            except Exception:
                center_block = [Paragraph("HOSPITAL LOGO", self.styles["LogoPlaceholder"])]
        else:
            center_block = [Paragraph("HOSPITAL LOGO", self.styles["LogoPlaceholder"])]

        right_block = [
            Paragraph(f"Date: {report_data['report_date']}", self.styles["BodyText"])
        ]

        table = Table(
            [[left_block, center_block, right_block]],
            colWidths=[doc_width() * 0.5, doc_width() * 0.2, doc_width() * 0.3],
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 2))
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
            Paragraph("PATIENT - PERSONAL INFORMATION", self.styles["SectionHeader"]),
            Paragraph(
                f"Subject Code: {_display(report_data['patient_id'])}",
                self.styles["BodyText"],
            ),
            Paragraph(
                f"Date of Birth: {_display(report_data['patient_dob'])}",
                self.styles["BodyText"],
            ),
            Paragraph(
                f"Age at study time: {_display(report_data['patient_age_at_study'])}",
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
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LINEBEFORE", (1, 0), (1, 0), 0.5, colors.black),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 2))
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
                f"Start: {report_data['start_datetime']}   "
                f"Stop: {report_data['stop_datetime']}   "
                f"Duration: {report_data['duration_minutes']} minutes   "
                f"Recorded: {report_data['recorded_minutes']} minutes"
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
        elements.append(Spacer(1, 4))

        details = [
            ["Medication at referral", _display(report_data["medication"])],
            ["Alertness", _display(report_data["alertness"])],
            ["Sensor group", _display(report_data["sensor_group"])],
        ]
        details_table = Table(details, colWidths=[45 * mm, doc_width() - 45 * mm])
        details_table.setStyle(
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
        elements.append(details_table)

        elements.append(Paragraph("Relative Band Powers", self.styles["DetailLabel"]))
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
        elements.append(Spacer(1, 3))
        return elements

    def _build_findings(self, report_data: dict) -> list:
        elements: list = []
        elements.append(Paragraph("FINDINGS", self.styles["SectionHeaderTight"]))
        elements.append(Spacer(1, 1))

        elements.append(Paragraph("Spectral Findings", self.styles["SubHeader"]))
        for finding in generate_band_findings(report_data["band_power"]):
            elements.append(Paragraph(finding["label"], self.styles["SubSubHeader"]))
            elements.append(
                Paragraph(
                    f"Properties: {finding['properties']}",
                    self.styles["IndentedBody"],
                )
            )
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 1))

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
            Paragraph(classification_statement, self.styles["IndentedBody"])
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
        left_column = [
            Paragraph("SUMMARY OF FINDINGS", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["summary_findings"], self.styles, width / 2 - 6
            ),
            Spacer(1, 6),
            Paragraph("DIAGNOSTIC SIGNIFICANCE", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["diagnostic_significance"],
                self.styles,
                width / 2 - 6,
                bold=True,
            ),
            Spacer(1, 6),
            Paragraph("CLINICAL COMMENTS", self.styles["SectionHeader"]),
            build_shaded_content_box(
                report_data["clinical_comments"], self.styles, width / 2 - 6
            ),
        ]

        right_column = [
            Paragraph("IMPORTANT DISCLAIMER", self.styles["SectionHeader"]),
            build_disclaimer_box(self.styles, width / 2 - 6),
        ]

        table = Table([[left_column, right_column]], colWidths=[width / 2, width / 2])
        table.setStyle(
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
        return table

    def _build_signature_block(self, report_data: dict) -> list:
        elements: list = []
        signature_line = "________________________"
        clinician_name = report_data.get("clinician_name", "Not recorded")
        clinician_occupation = report_data.get("clinician_occupation", "")
        clinician_display = (
            f"{clinician_name}<br/>{clinician_occupation}"
            if clinician_occupation and clinician_occupation != "Not recorded"
            else clinician_name
        )
        rows = [
            [signature_line, signature_line, signature_line],
            [
                report_data["technician_name"],
                Paragraph(clinician_display, self.styles["BodyText"]),
                report_data["supervising_physician"],
            ],
            ["Technician", "Clinician", "Supervising clinician"],
        ]
        table = Table(rows, colWidths=[doc_width() / 3] * 3)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
                    ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(table)
        return elements


def doc_width() -> float:
    return A4[0] - (18 * mm) * 2
