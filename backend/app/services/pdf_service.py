"""PDF Report Generation Service for EEG Analysis Results."""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Line, Rect, String
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


def generate_band_findings(band_power: dict[str, float]) -> list[str]:
    findings: list[str] = []
    alpha = band_power.get("alpha", 0.0)
    beta = band_power.get("beta", 0.0)
    theta = band_power.get("theta", 0.0)
    tbr = theta / beta if beta > 0 else 0.0

    if alpha > 12:
        label = f"Increased Alpha activity ({alpha:.2f}%)"
        note = (
            "May reflect enhanced relaxed wakefulness, reduced cognitive engagement, "
            "or inattentive state consistent with some ADHD presentations."
        )
    elif alpha < 8:
        label = f"Decreased Alpha activity ({alpha:.2f}%)"
        note = (
            "Reduced posterior alpha; may indicate heightened cortical arousal "
            "or hyperactive/impulsive state."
        )
    else:
        label = f"Alpha activity within normal limits ({alpha:.2f}%)"
        note = "Posterior alpha rhythm is appropriate for age and vigilance state."
    findings.append(f"Alpha (8-13 Hz): {label}. {note}")

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
    findings.append(f"Beta (13-30 Hz): {label}. {note}")

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
    findings.append(f"Theta (4-8 Hz): {label}. {note}")

    if tbr > 3.0:
        tbr_label = f"Elevated Theta/Beta Ratio ({tbr:.2f})"
        tbr_note = (
            "TBR elevation has been associated with ADHD in research literature; "
            "interpret alongside clinical presentation."
        )
    elif tbr >= 2.0:
        tbr_label = f"Theta/Beta Ratio borderline elevated ({tbr:.2f})"
        tbr_note = "Mildly elevated; monitor in clinical context."
    else:
        tbr_label = f"Theta/Beta Ratio within normal limits ({tbr:.2f})"
        tbr_note = "Theta/Beta ratio does not indicate excessive slow-wave dominance."
    findings.append(f"Theta/Beta Ratio: {tbr_label}. {tbr_note}")

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
    findings.append(f"Cortical Arousal: {arousal} (Combined Alpha+Beta: {ab:.2f}%)")

    return findings


def build_band_power_chart(band_power: dict[str, float], width: float, height: float) -> Drawing:
    bands = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-13 Hz)",
        "Beta (13-30 Hz)",
        "Gamma (30-100 Hz)",
    ]
    keys = ["delta", "theta", "alpha", "beta", "gamma"]
    colors_map = ["#111111", "#444444", "#777777", "#aaaaaa", "#cccccc"]
    values = [_clean_percent(band_power.get(k, 0.0)) for k in keys]

    bar_h = 10
    bar_spacing = 6
    label_w = 130
    chart_w = width - label_w - 40
    y_start = height - 10
    axis_max = 80

    d = Drawing(width, height)
    for i, (band, val, color) in enumerate(zip(bands, values, colors_map)):
        y = y_start - i * (bar_h + bar_spacing)
        bar_w = (min(val, axis_max) / axis_max) * chart_w
        d.add(String(0, y + 2, band, fontSize=6.5, fontName="Helvetica"))
        rect = Rect(label_w, y, bar_w, bar_h)
        rect.fillColor = colors.HexColor(color)
        rect.strokeColor = colors.HexColor(color)
        rect.strokeWidth = 0
        d.add(rect)
        d.add(
            String(
                label_w + bar_w + 3,
                y + 2,
                f"{val:.1f}%",
                fontSize=6.5,
                fontName="Helvetica",
            )
        )

    axis_y = 0
    d.add(Line(label_w, axis_y, width - 40, axis_y, strokeWidth=0.5))
    for tick in (0, 17, 34, 51, 68):
        x = label_w + (tick / axis_max) * chart_w
        d.add(Line(x, axis_y, x, axis_y - 3, strokeWidth=0.5))
        d.add(String(x - 6, axis_y - 12, f"{tick}%", fontSize=6, fontName="Helvetica"))

    return d


def _build_disclaimer_summary_box(
    summary: str,
    significance: str,
    comments: str,
    styles: dict[str, ParagraphStyle],
    content_width: float,
) -> Table:
    amber_border = colors.HexColor("#e6a817")
    amber_bg = colors.HexColor("#fffbe6")

    disclaimer_text = (
        "This report is generated by a computational EEG analysis tool and is intended "
        "to support - not replace - clinical evaluation. The findings, classifications, "
        "and comments presented here do not constitute a definitive medical diagnosis. "
        "All results must be interpreted by a qualified healthcare professional in the "
        "context of the patient's full clinical history and presentation."
    )

    inner = [
        Paragraph("[!]  IMPORTANT DISCLAIMER", styles["DisclaimerHeader"]),
        Spacer(1, 4),
        Paragraph(disclaimer_text, styles["DisclaimerBody"]),
        Spacer(1, 6),
        Paragraph("SUMMARY OF FINDINGS", styles["SectionHeaderAmber"]),
        Paragraph(summary, styles["ShadedBoxText"]),
        Spacer(1, 4),
        Paragraph("DIAGNOSTIC SIGNIFICANCE", styles["SectionHeaderAmber"]),
        Paragraph(significance, styles["ShadedBoxText"]),
        Spacer(1, 4),
        Paragraph("CLINICAL COMMENTS", styles["SectionHeaderAmber"]),
        Paragraph(comments, styles["ShadedBoxText"]),
    ]

    table = Table([[inner]], colWidths=[content_width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), amber_bg),
                ("BOX", (0, 0), (-1, -1), 1.5, amber_border),
                ("LINEBEFORE", (0, 0), (-1, -1), 4, amber_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
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
            spaceBefore=6,
            spaceAfter=2,
        )
        styles["SectionHeaderAmber"] = ParagraphStyle(
            name="SectionHeaderAmber",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.HexColor("#7a4f00"),
            spaceBefore=2,
            spaceAfter=2,
        )
        styles["SubHeader"] = ParagraphStyle(
            name="SubHeader",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=2,
        )
        styles["SubSubHeader"] = ParagraphStyle(
            name="SubSubHeader",
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=2,
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
            leading=9.5,
            spaceAfter=2,
        )
        styles["IndentedBody"] = ParagraphStyle(
            name="IndentedBody",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leftIndent=10,
            leading=9.5,
            spaceAfter=2,
        )
        styles["ShadedBoxText"] = ParagraphStyle(
            name="ShadedBoxText",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.black,
            leading=9.5,
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
            textColor=colors.HexColor("#7a4f00"),
            spaceAfter=2,
        )
        styles["DisclaimerBody"] = ParagraphStyle(
            name="DisclaimerBody",
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=colors.HexColor("#5a3a00"),
            leading=9,
            spaceAfter=2,
        )
        styles["ClassificationLabel"] = ParagraphStyle(
            name="ClassificationLabel",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#1a3a5c"),
            spaceAfter=2,
        )
        styles["ClassificationValue"] = ParagraphStyle(
            name="ClassificationValue",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.HexColor("#0d2b00"),
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
        story.append(self._build_model_classification(report_data))
        story.append(Spacer(1, 6))
        story.append(
            _build_disclaimer_summary_box(
                report_data["summary_findings"],
                report_data["diagnostic_significance"],
                report_data["clinical_comments"],
                self.styles,
                doc.width,
            )
        )
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
            "technician_name": clinician_name or "Not recorded",
            "physician_name": "Not recorded",
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
        elements.append(Spacer(1, 6))
        return elements

    def _build_recording_assessment(self, report_data: dict) -> list:
        elements: list = []

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
            ["EEG type", _display(report_data["eeg_type"])],
            ["Indication for EEG", _display(report_data["indication"])],
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
        elements.append(Spacer(1, 8))
        return elements

    def _build_findings(self, report_data: dict) -> list:
        elements: list = []
        elements.append(Paragraph("FINDINGS", self.styles["SectionHeader"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
        elements.append(Spacer(1, 4))

        elements.append(
            Paragraph("Relative Band Power Analysis", self.styles["SubHeader"])
        )
        elements.append(
            Paragraph(
                "Averaged Relative Band Power Across All Channels",
                self.styles["SubSubHeaderItalic"],
            )
        )
        elements.append(Spacer(1, 4))

        chart = build_band_power_chart(report_data["band_power"], doc_width() * 0.6, 90)
        elements.append(chart)
        elements.append(Spacer(1, 6))

        elements.append(self._build_band_power_table(report_data["band_power"]))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Spectral Findings", self.styles["SubSubHeader"]))
        for finding in generate_band_findings(report_data["band_power"]):
            elements.append(Paragraph(finding, self.styles["BodyText"]))
        elements.append(Spacer(1, 6))

        return elements

    def _build_band_power_table(self, band_power: dict) -> Table:
        freq_ranges = {
            "delta": "0.5-4 Hz",
            "theta": "4-8 Hz",
            "alpha": "8-13 Hz",
            "beta": "13-30 Hz",
            "gamma": "30-100 Hz",
        }
        rows = [["Band", "Frequency Range", "Relative Power"]]
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            rows.append(
                [
                    band.capitalize(),
                    freq_ranges[band],
                    f"{band_power.get(band, 0.0):.2f}%",
                ]
            )
        table = Table(rows, colWidths=[35 * mm, 45 * mm, 35 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        return table

    def _build_model_classification(self, report_data: dict) -> Table:
        model = report_data["model_classification"]
        confidence = model["confidence"]
        confidence_pct = confidence * 100

        left_data = [
            [
                Paragraph("Classification:", self.styles["ClassificationLabel"]),
                Paragraph(model["label"], self.styles["ClassificationValue"]),
            ],
            [
                Paragraph("Confidence:", self.styles["BodyText"]),
                Paragraph(f"{confidence_pct:.1f}%", self.styles["BodyText"]),
            ],
            [
                Paragraph("Theta/Beta Ratio:", self.styles["BodyText"]),
                Paragraph(f"{model['theta_beta_ratio']:.2f}", self.styles["BodyText"]),
            ],
            [
                Paragraph("Model Version:", self.styles["BodyText"]),
                Paragraph(model["model_version"], self.styles["BodyText"]),
            ],
        ]
        left_table = Table(left_data, colWidths=[40 * mm, doc_width() * 0.6 - 40 * mm])
        left_table.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )

        bar = _confidence_bar(confidence, doc_width() * 0.4 - 8)
        right_stack = [
            Paragraph("Model Confidence", self.styles["BodyText"]),
            Spacer(1, 4),
            bar,
            Spacer(1, 4),
            Paragraph(f"{confidence_pct:.1f}%", self.styles["BodyText"]),
        ]

        rows = [[left_table, right_stack]]
        if model.get("notes"):
            notes_flow = [
                Paragraph("Notes:", self.styles["SubSubHeader"]),
                Paragraph(model["notes"], self.styles["IndentedBody"]),
            ]
            rows.append([notes_flow, ""])

        table = Table(rows, colWidths=[doc_width() * 0.6, doc_width() * 0.4])
        table_style = [
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf4ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2a6099")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        if model.get("notes"):
            table_style.append(("SPAN", (0, 1), (1, 1)))
        table.setStyle(TableStyle(table_style))
        return table

    def _build_signature_block(self, report_data: dict) -> list:
        elements: list = []
        rows = [
            [
                report_data["technician_name"],
                report_data["physician_name"],
                report_data["supervising_physician"],
            ],
            ["Technician", "Physician (signed)", "Supervising physician"],
        ]
        table = Table(rows, colWidths=[doc_width() / 3] * 3)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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
