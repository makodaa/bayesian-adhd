"""PDF Report Generation Service for EEG Analysis Results."""

from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

# Color palette - professional, subdued colors
COLORS = {
    'primary': colors.HexColor('#5a6c7d'),
    'secondary': colors.HexColor('#6c9bd1'),
    'text': colors.HexColor('#333333'),
    'light_gray': colors.HexColor('#f5f7fa'),
    'border': colors.HexColor('#e0e0e0'),
    'warning': colors.HexColor('#f0ad4e'),
    'success': colors.HexColor('#5cb85c'),
}

# Band power colors - muted versions
BAND_COLORS = {
    'delta': colors.HexColor('#7B1FA2'),  # Muted purple
    'theta': colors.HexColor('#1976D2'),  # Muted blue
    'alpha': colors.HexColor('#388E3C'),  # Muted green
    'beta': colors.HexColor('#F57C00'),   # Muted orange
    'gamma': colors.HexColor('#D32F2F'),  # Muted red
}


class PDFReportService:
    """Generate professional clinical PDF reports for EEG analysis results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style - left aligned for professional look
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.black,
            spaceAfter=3,
            spaceBefore=0,
            alignment=0,  # Left align
            leftIndent=0,
        ))
        
        # Section header style - no left indent to align with text
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=4,
            leftIndent=0,
            firstLineIndent=0,
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='Subsection',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=3,
            leftIndent=0,
        ))
        
        # Normal text with compact spacing
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=2,
            leading=12,
            leftIndent=0,
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=4,
            leading=11,
            leftIndent=0,
        ))
        
        # Classification result style
        self.styles.add(ParagraphStyle(
            name='Classification',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.black,
            alignment=0,  # Left align
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=0,
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=1,  # Center
        ))
    
    def generate_report(self, result_data: dict, clinician_data: dict) -> bytes:
        """
        Generate a PDF report for the given result data.
        
        Args:
            result_data: Dictionary containing full result details from 
                        ResultsService.get_result_with_full_details()
            clinician_data: Dictionary containing current clinician info
        
        Returns:
            bytes: PDF file content
        """
        logger.info(f"Generating PDF report for result {result_data.get('result_id')}")
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=25*mm,
        )
        
        # Build story (list of flowable elements)
        story = []
        
        # Build each section
        story.extend(self._build_header_section(result_data))
        story.extend(self._build_disclaimer_section())
        story.extend(self._build_subject_section(result_data))
        story.extend(self._build_recording_section(result_data))
        story.extend(self._build_classification_section(result_data))
        story.extend(self._build_band_powers_section(result_data))
        story.extend(self._build_ratios_section(result_data))
        story.extend(self._build_data_quality_notes(result_data))
        story.extend(self._build_signatory_section(clinician_data))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        pdf_content = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF report generated successfully, size: {len(pdf_content)} bytes")
        return pdf_content
    
    def _add_page_number(self, canvas, doc):
        """Add page number to footer."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.gray)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawCentredString(A4[0]/2, 15*mm, text)
        canvas.restoreState()
    
    def _build_header_section(self, result_data: dict) -> list:
        """Build the report header section."""
        elements = []
        
        # Title - left aligned
        elements.append(Paragraph("EEG Analysis Report", self.styles['ReportTitle']))
        
        # Subtitle with date - left aligned
        analysis_date = result_data.get('inferenced_at')
        if analysis_date:
            if isinstance(analysis_date, str):
                date_str = analysis_date[:10]
            else:
                date_str = analysis_date.strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Left-aligned report info as table
        info_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Analysis Date:', date_str],
            ['Reference:', f"R-{result_data.get('result_id', 'N/A')}"],
        ]
        
        info_table = Table(info_data, colWidths=[40*mm, 130*mm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#666666')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['border']))
        elements.append(Spacer(1, 6))
        
        return elements
    
    def _build_disclaimer_section(self) -> list:
        """Build the prominent disclaimer section at the top."""
        elements = []
        
        disclaimer_text = """
        <b>Important Notice:</b> This report presents results from an automated EEG analysis 
        system and does not constitute a definitive clinical diagnosis. These findings should 
        be interpreted by a qualified healthcare professional in conjunction with comprehensive 
        clinical evaluation, patient history, and other diagnostic criteria. This tool is 
        intended to support, not replace, clinical judgment.
        """
        
        # Create a table for the disclaimer box
        disclaimer_para = Paragraph(disclaimer_text.strip(), self.styles['ReportBody'])
        disclaimer_table = Table([[disclaimer_para]], colWidths=[170*mm])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff8e1')),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['warning']),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(disclaimer_table)
        elements.append(Spacer(1, 8))
        
        return elements
    
    def _build_subject_section(self, result_data: dict) -> list:
        """Build subject information section."""
        elements = []
        elements.append(Paragraph("Subject Information", self.styles['SectionHeader']))
        
        data = [
            ['Subject Code:', result_data.get('subject_code', 'N/A')],
            ['Age:', f"{result_data.get('age', 'N/A')} years"],
            ['Gender:', result_data.get('gender', 'N/A')],
        ]
        
        # Consistent table width: 170mm (50mm label + 120mm value)
        table = Table(data, colWidths=[50*mm, 120*mm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 8))
        
        return elements
    
    def _build_recording_section(self, result_data: dict) -> list:
        """Build recording environment section."""
        elements = []
        elements.append(Paragraph("Recording Environment", self.styles['SectionHeader']))
        
        # Build data rows
        data = [
            ['File Name:', result_data.get('file_name', 'N/A')],
        ]
        
        if result_data.get('sleep_hours') is not None:
            data.append(['Sleep Hours:', f"{result_data.get('sleep_hours')} hours"])
        
        if result_data.get('food_intake'):
            data.append(['Food Intake:', result_data.get('food_intake')])
        
        if result_data.get('caffeinated') is not None:
            data.append(['Caffeinated:', 'Yes' if result_data.get('caffeinated') else 'No'])
        
        if result_data.get('medicated') is not None:
            data.append(['Medicated:', 'Yes' if result_data.get('medicated') else 'No'])
        
        if result_data.get('medication_intake'):
            data.append(['Medication Details:', result_data.get('medication_intake')])
        
        if result_data.get('notes'):
            data.append(['Notes:', result_data.get('notes')])
        
        # Consistent table width: 170mm (50mm label + 120mm value)
        table = Table(data, colWidths=[50*mm, 120*mm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 8))
        
        return elements
    
    def _build_classification_section(self, result_data: dict) -> list:
        """Build classification result section with clinical interpretation."""
        elements = []
        elements.append(Paragraph("Classification Result", self.styles['SectionHeader']))
        
        predicted_class = result_data.get('predicted_class', 'Unknown')
        confidence = result_data.get('confidence_score', 0)
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        
        # Determine interpretation based on classification and confidence
        interpretation = self._get_clinical_interpretation(predicted_class, confidence_pct)
        
        # Create result display - left aligned professional table (black/white)
        result_data_table = [
            ['Classification:', predicted_class],
            ['Confidence Score:', f"{confidence_pct:.1f}%"],
        ]
        
        # Consistent table width: 170mm (50mm label + 120mm value)
        result_table = Table(result_data_table, colWidths=[50*mm, 120*mm])
        result_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ]))
        
        elements.append(result_table)
        elements.append(Spacer(1, 6))
        
        # Clinical interpretation
        elements.append(Paragraph("<b>Clinical Interpretation:</b>", self.styles['ReportBody']))
        elements.append(Paragraph(interpretation, self.styles['ReportBody']))
        elements.append(Spacer(1, 8))
        
        return elements
    
    def _get_clinical_interpretation(self, predicted_class: str, confidence_pct: float) -> str:
        """Generate clinical interpretation text based on prediction and confidence."""
        # Determine ADHD subtype
        adhd_subtypes = {
            'Combined / C (ADHD-C)': 'ADHD Combined Presentation (ADHD-C)',
            'Hyperactive-Impulsive (ADHD-H)': 'ADHD Hyperactive-Impulsive Presentation (ADHD-H)',
            'Inattentive (ADHD-I)': 'ADHD Inattentive Presentation (ADHD-I)',
        }
        
        is_adhd = predicted_class in adhd_subtypes
        subtype_description = adhd_subtypes.get(predicted_class, '')
        
        if is_adhd:
            if confidence_pct >= 80:
                return (f"The EEG analysis findings strongly support clinical suspicion of {subtype_description}. "
                       "The observed patterns are consistent with those typically associated with "
                       "this ADHD presentation. Further clinical evaluation is recommended "
                       "to confirm diagnosis.")
            elif confidence_pct >= 60:
                return (f"The EEG analysis may support clinical suspicion of {subtype_description}. "
                       "The observed brainwave patterns show characteristics that are sometimes "
                       "associated with this ADHD presentation. Comprehensive clinical "
                       "assessment is advised.")
            else:
                return (f"The EEG analysis findings should be interpreted with caution; "
                       f"patterns may be suggestive of {subtype_description}. Due to the lower confidence level, "
                       "additional testing and thorough clinical evaluation are strongly recommended.")
        else:
            if confidence_pct >= 80:
                return ("EEG patterns do not suggest findings typically associated with ADHD. "
                       "The analysis indicates brainwave activity within typical ranges. "
                       "However, this does not rule out ADHD, as clinical diagnosis requires "
                       "comprehensive evaluation.")
            elif confidence_pct >= 60:
                return ("EEG patterns do not strongly indicate characteristics typically associated with ADHD. "
                       "The findings may suggest typical brainwave activity, though clinical correlation "
                       "is recommended.")
            else:
                return ("The EEG analysis results are inconclusive. The patterns observed do not "
                       "clearly indicate either ADHD or non-ADHD characteristics. Further evaluation "
                       "and additional testing are recommended.")
    
    def _build_band_powers_section(self, result_data: dict) -> list:
        """Build relative band power analysis section with professional bar chart."""
        elements = []
        elements.append(Paragraph("Relative Band Power Analysis", self.styles['SectionHeader']))
        
        band_powers = result_data.get('band_powers', [])
        if not band_powers:
            elements.append(Paragraph("No band power data available.", self.styles['ReportBody']))
            return elements
        
        # Calculate average band powers across electrodes
        avg_powers = {}
        for bp in band_powers:
            band = bp.get('frequency_band', '').lower()
            power = bp.get('relative_power', 0)
            if band not in avg_powers:
                avg_powers[band] = []
            avg_powers[band].append(power)
        
        # Average the values
        for band in avg_powers:
            avg_powers[band] = sum(avg_powers[band]) / len(avg_powers[band])
        
        # Band order and labels
        band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_labels = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 
                       'Beta (13-30 Hz)', 'Gamma (30-100 Hz)']
        
        # Prepare data for chart
        chart_data = []
        labels = []
        
        for band, label in zip(band_order, band_labels):
            if band in avg_powers:
                chart_data.append(avg_powers[band] * 100)  # Convert to percentage
                labels.append(label)
        
        if chart_data:
            # Chart dimensions - consistent with page margins
            chart_height = len(chart_data) * 22
            drawing = Drawing(170*mm, chart_height + 20)
            
            chart = HorizontalBarChart()
            chart.x = 50*mm  # Left margin for labels
            chart.y = 10
            chart.width = 110*mm
            chart.height = chart_height
            
            # Data - reverse for top-to-bottom display (Delta at top)
            chart.data = [list(reversed(chart_data))]
            
            # Black/white professional styling - all bars dark gray
            bar_color = colors.HexColor('#4a4a4a')
            for i in range(len(chart_data)):
                chart.bars[(0, i)].fillColor = bar_color
                chart.bars[(0, i)].strokeColor = colors.black
                chart.bars[(0, i)].strokeWidth = 0.5
            
            # Configure category axis (Y-axis - band names)
            chart.categoryAxis.categoryNames = list(reversed(labels))
            chart.categoryAxis.labels.fontName = 'Helvetica'
            chart.categoryAxis.labels.fontSize = 9
            chart.categoryAxis.labels.dx = -5
            chart.categoryAxis.labels.textAnchor = 'end'
            chart.categoryAxis.strokeWidth = 0.5
            chart.categoryAxis.strokeColor = colors.black
            chart.categoryAxis.visibleTicks = 0
            
            # Configure value axis (X-axis - percentages)
            max_val = max(chart_data)
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max_val * 1.2  # 20% headroom for labels
            chart.valueAxis.valueStep = max_val / 4
            chart.valueAxis.labels.fontName = 'Helvetica'
            chart.valueAxis.labels.fontSize = 8
            chart.valueAxis.labelTextFormat = '%.0f%%'
            chart.valueAxis.strokeWidth = 0.5
            chart.valueAxis.strokeColor = colors.black
            chart.valueAxis.visibleGrid = True
            chart.valueAxis.gridStrokeColor = colors.HexColor('#e0e0e0')
            chart.valueAxis.gridStrokeWidth = 0.5
            
            # Bar styling - no gaps between bars
            chart.barWidth = 16
            chart.barSpacing = 0
            chart.groupSpacing = 0
            
            drawing.add(chart)
            
            # Add value labels at end of each bar
            for i, val in enumerate(reversed(chart_data)):
                bar_y = 10 + (i * (chart_height / len(chart_data))) + (chart_height / len(chart_data) / 2) - 3
                bar_x = chart.x + (val / (max_val * 1.2)) * chart.width + 4
                val_label = String(bar_x, bar_y, f'{val:.1f}%')
                val_label.fontName = 'Helvetica'
                val_label.fontSize = 8
                val_label.fillColor = colors.black
                drawing.add(val_label)
            
            elements.append(drawing)
            elements.append(PageBreak())
            
            # Add summary table below chart - same width as other tables
            summary_data = [['Band', 'Frequency Range', 'Relative Power']]
            freq_ranges = {
                'delta': '0.5-4 Hz',
                'theta': '4-8 Hz', 
                'alpha': '8-13 Hz',
                'beta': '13-30 Hz',
                'gamma': '30-100 Hz'
            }
            
            for band in band_order:
                if band in avg_powers:
                    pct = avg_powers[band] * 100
                    summary_data.append([band.capitalize(), freq_ranges.get(band, ''), f"{pct:.2f}%"])
            
            # Use consistent column widths matching page width
            summary_table = Table(summary_data, colWidths=[50*mm, 60*mm, 60*mm])
            summary_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(summary_table)
        
        elements.append(Spacer(1, 8))
        return elements
    
    def _build_ratios_section(self, result_data: dict) -> list:
        """Build clinical ratios section."""
        elements = []
        elements.append(Paragraph("Clinical Ratios", self.styles['SectionHeader']))
        
        ratios = result_data.get('ratios', [])
        if not ratios:
            elements.append(Paragraph("No ratio data available.", self.styles['ReportBody']))
            return elements
        
        # Create table with ratios - consistent 170mm width
        data = [['Ratio Name', 'Value']]
        
        for ratio in ratios:
            name = ratio.get('ratio_name', '').replace('_', ' ').title()
            value = ratio.get('ratio_value', 0)
            data.append([name, f"{value:.3f}"])
        
        table = Table(data, colWidths=[110*mm, 60*mm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 8))
        
        # Add note about theta/beta ratio if applicable
        theta_beta = next((r for r in ratios if 'theta' in r.get('ratio_name', '').lower() 
                          and 'beta' in r.get('ratio_name', '').lower()), None)
        if theta_beta:
            note = ("Note: The theta/beta ratio is commonly studied in ADHD research. "
                   "Elevated values may be observed in some individuals with ADHD, though "
                   "this ratio alone is not diagnostic.")
            elements.append(Paragraph(f"<i>{note}</i>", self.styles['ReportBody']))
        
        elements.append(Spacer(1, 8))
        return elements
    
    def _build_data_quality_notes(self, result_data: dict) -> list:
        """Build data quality notes section based on recording conditions."""
        elements = []
        notes = []
        
        # Check sleep hours
        sleep_hours = result_data.get('sleep_hours')
        if sleep_hours is None:
            notes.append("Sleep hours were not recorded. Sleep deprivation can affect EEG patterns and may influence analysis results.")
        elif float(sleep_hours) < 4:
            notes.append(f"Subject reported only {sleep_hours} hours of sleep. Significant sleep deprivation may affect EEG recordings and could influence the analysis.")
        
        # Check food intake
        if not result_data.get('food_intake'):
            notes.append("Food intake information was not recorded. Dietary factors may influence EEG patterns.")
        
        # Check caffeine
        if result_data.get('caffeinated') is True:
            notes.append("Subject reported recent caffeine intake. Caffeine is a stimulant that can alter EEG patterns and may affect the analysis.")
        elif result_data.get('caffeinated') is None:
            notes.append("Caffeine intake status was not recorded.")
        
        # Check medication
        if result_data.get('medicated') is True:
            med_details = result_data.get('medication_intake', 'unspecified medication')
            notes.append(f"Subject was medicated ({med_details}). Medication can significantly impact EEG recordings and should be considered when interpreting results.")
        elif result_data.get('medicated') is None:
            notes.append("Medication status was not recorded.")
        
        if notes:
            elements.append(Paragraph("Data Quality Notes", self.styles['SectionHeader']))
            for note in notes:
                bullet = f"<bullet>&bull;</bullet> {note}"
                elements.append(Paragraph(bullet, self.styles['ReportBody']))
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_signatory_section(self, clinician_data: dict) -> list:
        """Build signatory section with clinician information."""
        elements = []
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['border']))
        elements.append(Spacer(1, 15))
        
        # Build clinician name
        first_name = clinician_data.get('first_name', '')
        middle_name = clinician_data.get('middle_name', '')
        last_name = clinician_data.get('last_name', '')
        occupation = clinician_data.get('occupation', '')
        
        full_name = ' '.join(filter(None, [first_name, middle_name, last_name]))
        if not full_name:
            full_name = 'N/A'
        
        clinician_info = full_name
        if occupation:
            clinician_info += f", {occupation}"
        
        # Create two-column signatory table
        sig_data = [
            ['Prepared by:', 'Acknowledged by:'],
            [clinician_info, '_________________________'],
            [f"Date: {datetime.now().strftime('%Y-%m-%d')}", 'Signature'],
        ]
        
        sig_table = Table(sig_data, colWidths=[85*mm, 85*mm])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['primary']),
            ('TEXTCOLOR', (0, 1), (-1, -1), COLORS['text']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(sig_table)
        
        return elements
