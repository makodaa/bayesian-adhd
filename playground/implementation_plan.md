# PDF Report Generation for EEG Analysis Results

This implementation plan outlines the approach to add PDF report generation for EEG analysis results, accessible via the results management page.

## Requirements Summary

- Generate professional, clinical-quality PDF reports from EEG analysis results
- Include appropriate clinical disclaimers and non-definitive diagnosis language
- Add notes for missing sleep/food/medication data that may affect clinical ratios
- Structure results using clinical language (e.g., "consistent with ADHD", "may support clinical suspicion")
- Add a prominent warning that results are not a definitive diagnosis
- Professional appearance: no emojis, subdued color palette

---

## Proposed Changes

### PDF Generation Service

#### [NEW] [pdf_service.py](file:///Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend/app/services/pdf_service.py)

Create a new service for PDF generation using **ReportLab** library:

- `PDFReportService` class with methods:
  - `generate_report(result_id: int, clinician_id: int) -> bytes`: Generate PDF report for a result
  - `_build_header_section()`: Clinic header with date and report title
  - `_build_subject_section()`: Subject demographics (code, age, gender)
  - `_build_recording_section()`: Recording metadata (sleep, caffeine, medication, etc.)
  - `_build_classification_section()`: Prediction result with clinical interpretation language
  - `_build_band_powers_section()`: Relative band power analysis with bar chart
  - `_build_ratios_section()`: Clinical ratios (theta/beta, etc.) with reference ranges
  - `_build_notes_section()`: Notes for missing or concerning data affecting ratios
  - `_build_disclaimers_section()`: Legal/medical disclaimers

**Classification Language Mapping:**
| Prediction | Clinical Interpretation |
|------------|-------------------------|
| ADHD 1/2/3 | "EEG analysis may support clinical suspicion of ADHD" |
| Non-ADHD | "EEG patterns do not suggest findings typically associated with ADHD" |

**Confidence Score Interpretation:**
- High (>80%): "findings strongly support clinical suspicion of"
- Moderate (60-80%): "may support clinical suspicion of"
- Lower (<60%): "findings should be interpreted with caution; may be suggestive of"

---

### Backend API Endpoint

#### [MODIFY] [main.py](file:///Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend/app/main.py)

Add a new API endpoint for generating and downloading PDF reports:

```python
@app.route('/api/results/<int:result_id>/pdf', methods=['GET'])
@login_required
def generate_result_pdf(result_id):
    """Generate PDF report for a result."""
```

- Fetch result data using `ResultsService.get_result_with_full_details()`
- Generate PDF using `PDFReportService.generate_report()`
- Return PDF as file download with appropriate headers
- Optionally save report metadata to `reports` table

---

### Report Service Update

#### [MODIFY] [report_service.py](file:///Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend/app/services/report_service.py)

Complete the stub implementation to integrate with `PDFReportService`:

- Implement `generate_report()` method to orchestrate PDF creation
- Implement `_generate_summary()` for report text summaries
- Implement `_generate_interpretation()` for clinical interpretation logic

---

### Frontend - Results Page

#### [MODIFY] [results.html](file:///Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend/app/templates/results.html)

Add PDF download button to the results modal:

- Add a "Download PDF Report" button in the modal footer
- Button triggers download of PDF via `/api/results/{id}/pdf`
- Add loading indicator during PDF generation

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend/requirements.txt)

Add PDF generation library:

```
reportlab
```

---

## PDF Report Structure

The generated PDF will include the following sections:

### 1. Header
- Report title: "EEG Analysis Report"
- Generation date/time
- Report ID/reference number

### 2. Subject Information
- Subject Code
- Age
- Gender

### 3. Recording Environment
- Analysis Date
- Sleep Hours (with note if missing or < 4 hours)
- Food Intake (with note if missing)
- Caffeine Status (with note if caffeinated)
- Medication Status (with note if medicated)

### 4. Classification Result
- Prediction result with clinical language
- Confidence score with interpretation
- Visual indicator (professional bar chart)

### 5. Band Power Analysis
- Bar chart of relative band powers (delta, theta, alpha, beta, gamma)
- Values table with percentages

### 6. Clinical Ratios
- Theta/Beta ratio with clinical reference
- Other computed ratios
- Notes on data quality affecting ratios

### 7. Data Quality Notes
- Warnings for missing sleep data
- Warnings for caffeine/medication affecting readings
- Notes on data completeness

### 8. Signatory Section
- **Prepared by**: [Logged-in clinician name and occupation]
- **Acknowledged by**: _________________________ (signature line for secondary review)

### 9. Disclaimers (Footer Section)
> **Important Notice:** This report presents results from an automated EEG analysis system and does not constitute a definitive clinical diagnosis. These findings should be interpreted by a qualified healthcare professional in conjunction with comprehensive clinical evaluation, patient history, and other diagnostic criteria.

---

## Approved Decisions

- **Clinical Interpretation Language**: Use "EEG analysis may support clinical suspicion of ADHD" as primary wording
- **PDF Library**: ReportLab
- **Signatory Section**: Include "Prepared by" (logged-in clinician) and "Acknowledged by" (blank signature line)

---

## Verification Plan

### Manual Testing

Since there are no existing automated tests in this project, verification will be manual:

1. **Start the Backend Server**
   ```bash
   cd /Users/mackenziecerenio/Developer/Python/bayesian-adhd/backend
   python -m app.main
   ```

2. **Navigate to Results Page**
   - Open browser to `http://localhost:5000/results.html`
   - Login with clinician credentials

3. **Test PDF Generation**
   - Click on a result row to open the modal
   - Click "Download PDF Report" button
   - Verify PDF downloads successfully
   - Verify PDF contains all required sections
   - Verify professional appearance (no emojis, subdued colors)
   - Verify disclaimers are present and prominently displayed

4. **Test Edge Cases**
   - Test with result that has missing sleep data
   - Test with result that has missing medication data
   - Test with ADHD classification result
   - Test with Non-ADHD classification result
   - Test with high confidence score
   - Test with low confidence score

### Content Verification Checklist

- [ ] Header section present with date and title
- [ ] Subject information correctly displayed
- [ ] Recording environment data shown with appropriate notes
- [ ] Classification uses non-definitive language
- [ ] Band power chart is professional and readable
- [ ] Clinical ratios displayed with references
- [ ] Missing data notes are generated correctly
- [ ] Disclaimer is prominent and clearly visible
- [ ] No emojis present anywhere in document
- [ ] Color palette is professional and subdued
