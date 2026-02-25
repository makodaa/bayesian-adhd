# Bayesian ADHD — Application Design Document

> This document describes the composition, intended use, experiential feel, and visual style of the Bayesian ADHD clinical decision-support application. It is intended to guide front-end development, UI/UX decisions, and stakeholder communication during Thesis 2 implementation.

---

## 1. Application Identity

**Full Name:** Bayesian ADHD — EEG-Based ADHD Classification System  
**Short Name:** BayesADHD  
**Tagline:** *Objective EEG insights for informed clinical decisions.*  
**Audience:** Clinicians — developmental pediatricians, psychiatrists, neurologists, and trained EEG technicians working in Philippine clinical settings.  
**Nature:** A clinical *support* tool. Not a standalone diagnostic instrument. The application's language, interface copy, and report framing must consistently reinforce this distinction.

---

## 2. Compositional Overview

The application is organized around a single, continuous clinical workflow: from subject registration, through EEG upload and processing, to results interpretation and report export. The interface does not expose the underlying model or signal processing as independent modules — these are invisible infrastructure. What the clinician sees is a streamlined, task-oriented surface.

### 2.1 Top-Level Navigation

The sidebar is the application's primary navigation element. It is narrow, icon-first, and persistent across all views. It contains exactly four destinations:

| Icon | Destination | Purpose |
|---|---|---|
| Dashboard | Home | Summary of recent activity and quick-access actions |
| Subjects | Subject Registry | Browse, register, and manage subject records |
| Assessments | Assessment History | Full log of all EEG classifications with filtering |
| Clinicians | Clinician Accounts | Admin-only panel for managing user accounts |

The sidebar does not expand to show labels by default on desktop — labels appear as tooltips on hover. On narrower screens (tablets used in clinical settings), the sidebar collapses to an off-canvas drawer triggered by a hamburger control. There is no top navigation bar. The sidebar carries the application logo at the top and the logged-in clinician's name and a logout control at the bottom.

### 2.2 Page Structure

Every page follows a consistent three-zone layout:

```
┌──────────────────────────────────────────────────────────┐
│ Sidebar  │  Page Header (title + contextual actions)     │
│          ├───────────────────────────────────────────────│
│          │  Page Body (primary content area)             │
│          │                                               │
│          │                                               │
│          │                                               │
└──────────────────────────────────────────────────────────┘
```

The **Page Header** contains the page title, a one-line description of what the page does, and any primary action buttons (e.g., "New Subject", "Export PDF"). It does not scroll — it remains anchored as the body scrolls beneath it.

The **Page Body** is the full working area. It uses a maximum content width of 1200px, centered, with generous padding. Wide data tables, EEG visualizations, and topographic maps are allowed to use the full width of this area.

### 2.3 The Assessment Flow (Core Workflow)

The assessment flow is the heart of the application. It is not a separate page — it is a **right-side drawer panel** that slides in over the current view without destroying context. This keeps the clinician in the Subjects view while they submit an EEG.

The drawer is divided into three sequential steps, displayed as a horizontal step indicator at the top of the panel:

**Step 1 — Subject & Recording Metadata**
- Subject code field with autocomplete from existing records
- Age and gender fields (auto-filled if subject exists)
- Environmental context: sleep hours the night before, food intake in the past 3 hours, caffeine status (yes/no), current medication status (yes/no), and clinical notes (free text, optional)
- This metadata does not affect the model — it informs the clinical report and triggers data quality warnings

**Step 2 — EEG Upload**
- Drag-and-drop CSV upload area with a fallback file picker
- Real-time column validation with a visual channel map — a schematic 10-20 scalp diagram lights up each electrode green as its column is confirmed present in the file
- Instant feedback on sampling rate inference and approximate recording duration
- If the file fails validation, a specific, plain-language error message identifies the problem (e.g., "Column 'T3' not found — please verify your export settings")

**Step 3 — Review & Submit**
- Summary card of entered metadata and file details
- A clearly labeled disclaimer — styled as an informational notice, not a warning box — reminding the clinician that results are not a definitive diagnosis
- A single "Run Assessment" button

The drawer does not close automatically after submission. It transitions into a **processing state**: a calm progress indicator with status messages ("Filtering signal...", "Extracting features...", "Running classification...") that reflect actual pipeline stages. When complete, the drawer closes and the new result appears at the top of the subject's assessment history.

### 2.4 Results View

Results are accessible from both the Assessment History page and each Subject's detail page. Selecting a result opens a dedicated full-page Results View — the most information-dense part of the application.

The Results View is organized into a vertical stack of collapsible sections:

1. **Classification Summary** — The primary output. Displayed as a large, unambiguous label (e.g., "Consistent with ADHD-I — Inattentive Type") accompanied by the confidence score as a horizontal bar, not a percentage in isolation. Below the label, two sentences of clinical impression language describe what this classification means and what it does not mean.

2. **Data Quality Notice** — Conditionally shown. If sleep data was missing, caffeine was reported, or confidence is below a set threshold (e.g., 60%), this section appears with amber styling and explains the specific factors that may affect reliability. If data quality is acceptable, this section is hidden entirely — not collapsed, not shown as "All Clear." Silence here means no concern.

3. **Spectral Band Power Analysis** — A horizontal grouped bar chart showing absolute and relative power across delta, theta, alpha, beta, and gamma bands, per electrode, with the option to toggle between per-electrode and averaged views. Clinical ratios (TBR, TAR, ATR) are shown as a compact table beneath the chart with reference ranges annotated.

4. **Topographic Scalp Maps** — A row of per-band heatmaps rendered on the standard 10-20 head outline. Clinicians can click any map to expand it. A toggle switches between absolute and relative power views.

5. **EEG Waveform Visualization** — The rendered EEG trace. Tabs allow switching between raw, bandpass-filtered, and individual band views. Electrode groups are visually separated with subtle dividing lines.

6. **Temporal Biomarker Evolution** — Time-series plots of the 20 computed biomarkers across the recording window. These are grouped into panels: band power ratios, spectral features, hemispheric asymmetry, and regional metrics. Each panel is collapsible. Summary statistics (mean, std, min, max) for each biomarker appear in a tooltip on hover.

7. **Intervention Suggestions** — A structured list of general recommendations appropriate to the classified subtype. Each suggestion is phrased in clinical, not prescriptive, language ("Behavioral assessment by a licensed psychologist may be warranted..."). These are not personalized treatment plans — that limitation is stated once, plainly, at the top of this section.

8. **Export Actions** — A sticky bar at the bottom of the results view with a single prominent "Download PDF Report" button and a secondary "Share via Link" option (generates a time-limited read-only URL to the result for colleagues).

---

## 3. Intended Use Patterns

### 3.1 Primary Use Case — In-Clinic EEG Review
A clinician has conducted an EEG session with a child and exported the recording as CSV. They open BayesADHD on a desktop workstation at their clinic, navigate to the subject's record (or create one if new), open the assessment drawer, upload the file, enter the session metadata from their intake notes, and submit. Within two to three minutes, they have a structured report to reference during their consultation or to attach to the child's file.

### 3.2 Secondary Use Case — Retrospective Review
A clinician reviews a subject's history across multiple assessments over several months. The Subject detail page shows a timeline of all past classifications with confidence scores, allowing the clinician to observe trends in EEG-based indicators over time. This supports follow-up consultations and monitoring of treatment effects, without the system making any claims about treatment efficacy.

### 3.3 Administrative Use Case — Clinic Setup
An administrator clinician account configures the system for a clinic: adding clinician accounts, reviewing all assessments across the team, and managing subject records. The admin panel is minimal and functional — a table of clinician accounts with add/deactivate controls. No configuration of the model, thresholds, or signal processing is exposed to any user.

### 3.4 Out-of-Scope Interactions
The application does not support real-time EEG streaming, mobile-native use, multi-clinic tenancy, or integration with external EMR systems in its current scope. These are deferred to Year 2 of the roadmap. No part of the interface implies these capabilities.

---

## 4. Visual Style

### 4.1 Design Philosophy
The visual design follows three principles: **clinical legibility, calm authority, and honest limitation**. It is not a consumer health app — it does not use illustrations, playful iconography, or motivational framing. It is not a raw data tool — it does not expose technical jargon without explanation. It sits deliberately between the two: professional, composed, and transparent about what it is and is not.

### 4.2 Color Direction

The palette should feel measured and restrained — appropriate for a tool opened alongside patient files and clinical notes. Bright consumer UI conventions (saturated primaries, vibrant accent colors) are out of place here. The general direction is cool, muted, and professional: something closer to a medical journal than a wellness app.

Regardless of the specific palette chosen, the color system needs to serve the following functional roles clearly:

- A **primary brand color** for the sidebar, primary buttons, and active navigation states
- A **light tint** of the primary for hover states, selected items, and highlighted rows
- A **neutral surface** color for page backgrounds, distinct from white card surfaces
- A **semantic success color** for validated channels and successful uploads
- A **semantic warning color** for data quality notices and low-confidence alerts
- A **semantic error color** for failed validations and file errors
- A **confidence gradient** — three distinct color states (high, mid, low) for the confidence bar, derived from the success, warning, and error colors respectively

The palette should remain legible and meaningful when the PDF report is printed in black and white. Color alone must never be the only carrier of meaning.

### 4.3 Typography Direction

The typeface should be clean, neutral, and highly legible at small sizes — a contemporary sans-serif appropriate for dense clinical data. It should not read as decorative or branded in the consumer sense. A secondary monospaced face is needed for numerical data: confidence scores, frequency values, band power readings, and timestamps. These values benefit from fixed-width rendering to align cleanly in tables and avoid visual noise.

The type scale should establish a clear hierarchy between page headings, section labels, body text, and helper text. The classification result label — the most important piece of information on the results page — should be the most visually prominent text element in the application.

Both typefaces should be self-hosted to avoid external font CDN dependencies, which may be unreliable in clinic network environments.

### 4.4 Iconography
Icons are sourced from the Lucide icon set — consistent stroke weight (1.5px), rounded line caps, and 24×24px base size. Icons are never used alone as the sole indicator of meaning — they always appear paired with a text label or tooltip. No filled icon style is used, maintaining visual lightness.

### 4.5 Data Visualization Style
All charts (band power bars, temporal biomarker plots) use the primary palette as a base. Chart axes are labeled in plain language ("Power (µV²/Hz)"), not abbreviated. Gridlines are light and minimal. No chart uses 3D rendering or decorative effects. Topographic maps use a perceptually uniform diverging colormap (blue–white–red) appropriate for scientific display, not a rainbow scale.

### 4.6 Motion and Transitions
Transitions are functional and brief (150–200ms ease-out). The assessment drawer slides in from the right. The processing state uses a subtle pulsing bar — not a spinning loader — to convey steady, ongoing computation rather than uncertainty. Section collapse/expand uses a smooth height animation. No transition exists purely for decorative effect.

---

## 5. Interaction Feel

### 5.1 Tone of the Interface
Interface copy is written in the second person, present-tense for instructions, and past-tense for completed actions. It is direct without being terse.

- Not: *"Please enter the required patient demographic information fields below."*  
- Yes: *"Enter the subject's age and gender."*

- Not: *"Classification complete! Your results are ready."*  
- Yes: *"Assessment complete. Review the findings below."*

Clinical impression text in results uses hedged, professional language consistent with how a supportive diagnostic aid would communicate findings. Phrases like "may suggest," "findings are consistent with," and "should be interpreted alongside" appear throughout. The word "diagnosis" is never used as a verb performed by the system.

### 5.2 Error Handling
Errors are surfaced inline, adjacent to the field or action that caused them. They name the problem specifically and, where possible, suggest a resolution. There are no modal error dialogs that block the interface. Failed uploads allow re-upload without clearing other fields. A global error toast appears at the top-center of the screen (not a corner — center-top for clinical visibility) only for server-level failures, and auto-dismisses after 6 seconds.

### 5.3 Loading and Processing States
Every asynchronous operation that takes more than 500ms shows a loading state. The EEG upload shows a progress bar tied to actual upload progress. The classification pipeline shows sequential status messages that reflect real pipeline stages. Data visualizations that take time to render show a skeleton placeholder in the correct dimensions before the image loads — preventing layout shift during the results review.

### 5.4 Accessibility Considerations
- All interactive elements are keyboard-navigable with visible focus rings styled in the primary teal color.
- Contrast ratios for all text/background combinations meet WCAG AA minimum (4.5:1 for body, 3:1 for large text).
- Color is never the sole indicator of meaning — the confidence bar uses both color and a text label; data quality notices use both color and an icon paired with text.
- Form fields include explicit `<label>` elements (not placeholder-only) to support screen readers used in accessibility-conscious clinic environments.

---

## 6. PDF Report Composition

The exported clinical PDF report is the primary artifact the application produces for clinical record-keeping. It is formatted for A4 paper and designed to be legible when printed in black and white, though color is used when viewed digitally.

**Report Structure:**

1. **Header** — Application name, report generation timestamp, system version number, and a centered disclaimer bar
2. **Subject Information** — Subject code, age, gender, and recording session metadata (sleep, food, caffeine, medication)
3. **Recording Conditions** — Channel coverage confirmed, sampling rate, recording duration, and any data quality flags
4. **Classification Result** — The primary classification label in bold, confidence score, and two-paragraph clinical impression
5. **Spectral Band Power Summary** — A compact table of band power values and clinical ratios for all 19 channels plus averaged values
6. **Band Power Bar Chart** — Printed in grayscale with pattern fills to distinguish bands
7. **Clinical Interpretation Notes** — Subtype-specific general observations drawn from the literature, written for a clinician audience
8. **Intervention Suggestions** — General recommendations appropriate to the classification
9. **Limitations and Disclaimers** — A standing section that appears in every report, enumerating what the system cannot determine
10. **Clinician Signature Block** — Clinician name, occupation, assessment date, and a signature line for physical annotation

The report header and footer carry the disclaimer on every page: *"This report is generated by an automated support tool and does not constitute a clinical diagnosis. All findings require interpretation by a qualified healthcare professional."*

---

## 7. State and Session Behavior

The application uses server-side session management. Sessions expire after 8 hours of inactivity — appropriate for a full clinical workday without forcing re-authentication between patient consultations. Concurrent login from two devices under the same clinician account is prevented; the newer login invalidates the previous session with a clear on-screen notification.

Unsaved metadata in the assessment drawer (Step 1 fields filled, EEG not yet submitted) is preserved if the clinician navigates away and returns within the same session. A subtle "Draft" indicator in the drawer trigger reminds them an unsubmitted assessment exists.

---

## 8. What the Application Deliberately Does Not Do

These are not omissions — they are explicit design decisions. Some may warrant brief explanatory copy in the interface itself:

- It does not generate a diagnosis. It generates a structured, evidence-supported *assessment report*.
- It does not interpret results for the clinician. It presents findings and leaves clinical judgment intact.
- It does not accept images, video, questionnaire data, or any modality other than 19-channel EEG CSV.
- It does not personalize intervention suggestions to the individual subject's history or comorbidities.
- It does not expose model internals, training data statistics, or hyperparameter values to clinical users.
- It does not suggest medication, dosage, or any pharmacological intervention.
- It does not indicate whether a previous assessment's classification has "improved" or "worsened" — the subject timeline displays classifications without comparative judgment language.

These constraints exist because the application is designed for clinical appropriateness, not feature completeness. A tool that does less, but does it honestly and reliably, serves its users better than one that overpromises.
