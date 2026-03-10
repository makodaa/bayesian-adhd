# Narrative Interpretation: Cortical Arousal & Attention Regulation

This document describes the narrative interpretation feature added to the
bayesian-adhd application. It covers the clinical rationale, the decision
logic, every file that was changed or created, exact insertion points, and
the output behaviour in each surface of the application.

---

## 1. Purpose and Clinical Rationale

Raw EEG classification outputs (a predicted class label and a confidence
score) provide a binary signal but no explanation. Clinicians reviewing a
result must manually correlate the band power values and the
theta/beta ratio against known ADHD neuroscience to form an interpretive
impression. The narrative feature automates that step.

The generated paragraph frames findings in terms of two neuroscientific
dimensions that are well-established in the ADHD EEG literature:

**Cortical arousal** refers to the overall activation level of the cortex.
ADHD is frequently associated with a hypoarousal pattern: the frontal
cortex is under-activated relative to a typically developing baseline,
which manifests spectrally as elevated slow-wave (theta) power and a
reduced fast-wave (beta) contribution. The opposite, hyperarousal, can
appear in cases involving anxiety, stimulant use, or hypervigilance and
presents as elevated beta with a suppressed theta/beta ratio.

**Attention regulation** refers to the cortex's ability to gate sensory
input, sustain focus, and apply inhibitory control. The theta/beta ratio
(TBR) is the most studied single EEG index for this dimension. An elevated
TBR is associated with reduced frontally mediated inhibitory control and is
one of the spectral features the classifier was trained on.

The narrative synthesises these two dimensions together with the
classification outcome and model confidence to produce a 5-sentence
plain-text paragraph that any clinician can read without needing to
interpret raw numbers.

---

## 2. Input Signals

All inputs are derived from values already computed by `BandAnalysisService`
and stored in the database. No additional model inference is performed.

| Input | Source | Role in narrative |
|---|---|---|
| `avg_relative_power["theta"]` | `band_powers` table, averaged across electrodes | Primary arousal indicator |
| `avg_relative_power["beta"]` | `band_powers` table, averaged across electrodes | Secondary arousal indicator |
| `avg_relative_power["alpha"]` | `band_powers` table, averaged across electrodes | Spectral overview context |
| `band_ratios["theta_beta_ratio"]` | `ratios` table | Core ADHD biomarker; drives attention regulation sentence |
| `predicted_class` | `results` table | Anchors the sentence framing to the classification label |
| `confidence_score` | `results` table | Qualifies the certainty language used throughout |

---

## 3. Decision Logic

### 3.1 Thresholds

Defined as module-level constants in `narrative_service.py`:

```python
_TBR_HYPO_THRESHOLD   = 3.0   # TBR above this → hypoarousal
_TBR_HYPER_THRESHOLD  = 1.5   # TBR below this (+ high beta) → hyperarousal
_THETA_HYPO_THRESHOLD = 0.35  # relative theta above this → hypoarousal signal
_BETA_HYPER_THRESHOLD = 0.25  # relative beta above this (+ low TBR) → hyperarousal
_CONF_HIGH            = 0.80  # confidence ≥ 80% → "high" tier
_CONF_MODERATE        = 0.60  # confidence ≥ 60% → "moderate" tier
```

The same numeric values are mirrored as `const` declarations in both
`index.html` and `results.html` so that JavaScript-generated narratives
are identical to Python-generated ones.

### 3.2 Sentence 1 — Spectral Overview

Always the same structure; no branching. Reports theta %, beta %, alpha %,
and TBR to one decimal place each.

> *"EEG spectral analysis shows relative theta power of 38.0%, relative
> beta power of 18.0%, relative alpha power of 22.0%, and a theta/beta
> ratio (TBR) of 4.10."*

### 3.3 Sentence 2 — Arousal Characterisation

Three possible outputs depending on TBR and band values:

| Condition | Label | Output |
|---|---|---|
| `TBR > 3.0` **or** `theta > 35%` | Hypoarousal | "The elevated theta activity and theta/beta ratio are consistent with a cortical hypoarousal pattern, suggesting reduced frontal activation and decreased inhibitory tone." |
| `TBR < 1.5` **and** `beta > 25%` | Hyperarousal | "The relatively high beta power and low theta/beta ratio suggest a cortical hyperarousal or heightened activation pattern, which may be associated with anxiety, hypervigilance, or stimulant effects." |
| Neither condition met | No clear dysregulation | "The theta/beta ratio and band power distribution do not clearly indicate cortical arousal dysregulation within this recording." |

Note: hypoarousal is checked first. A recording can only be labelled
hyperarousal if it does not also meet the hypoarousal criteria.

### 3.4 Sentence 3 — Attention Regulation Linked to Classification

Four branches based on the cross of `is_adhd` (bool) and `elevated_tbr`
(TBR > 3.0):

| `is_adhd` | `elevated_tbr` | Template |
|---|---|---|
| True | True | "Combined with a `{class}` classification at `{conf}`% confidence, this profile is consistent with reduced attentional gating and impaired frontally mediated inhibitory control, patterns commonly observed in ADHD." |
| True | False | "The `{class}` classification (`{conf}`% confidence) was reached despite a theta/beta ratio that does not fall in the typically elevated range; other spectral and temporal features may have contributed to this classification." |
| False | True | "Although the recording was classified as `{class}` at `{conf}`% confidence, the elevated theta/beta ratio warrants clinical consideration, as this pattern can occasionally appear in the context of other neurodevelopmental or attentional presentations." |
| False | False | "Consistent with the `{class}` classification (`{conf}`% confidence), the attention regulation markers in this recording do not indicate the spectral patterns typically associated with ADHD-related cortical dysregulation." |

### 3.5 Sentence 4 — Confidence Qualifier

Three tiers, matching the existing confidence thresholds used throughout
the application (`getConfidenceState` in JS, `_confidence_tier` in Python):

| Tier | Range | Output |
|---|---|---|
| High | ≥ 80% | "Model confidence is high (`{conf}`%), supporting a stronger weighting of these findings within a broader clinical assessment." |
| Moderate | 60–79% | "Model confidence is moderate (`{conf}`%); these findings should be corroborated with additional clinical and behavioral data before drawing conclusions." |
| Low | < 60% | "Model confidence is low (`{conf}`%), and these findings should be interpreted with considerable caution; additional testing is strongly recommended." |

### 3.6 Sentence 5 — Closing Disclaimer

Fixed text, always appended:

> *"These findings are intended to support clinical decision-making and
> should not be interpreted in isolation from clinical history, behavioral
> evaluation, and qualified professional judgment."*

---

## 4. Files Changed

### 4.1 New file — `backend/app/services/narrative_service.py`

Canonical source of truth for all narrative decision logic. Contains
`NarrativeService` with one public method and four private static helpers.

```
NarrativeService
├── generate_arousal_narrative(predicted_class, confidence_score,
│       avg_relative_power, band_ratios) → str
├── _is_adhd(predicted_class) → bool
├── _confidence_tier(confidence_score) → str
├── _arousal_sentence(theta, beta, tbr) → str
├── _attention_sentence(is_adhd, tbr, predicted_class,
│       conf_tier, conf_pct) → str
└── _confidence_sentence(conf_tier, conf_pct) → str
```

**Signature of `generate_arousal_narrative`:**

```python
def generate_arousal_narrative(
    self,
    predicted_class: str,
    confidence_score: float,          # [0, 1] fraction
    avg_relative_power: dict[str, float],  # {"theta": 0.38, "beta": 0.18, ...}
    band_ratios: dict[str, float],         # {"theta_beta_ratio": 4.1, ...}
) -> str
```

Input values for `avg_relative_power` should be fractional (0–1), not
percentages. The method normalises `confidence_score` if it appears to be
a percentage (> 1).

---

### 4.2 Edited — `backend/app/services/pdf_service.py`

**Line 17** — import added:
```python
from .narrative_service import NarrativeService
```

**`PDFReportService.__init__`** — instance created:
```python
self._narrative_service = NarrativeService()
```

**`generate_report()`** — new call inserted between
`_build_classification_section()` and `_build_band_powers_section()`:
```python
story.extend(self._build_narrative_section(result_data))
```

**New method `_build_narrative_section(result_data)`** (line 416):

Derives `avg_relative_power` by averaging the `relative_power` values
across all electrodes from the `band_powers` list in `result_data`. Builds
`band_ratios` from the `ratios` list. Calls
`self._narrative_service.generate_arousal_narrative(...)`. Renders the
output inside a lightly shaded ReportLab table cell
(`background: #f0f4f8`, `border: 0.5pt #c0cdd8`) at full page width
(170 mm). Returns an empty fallback paragraph when spectral data is
absent.

**PDF section order after this change:**

```
1  Header
2  Disclaimer
3  Subject Information
4  Recording Environment
5  Classification Result
6  Arousal & Attention Profile   ← new
7  Relative Band Power Analysis
8  Clinical Ratios
9  EEG Signal Visualizations
10 Data Quality Notes
11 Signatory
```

---

### 4.3 Edited — `backend/app/static/index.css`

Two additions to the grid layout and typography sections:

**Grid order** (inserted after `.summary-card` rule):
```css
#narrative-card {
  grid-column: 1 / -1;
  order: 2;
}
```
This places the narrative card at `order: 2` — after Classification
Summary (`order: 1`) and before Data Quality Warnings (`order: 3`).

**Typography** (inserted after `.summary-card h3` block):
```css
.narrative-text {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--text-700);
}
```

---

### 4.4 Edited — `backend/app/templates/index.html`

#### HTML change — new article element (lines 284–287)

Inserted between `#prediction-result` card and `#quality-notice-section`:

```html
<article class="summary-card" id="narrative-card">
  <h3>Arousal &amp; Attention Profile</h3>
  <p id="narrative-text" class="narrative-text"></p>
</article>
```

The card is always visible after a result loads. It shares the
`summary-card` class so it inherits the same border, background,
padding, and heading style as the Classification Summary card above it.

#### JavaScript additions (inserted before `clinicalImpressionText`)

Six functions added, all prefixed to avoid naming collisions with the
shared `results.html` implementations:

| Function | Purpose |
|---|---|
| `_narrativeArousalSentence(theta, beta, tbr)` | Returns sentence 2 (arousal characterisation) |
| `_narrativeAttentionSentence(isAdhd, tbr, classification, confPct)` | Returns sentence 3 (attention regulation) |
| `_narrativeConfidenceSentence(confidence)` | Returns sentence 4 (confidence qualifier) |
| `generateNarrative(classification, confidence, avgRelPower, bandRatios)` | Assembles all 5 sentences into a full paragraph |
| `renderNarrativeCard(classification, confidence, bandData)` | Extracts `average_relative_power` and `band_ratios` from the `band_analysis` response object, calls `generateNarrative`, writes the result to `#narrative-text` |

Threshold constants used by these functions:

```js
const _TBR_HYPO   = 3.0;
const _TBR_HYPER  = 1.5;
const _THETA_HYPO = 0.35;
const _BETA_HYPER = 0.25;
```

#### Call site — post-analysis render path

`renderNarrativeCard` is called immediately after `displayPrediction`
in `handleAssessmentSubmit`:

```js
displayPrediction(result.classification, result.confidence_score);
renderNarrativeCard(result.classification, result.confidence_score, result.band_analysis || {});
displayBandPowers(result.band_analysis || {});
```

The `band_analysis` object returned from `/predict` already contains
`average_relative_power` and `band_ratios`, so no additional fetch is
required.

**Dashboard section order after this change:**

```
Classification Summary        ← existing (order 1)
Arousal & Attention Profile   ← new     (order 2)
Data Quality Warnings         ← existing (order 3)
Spectral Band Power Analysis  ← existing (order 5)
EEG Waveform Visualization    ← existing (order 4)
Clinical Ratios               ← existing (order 6)
Download PDF Report bar       ← existing (order 7)
Topographic Scalp Maps        ← existing (order 8)
Temporal Biomarker Evolution  ← existing (order 9)
Intervention Suggestions      ← existing (order 10)
```

---

### 4.5 Edited — `backend/app/templates/results.html`

#### JavaScript additions (inserted before `const BAND_COLORS`)

Seven functions added with the `_narr` prefix to avoid conflicts:

| Function | Purpose |
|---|---|
| `_narrArousalSentence(theta, beta, tbr)` | Returns sentence 2 |
| `_narrAttentionSentence(isAdhd, tbr, classification, confPct)` | Returns sentence 3 |
| `_narrConfidenceSentence(confidence)` | Returns sentence 4 |
| `buildResultNarrative(data)` | Receives the full result object from `/api/results/<id>`, computes average relative power from the raw `band_powers` rows, extracts TBR from the `ratios` array, assembles and returns the full 5-sentence paragraph, or `null` if band data is absent |

`buildResultNarrative` is the results-drawer equivalent of
`renderNarrativeCard` in `index.html`. It operates directly on the
`data` object that `renderResultDetail` already receives, requiring no
additional fetch.

#### `renderResultDetail` change

A `narrativeSection` constant is constructed after `ratioSection` is built:

```js
const narrativeText = buildResultNarrative(data);
const narrativeSection = `
  <section class="detail-section">
    <h3>Arousal &amp; Attention Profile</h3>
    ${ narrativeText
        ? `<p style="margin:0;font-size:0.84rem;line-height:1.6;
                     color:var(--text-700)">${escapeHtml(narrativeText)}</p>`
        : '<p style="...">Insufficient spectral data to generate an arousal profile.</p>'
    }
  </section>`;
```

It is inserted into `body.innerHTML` between `bandSection` and
`ratioSection`:

```js
body.innerHTML =
  classificationSection +
  subjectSection        +
  bandSection           +
  narrativeSection      +   // ← new
  ratioSection          +
  notesSection          +
  eegVizSection         +
  topoSection;
```

**Results drawer section order after this change:**

```
Classification
Subject & Recording
Spectral Band Power
Arousal & Attention Profile   ← new
Clinical Ratios
Notes & Disclaimers
EEG Signal Visualizations
Topographic Maps
```

---

## 5. Output Examples

### 5.1 ADHD Inattentive — High TBR — High Confidence

Inputs: `predicted_class = "ADHD Inattentive (ADHD-I)"`,
`confidence = 0.85`, `theta = 0.38`, `beta = 0.18`, `alpha = 0.22`,
`TBR = 4.10`

> EEG spectral analysis shows relative theta power of 38.0%, relative
> beta power of 18.0%, relative alpha power of 22.0%, and a theta/beta
> ratio (TBR) of 4.10. The elevated theta activity and theta/beta ratio
> are consistent with a cortical hypoarousal pattern, suggesting reduced
> frontal activation and decreased inhibitory tone. Combined with a ADHD
> Inattentive (ADHD-I) classification at 85.0% confidence, this profile
> is consistent with reduced attentional gating and impaired frontally
> mediated inhibitory control, patterns commonly observed in ADHD. Model
> confidence is high (85.0%), supporting a stronger weighting of these
> findings within a broader clinical assessment. These findings are
> intended to support clinical decision-making and should not be
> interpreted in isolation from clinical history, behavioral evaluation,
> and qualified professional judgment.

### 5.2 Non-ADHD — Normal TBR — Moderate Confidence

Inputs: `predicted_class = "Non-ADHD"`, `confidence = 0.72`,
`theta = 0.28`, `beta = 0.22`, `alpha = 0.30`, `TBR = 1.8`

> EEG spectral analysis shows relative theta power of 28.0%, relative
> beta power of 22.0%, relative alpha power of 30.0%, and a theta/beta
> ratio (TBR) of 1.80. The theta/beta ratio and band power distribution
> do not clearly indicate cortical arousal dysregulation within this
> recording. Consistent with the Non-ADHD classification (72.0%
> confidence), the attention regulation markers in this recording do not
> indicate the spectral patterns typically associated with ADHD-related
> cortical dysregulation. Model confidence is moderate (72.0%); these
> findings should be corroborated with additional clinical and behavioral
> data before drawing conclusions. These findings are intended to support
> clinical decision-making and should not be interpreted in isolation from
> clinical history, behavioral evaluation, and qualified professional
> judgment.

### 5.3 Non-ADHD — Elevated TBR Mismatch — Moderate Confidence

Inputs: `predicted_class = "Non-ADHD"`, `confidence = 0.65`,
`theta = 0.40`, `beta = 0.16`, `alpha = 0.24`, `TBR = 3.5`

> EEG spectral analysis shows relative theta power of 40.0%, relative
> beta power of 16.0%, relative alpha power of 24.0%, and a theta/beta
> ratio (TBR) of 3.50. The elevated theta activity and theta/beta ratio
> are consistent with a cortical hypoarousal pattern, suggesting reduced
> frontal activation and decreased inhibitory tone. Although the recording
> was classified as Non-ADHD at 65.0% confidence, the elevated theta/beta
> ratio warrants clinical consideration, as this pattern can occasionally
> appear in the context of other neurodevelopmental or attentional
> presentations. Model confidence is moderate (65.0%); these findings
> should be corroborated with additional clinical and behavioral data
> before drawing conclusions. These findings are intended to support
> clinical decision-making and should not be interpreted in isolation from
> clinical history, behavioral evaluation, and qualified professional
> judgment.

### 5.4 ADHD Hyperactive-Impulsive — Low TBR — Low Confidence

Inputs: `predicted_class = "Hyperactive-Impulsive (ADHD-H)"`,
`confidence = 0.55`, `theta = 0.25`, `beta = 0.30`, `alpha = 0.28`,
`TBR = 1.2`

> EEG spectral analysis shows relative theta power of 25.0%, relative
> beta power of 30.0%, relative alpha power of 28.0%, and a theta/beta
> ratio (TBR) of 1.20. The relatively high beta power and low theta/beta
> ratio suggest a cortical hyperarousal or heightened activation pattern,
> which may be associated with anxiety, hypervigilance, or stimulant
> effects. The Hyperactive-Impulsive (ADHD-H) classification (55.0%
> confidence) was reached despite a theta/beta ratio that does not fall in
> the typically elevated range; other spectral and temporal features may
> have contributed to this classification. Model confidence is low
> (55.0%), and these findings should be interpreted with considerable
> caution; additional testing is strongly recommended. These findings are
> intended to support clinical decision-making and should not be
> interpreted in isolation from clinical history, behavioral evaluation,
> and qualified professional judgment.

---

## 6. Architecture Notes

### No new API endpoints

The narrative is generated entirely from data already present in the
existing API responses:

- Dashboard: `result.band_analysis.average_relative_power` and
  `result.band_analysis.band_ratios` are returned by `/predict`.
- Results drawer: `data.band_powers` and `data.ratios` are returned by
  `/api/results/<id>`.
- PDF: `result_data['band_powers']` and `result_data['ratios']` are
  already loaded by `generate_result_pdf()` before PDF generation begins.

### No database changes

Narratives are derived, not stored. No schema migrations, no new tables,
no new columns.

### No external dependencies

The feature uses only Python standard library types and the logging
infrastructure already in place. No NLP libraries or LLM calls are made.

### Consistency between Python and JavaScript

The Python (`NarrativeService`) and JavaScript (`generateNarrative` /
`buildResultNarrative`) implementations share identical threshold values
and identical branching logic. The Python version is the canonical
reference; the JS versions were written to match it exactly. If thresholds
are updated in `narrative_service.py`, the corresponding `const`
declarations in both `index.html` and `results.html` must be updated to
match.

### Fallback behaviour

All three surfaces handle absent spectral data gracefully:

| Surface | Fallback |
|---|---|
| Dashboard (`#narrative-text`) | "Spectral data is not yet available to generate an arousal profile." |
| Results drawer (`narrativeSection`) | "Insufficient spectral data to generate an arousal profile." |
| PDF (`_build_narrative_section`) | "Insufficient spectral data to generate an arousal and attention profile." (same `ReportBody` style) |

---

## 7. Threshold Adjustment Guide

All numeric thresholds are centralised. To change a threshold:

1. Update the constant in `backend/app/services/narrative_service.py`
   (lines 14–17).
2. Update the matching `const` in the `index.html` JS block
   (`_TBR_HYPO`, `_TBR_HYPER`, `_THETA_HYPO`, `_BETA_HYPER`).
3. Update the matching `const` in the `results.html` JS block
   (`_NARR_TBR_HYPO`, `_NARR_TBR_HYPER`, `_NARR_THETA_HYPO`,
   `_NARR_BETA_HYPER`).
4. Update the example outputs in this document if materially affected.

Confidence tier boundaries (`_CONF_HIGH = 0.80`, `_CONF_MODERATE = 0.60`)
are shared with the existing `getConfidenceState` function in both
templates. Any change to confidence tiers should be applied consistently
across all three locations.
