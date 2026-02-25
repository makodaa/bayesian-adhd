# Bayesian ADHD (BayesADHD) — Design & Layout Specification

## 1. Application Architecture & Layout

The interface is built around a rigid, three-zone structure that prioritizes task focus and clinical workflow over exploratory navigation.

### 1.1 Global Navigation (Sidebar)
* **Behavior:** Persistent across all views, narrow, and icon-first. On desktop, text labels appear only as tooltips on hover. On tablet screens, it collapses into an off-canvas drawer activated by a hamburger menu.
* **Content:** Contains exactly four destinations: Dashboard, Subjects, Assessments, and Clinicians (Admin). The top houses the application logo, and the bottom houses the clinician's name and logout control.
* **Iconography:** Sourced from the Lucide icon set at 24x24px with a 1.5px stroke weight and rounded caps. Icons are never used without a text label or tooltip.

### 1.2 Page Structure
* **Page Header:** Anchored at the top of the viewport and does not scroll. It contains the page title, a single-line description of the page's function, and primary action buttons (e.g., "New Subject").
* **Page Body:** The primary content area operates with a maximum width of 1200px, centered on the screen with generous padding. Wide data tables, topographic maps, and EEG visualizations are permitted to span this full width.

---

## 2. Core Workflow Components

### 2.1 The Assessment Drawer (Right-Panel)
The primary assessment flow does not use a separate page; it utilizes a right-side drawer panel that slides over the current view to preserve the user's context. 

* **Header & Navigation:** Features a horizontal step indicator detailing the three sequential stages: Subject Metadata, EEG Upload, and Review & Submit.
* **Upload Interface (Step 2):** Uses a drag-and-drop CSV area with a fallback file picker. It includes a schematic 10-20 scalp diagram that illuminates each electrode in green as the corresponding column is validated in the uploaded file.
* **Processing State:** Upon submission, the drawer transitions into a processing state utilizing a calm, pulsing progress bar rather than a spinning loader. It displays sequential status messages reflecting actual pipeline computations.

### 2.2 Results View (Information Density)
The Results View is an information-dense, full-page layout organized into a vertical stack of collapsible sections.

* **Classification Summary:** The most prominent visual element. It displays a large, unambiguous label paired with a horizontal confidence bar (not an isolated percentage) and two sentences of clinical impression text.
* **Data Quality Notice:** A conditional component styled in amber. It appears only if specific triggers are met (e.g., missing sleep data, high caffeine, confidence below threshold) and remains entirely hidden otherwise.
* **Spectral & Topographic Data:** Spectral band power is displayed via a horizontal grouped bar chart using a primary base color, accompanied by a compact clinical ratio table. Topographic maps use a standard 10-20 head outline rendered with a perceptually uniform blue-white-red diverging colormap.
* **Export Actions:** A sticky bar anchored to the bottom of the view containing a prominent "Download PDF Report" action and a secondary read-only sharing link.

---

## 3. Visual & Styling System

The application completely avoids consumer UI aesthetics (vibrant colors, playful illustrations) in favor of clinical legibility and calm authority.

### 3.1 Typography
* **Application UI:** A self-hosted, clean, contemporary sans-serif typeface used for all general interface copy, headers, and clinical notes.
* **Numerical Data:** A self-hosted monospace typeface strictly used for fixed-width rendering of confidence scores, frequency values, band power readings, and timestamps to ensure clean alignment in tables. 
* **Form Inputs:** Explicit `<label>` elements are required for all form fields; placeholder-only inputs are prohibited for accessibility reasons.

### 3.2 Color Tokens
* **Primary Brand:** A cool, muted tone used for the sidebar, primary buttons, active navigation states, and chart bases.
* **Surface/Neutral:** Distinct from white card surfaces to clearly delineate page background from content areas.
* **Semantic Success:** Used for validated upload channels.
* **Semantic Warning:** Used for low-confidence alerts and data quality notices.
* **Semantic Error:** Used for failed validations.

---

## 4. Interaction & Motion Guidelines

* **Transitions:** Brief, functional animations constrained to 150–200ms ease-out curves. 
* **Loading States:** Asynchronous operations exceeding 500ms require a loading state. Data visualizations must utilize skeleton placeholders matching their exact final dimensions to prevent layout shifting upon render.
* **Error Handling:** Errors are surfaced inline directly adjacent to the relevant field, utilizing plain-language resolutions. Modal error dialogs that block the UI are strictly prohibited. Global server errors utilize an auto-dismissing (6-second) toast placed at the top-center of the screen.
* **Accessibility Constraints:** All interactive elements mandate keyboard navigation with visible primary-color focus rings. Color must never act as the sole communicator of meaning; it must always be paired with text labels or iconography. Text-to-background contrast must meet WCAG AA standards.