# AI-Powered Clinical Trial Matching System

## Overview

The AI-Powered Clinical Trial Matching System is a patient-trial matching platform that identifies the most suitable clinical trials from ClinicalTrials.gov data using information extracted from Electronic Health Records (EHRs).

The system combines biomedical NLP for information extraction, rule-based eligibility matching, and machine learning-based ranking. The application is built with Streamlit and provides an interactive workflow for either manually entering medical data or uploading patient records, reviewing extracted information, and viewing ranked clinical trial recommendations.

---

## Features

- Upload patient EHR documents (PDF)
- Extraction of patient information
- Hybrid rule-based and machine learning trial matching
- Ranked clinical trial recommendations
- Match score and eligibility explanation
- Interactive Streamlit interface

---

## Methodology


### 1. Clinical Trial Data Extraction

Clinical trial data is collected from the **ClinicalTrials.gov API v2** and transformed into a structured dataset using a combination of BioBERT, ClinicalBERT and scispacy. 

The extracted information includes:

- Trial identifiers (NCT ID)
- Study title and summary
- Conditions and interventions
- Recruitment status
- Study phase
- Eligibility criteria (Inclusion & Exclusion Criteria)
- Age and sex requirements
- Healthy volunteer status
- Study locations
- Sponsor information

The processed records are cleaned, standardized, and stored as a structured JSON database (`trials_extracted.json`), which serves as the primary knowledge base for the matching system.

---

### 2. Information Extraction

Patient information is extracted from uploaded EHRs using BioBERT.

Extracted entities include:

- Demographics
- Medical conditions
- Previous treatments
- Laboratory values
- Vital signs
- Performance status
- Medications

---

### 3. Rule-Based Eligibility Matching (70%)

Clinical trials are first evaluated against mandatory eligibility criteria such as:

- Disease compatibility
- Age
- Gender
- Laboratory thresholds
- Vital measurements
- ECOG performance status
- Treatment history

This stage removes trials that clearly violate inclusion or exclusion criteria.

---

### 4. Machine Learning Scoring (30%)

Remaining trials are scored using a trained XGBoost model that learns patient-trial compatibility from engineered clinical features.

---

### 5. Hybrid Ranking

The final ranking is computed as

```
Final Score =
0.7 × Rule-Based Score
+
0.3 × Machine Learning Score
```

This combines explicit clinical eligibility with learned similarity patterns to produce the final ranked recommendations.

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Backend | Python |
| NLP | BioBERT, ClinicalBERT, spaCy, scispaCy |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Source | ClinicalTrials.gov |

---

## Installation

Clone the repository

```bash
git clone https://github.com/HarshiniReddy13/AI-Powered-Clinical-Trial-Matching-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```


---

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit application |
| `backend.py` | Core trial matching pipeline |
| `models.py` | Model loading and inference utilities |
| `trial_matching_bundle.pkl` | Serialized machine learning model |
| `trials_extracted.json` | Structured clinical trial dataset |
| `requirements.txt` | Python dependencies |

---

## Future Work

- Support additional EHR formats
- Real-time synchronization with ClinicalTrials.gov
- Enhanced explainability for trial recommendations
- Integration with FHIR-compatible EHR systems
- Multi-patient batch matching

---

## License

This project is intended for educational and research purposes.
