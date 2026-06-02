import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import time
import pdfplumber
import re
from datetime import date
from backend import run_pipeline, render_results
from streamlit.runtime.scriptrunner import get_script_run_ctx

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="TrialMatch — AI Clinical Trial Matching",
    page_icon="🧬",
    layout="wide"
)

# =====================================================
# GLOBAL STYLES  (light + dark adaptive)
# =====================================================

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════════════
   THEME TOKENS  —  light default, dark override
══════════════════════════════════════════════════ */
:root {
    --font-sans: 'Geist', -apple-system, sans-serif;
    --font-serif: 'Instrument Serif', Georgia, serif;
    --font-mono: 'Geist Mono', 'Courier New', monospace;

    /* Light palette */
    --bg:           #FAFAF8;
    --bg-card:      #FFFFFF;
    --bg-elevated:  #F3F3F0;
    --bg-input:     #FFFFFF;

    --border:       rgba(0,0,0,0.07);
    --border-md:    rgba(0,0,0,0.13);
    --border-focus: rgba(15,140,80,0.45);

    --text:         #111110;
    --text-muted:   #6B6A66;
    --text-hint:    #A0A09A;

    --accent:       #4FA892;
    --accent-light: #69B8A4;
    --accent-dim:   rgba(10,155,90,0.09);
    --accent-glow:  rgba(10,155,90,0.18);

    --amber:        #B45309;
    --amber-dim:    rgba(180,83,9,0.09);

    --red:          #C0392B;
    --red-dim:      rgba(192,57,43,0.08);

    --divider:      rgba(0,0,0,0.07);
    --section-rule: rgba(0,0,0,0.07);

    --radius-sm:    6px;
    --radius:       10px;
    --radius-lg:    16px;
    --radius-xl:    22px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg:           #0C0C0E;
        --bg-card:      #141416;
        --bg-elevated:  #1A1A1E;
        --bg-input:     #111113;

        --border:       rgba(255,255,255,0.07);
        --border-md:    rgba(255,255,255,0.12);
        --border-focus: rgba(62,224,154,0.45);

        --text:         #F2F0EB;
        --text-muted:   #888886;
        --text-hint:    #555553;

        --accent:       #3EE09A;
        --accent-light: #5AEBB0;
        --accent-dim:   rgba(62,224,154,0.12);
        --accent-glow:  rgba(62,224,154,0.25);

        --amber:        #F5A623;
        --amber-dim:    rgba(245,166,35,0.12);

        --red:          #FF5C5C;
        --red-dim:      rgba(255,92,92,0.10);

        --divider:      rgba(255,255,255,0.07);
        --section-rule: rgba(255,255,255,0.07);

    }
}

/* ══════════════════════════════════════════════════
   STREAMLIT CHROME
══════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--font-sans) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-md) !important;
}
[data-testid="stSidebar"] {
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
}
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1100px !important;
}

h1, h2, h3 {
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}

label,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p {
    font-family: var(--font-sans) !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
css[data-testid="stAppViewContainer"] > section > div:first-child {
    margin-top: 3rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stTextArea"] textarea {
    resize: vertical !important;
    min-height: 100px !important;
    line-height: 1.6 !important;
}
[data-baseweb="select"] * {
    background: var(--bg-input) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}
[data-baseweb="popover"] [role="option"] {
    background: var(--bg-elevated) !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: var(--bg-card) !important;
}

[data-testid="stNumberInput"] button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
}
[data-testid="stDateInput"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: var(--radius) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 4px 22px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}

/* ── Radio ── */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    color: var(--text) !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--accent) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--divider) !important;
    margin: 2rem 0 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border-md) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

[data-testid="stFileUploader"] > div {
    background: var(--bg-input) !important;
    border: 1.5px dashed var(--border-md) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--border-focus) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
}
[data-testid="stFileUploader"] button span {
    display: none !important;
}

[data-testid="stFileUploader"] button::after {
    content: "Browse files";
    font-size: 14px !important;
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
}

            
/* ── Sidebar radio toggle cards ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div > div > label {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 0.85rem !important;
    margin-bottom: 6px !important;
    transition: border-color 0.15s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > div > label:has(input:checked) {
    border-color: var(--border-focus) !important;
    background: var(--accent-dim) !important;
}


::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 3px; }

/* ── Reusable section-rule utility ── */
.sec-rule {
    display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;
}
.sec-rule-label {
    font-family: 'Geist', sans-serif; font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-hint); white-space: nowrap;
}
.sec-rule-line {
    flex: 1; height: 1px; background: var(--section-rule);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    try:
        with open("trial_matching_bundle.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model_bundle = load_model()

# =====================================================
# SESSION STATE
# =====================================================

if "start_app" not in st.session_state:
    st.session_state.start_app = False

# =====================================================
# LANDING PAGE
# =====================================================

if not st.session_state.start_app:

    # Hide Streamlit padding so the landing fills the viewport
    st.markdown("""
    <style>
    .block-container { padding: 0 !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    # Self-contained landing — own fonts, own media query, no var() inheritance needed
    components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* Light defaults */
  :root {
    --bg:           #F7F6F2;
    --glow:         rgba(10,155,90,0.06);
    --badge-bg:     rgba(10,155,90,0.09);
    --badge-border: rgba(10,155,90,0.22);
    --badge-text:   #0A7A46;
    --dot:          #0A9B5A;
    --dot-shadow:   rgba(10,155,90,0.40);
    --headline:     #111110;
    --accent:       #0A7A46;
    --sub:          #6B6A66;
    --chip-bg:      rgba(0,0,0,0.04);
    --chip-border:  rgba(0,0,0,0.09);
    --chip-text:    #3D3C38;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg:           #0C0C0E;
      --glow:         rgba(62,224,154,0.07);
      --badge-bg:     rgba(62,224,154,0.10);
      --badge-border: rgba(62,224,154,0.22);
      --badge-text:   #3EE09A;
      --dot:          #3EE09A;
      --dot-shadow:   rgba(62,224,154,0.50);
      --headline:     #F2F0EB;
      --accent:       #3EE09A;
      --sub:          #888886;
      --chip-bg:      rgba(255,255,255,0.03);
      --chip-border:  rgba(255,255,255,0.08);
      --chip-text:    #BCBAB4;
    }
  }

  body {
    background: var(--bg);
    font-family: 'Geist', -apple-system, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 2rem 3rem;
    position: relative;
    overflow: hidden;
  }

  .glow {
    position: absolute; top: 10%; left: 50%; transform: translateX(-50%);
    width: 640px; height: 400px; border-radius: 50%;
    background: radial-gradient(ellipse, var(--glow) 0%, transparent 70%);
    pointer-events: none;
  }

  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--badge-bg); border: 1px solid var(--badge-border);
    border-radius: 100px; padding: 5px 14px; margin-bottom: 2rem;
  }
  .badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--dot); box-shadow: 0 0 8px var(--dot-shadow);
    display: inline-block;
  }
  .badge-text {
    font-size: 12px; font-weight: 500;
    color: var(--badge-text); letter-spacing: 0.04em;
  }

  h1 {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: clamp(40px, 7vw, 72px);
    font-weight: 400;
    color: var(--headline);
    line-height: 1.08;
    letter-spacing: -0.03em;
    max-width: 780px;
    margin: 0 auto 1.25rem;
  }
  h1 em { color: var(--accent); font-style: italic; }

  .sub {
    font-size: 17px; font-weight: 300;
    color: var(--sub); max-width: 500px;
    line-height: 1.7; margin: 0 auto 2.75rem;
  }

  .chips {
    display: flex; gap: 0.85rem; flex-wrap: wrap;
    justify-content: center; margin-bottom: 0;
  }
  .chip {
    background: var(--chip-bg); border: 1px solid var(--chip-border);
    border-radius: 10px; padding: 0.75rem 1.2rem;
    display: flex; align-items: center; gap: 9px;
  }
  .chip-icon { font-size: 17px; }
  .chip-text {
    font-size: 13px; font-weight: 500; color: var(--chip-text);
  }
</style>
</head>
<body>
  <div class="glow"></div>

  <div class="badge">
    <span class="badge-dot"></span>
    <span class="badge-text">AI-Powered Clinical Trial Matching System</span>
  </div>

  <h1>
    Match patients to the right<br>
    <em>clinical trials</em>
  </h1>

  <p class="sub">
    Upload patient reports or enter data manually.
    Our NLP pipeline extracts medical entities and predicts
    trial eligibility in seconds.
  </p>
  </div>
</body>
</html>
""", height=520, scrolling=False)

    _, center, _ = st.columns([2, 1, 2])
    with center:
        if st.button("Begin Matching  →", use_container_width=True):
            st.session_state.start_app = True
            st.rerun()

    st.stop()

# =====================================================
# HELPERS
# =====================================================
def _is_dark_mode():
    theme = st.get_option("theme.base")
    return theme == "dark"

def compute_age(dob: date) -> int:
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def parse_kv_labs(raw: str) -> dict:
    result = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Za-z][^\:=\d]*)[\:=\s]+(\S+)', line)
        if m:
            result[m.group(1).strip()] = m.group(2).strip()
    return result


# =====================================================
# PDF EXTRACTION 
# =====================================================

def extract_patient_info(uploaded_file):
    text = ""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception as e:
            st.error(f"PDF Extraction Error: {e}")

    elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]):
        try:
            from PIL import Image
            import pytesseract, io
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            st.warning(f"Image OCR note: {e}")

    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            st.warning(f"DOCX read note: {e}")

    # rest of your regex extraction on `text` stays unchanged ...

    extracted = {}

    dob_match = re.search(r'DOB[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
    if dob_match:
        extracted["DOB"] = dob_match.group(1).strip()
    else:
        age_match = re.search(r'(\d+)-year-old', text)
        if age_match:
            extracted["Age"] = age_match.group(1)

    if "Female" in text:
        extracted["Gender"] = "Female"
    elif "Male" in text:
        extracted["Gender"] = "Male"

    diagnosis_match = re.search(
        r'(?:Diagnosis|Primary Diagnosis|Impression|Assessment)[:\s]+([^\n]{5,80})',
        text, re.IGNORECASE
    )
    if diagnosis_match:
        extracted["Primary Diagnosis"] = diagnosis_match.group(1).strip()

    stage_match = re.search(r'Stage\s*(I|II|III|IV)', text, re.IGNORECASE)
    if stage_match:
        extracted["Disease Stage"] = stage_match.group(1)

    ecog_match = re.search(r'ECOG\s*(\d)', text)
    if ecog_match:
        extracted["ECOG"] = ecog_match.group(1)

    vitals_patterns = {
        "Blood Pressure": r'(?:BP|Blood Pressure)[:\s]+(\d{2,3}/\d{2,3})',
        "Heart Rate":     r'(?:HR|Heart Rate|Pulse)[:\s]+(\d{2,3})\s*bpm',
        "Temperature":    r'(?:Temp|Temperature)[:\s]+(\d{2,3}\.?\d*)',
        "SpO2":           r'(?:SpO2|Oxygen Saturation)[:\s]+(\d{2,3})',
        "Weight":         r'(?:Weight|Wt)[:\s]+(\d{2,3}\.?\d*)',
        "Height":         r'(?:Height|Ht)[:\s]+(\d{2,3}\.?\d*)',
    }
    for vital, pattern in vitals_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            extracted[vital] = m.group(1)

    lab_patterns = {
        "Hemoglobin": r'Hemoglobin\s*(\d+\.?\d*)',
        "Platelets":  r'Platelets\s*(\d+\.?\d*)',
        "WBC":        r'WBC\s*(\d+\.?\d*)',
        "Creatinine": r'Creatinine\s*(\d+\.?\d*)',
        "eGFR":       r'eGFR\s*(\d+\.?\d*)',
        "AST":        r'AST\s*(\d+\.?\d*)',
        "ALT":        r'ALT\s*(\d+\.?\d*)',
        "Albumin":    r'Albumin\s*(\d+\.?\d*)',
        "PSA":        r'PSA\s*(\d+\.?\d*)',
        "LDH":        r'LDH\s*(\d+\.?\d*)',
        "HbA1c":      r'HbA1c\s*(\d+\.?\d*)',
        "TSH":        r'TSH\s*(\d+\.?\d*)',
        "BNP":        r'BNP\s*(\d+\.?\d*)',
    }
    labs_found = {}
    for lab, pattern in lab_patterns.items():
        m = re.search(pattern, text)
        if m:
            labs_found[lab] = m.group(1)
    if labs_found:
        extracted["Lab Values"] = labs_found

    treatment_match = re.search(
        r'Prior Treatments(.*?)(Past Medical History|Current Medications|$)',
        text, re.DOTALL
    )
    if treatment_match:
        extracted["Prior Treatments"] = (
            treatment_match.group(1).replace("\n", " ").strip()
        )

    return extracted, text


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem; display:flex; align-items:center; gap:10px;">
        <div style="
            width:32px; height:32px; border-radius:8px;
            background:linear-gradient(135deg, var(--accent-light), var(--accent));
            display:flex; align-items:center; justify-content:center;
            font-size:16px; flex-shrink:0;
        ">🧬</div>
        <div>
            <div style="font-family:'Geist',sans-serif; font-size:14px; font-weight:600;
                        color:var(--text); letter-spacing:-0.02em;">TrialMatch</div>
            <div style="font-family:'Geist',sans-serif; font-size:11px;
                        color:var(--text-hint); margin-top:1px;">Clinical AI v1.0</div>
        </div>
    </div>
    <div style="height:1px; background:var(--border-md); margin-bottom:1.25rem;"></div>
    <div style="font-family:'Geist',sans-serif; font-size:10px; font-weight:600;
                letter-spacing:0.08em; text-transform:uppercase;
                color:var(--text-hint); margin-bottom:10px;">Input Method</div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Input Method",
        ["Manual Entry", "Upload Report"],
        label_visibility="collapsed"
    )


# =====================================================
# PAGE HEADER
# =====================================================

st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="font-family:'Geist',sans-serif; font-size:11px; font-weight:600;
                letter-spacing:0.08em; text-transform:uppercase;
                color:var(--accent); margin-bottom:6px;">
        Patient Intake
    </div>
    <h1 style="
        font-family:'Instrument Serif',Georgia,serif !important;
        font-size:clamp(28px,4vw,42px) !important; font-weight:400 !important;
        color:var(--text) !important; letter-spacing:-0.03em !important;
        line-height:1.1 !important; margin:0 0 8px !important;
    ">Clinical Trial Matching</h1>

</div>
""", unsafe_allow_html=True)

# =====================================================
# SHARED STATE + UTILITIES
# =====================================================

patient_data  = {}
uploaded_file = None
results       = None


def sec_header(label: str):
    st.markdown(f"""
    <div class="sec-rule">
        <span class="sec-rule-label">{label}</span>
        <span class="sec-rule-line"></span>
    </div>
    """, unsafe_allow_html=True)


def spacer(rem: float = 1.5):
    st.markdown(f'<div style="margin:{rem}rem 0;"></div>', unsafe_allow_html=True)


def inline_divider():
    st.markdown(
        '<div style="height:1px; background:var(--divider); margin:2rem 0 1.5rem;"></div>',
        unsafe_allow_html=True
    )


def success_banner(msg: str):
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:8px;
        background:var(--accent-dim); border:1px solid var(--border-focus);
        border-radius:var(--radius); padding:10px 16px; margin-bottom:1.5rem;
    ">
        <span style="font-size:13px; color:var(--accent);">✓</span>
        <span style="font-family:'Geist',sans-serif; font-size:13px;
                     color:var(--accent); font-weight:500;">{msg}</span>
    </div>
    """, unsafe_allow_html=True)


def warning_banner(msg: str):
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:8px;
        background:var(--amber-dim); border:1px solid rgba(180,83,9,0.22);
        border-radius:var(--radius); padding:10px 16px; margin-top:0.75rem;
    ">
        <span style="font-size:13px; color:var(--amber);">⚠</span>
        <span style="font-family:'Geist',sans-serif; font-size:13px;
                     color:var(--amber); font-weight:500;">{msg}</span>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# MANUAL ENTRY
# =====================================================

if input_mode == "Manual Entry":

    # Demographics
    sec_header("Demographics")
    dem_col1, dem_col2, dem_col3, dem_col4 = st.columns([1.2, 1, 1, 1])

    with dem_col1:
        dob = st.date_input(
            "Date of Birth", value=None,
            min_value=date(1900, 1, 1), max_value=date.today()
        )
        if dob:
            computed_age = compute_age(dob)
            st.caption(f"↳ {computed_age} years old")
            patient_data["DOB"] = dob.strftime("%Y-%m-%d")
            patient_data["Age"] = computed_age

    with dem_col2:
        patient_data["Gender"] = st.selectbox(
            "Biological Sex", ["", "Male", "Female", "Other"], index=0
        )

    with dem_col3:
        patient_data["ECOG"] = st.number_input(
            "ECOG Score", min_value=0, max_value=4, step=1
        )

    with dem_col4:
        patient_data["Disease Stage"] = st.selectbox(
            "Disease Stage", ["", "I", "II", "III", "IV"]
        )

    spacer()

    # Diagnosis
    sec_header("Diagnosis")
    diag_col1, diag_col2 = st.columns([1.4, 1])

    with diag_col1:
        patient_data["Primary Diagnosis"] = st.text_input(
            "Primary Diagnosis",
            placeholder="e.g. Non-small cell lung cancer, Stage III breast cancer"
        )

    with diag_col2:
        patient_data["HER2 Status"] = st.selectbox(
            "HER2 Status", ["", "Positive", "Negative", "Unknown"]
        )

    spacer()

    # Vitals
    sec_header("Vital Signs")
    v1, v2, v3 = st.columns(3)

    with v1:
        bp = st.text_input("Blood Pressure", placeholder="120/80 mmHg")
        if bp:
            patient_data["Blood Pressure"] = bp

    with v2:
        hr = st.number_input("Heart Rate", min_value=0, max_value=300,
                             step=1, value=None, placeholder="bpm")
        if hr is not None:
            patient_data["Heart Rate"] = hr

    with v3:
        spo2 = st.number_input("SpO₂", min_value=0, max_value=100,
                               step=1, value=None, placeholder="%")
        if spo2 is not None:
            patient_data["SpO2"] = spo2

    spacer(0.75)
    v4, v5, v6 = st.columns(3)

    with v4:
        temp = st.number_input("Temperature", min_value=0.0, max_value=45.0,
                               step=0.1, format="%.1f", value=None, placeholder="°C")
        if temp is not None:
            patient_data["Temperature"] = temp

    with v5:
        weight = st.number_input("Weight", min_value=0.0, max_value=300.0,
                                 step=0.1, format="%.1f", value=None, placeholder="kg")
        if weight is not None:
            patient_data["Weight"] = weight

    with v6:
        height = st.number_input("Height", min_value=0.0, max_value=250.0,
                                 step=0.1, format="%.1f", value=None, placeholder="cm")
        if height is not None:
            patient_data["Height"] = height

    spacer()

    # Lab Values
    sec_header("Laboratory Values")
    st.markdown("""
    <p style="font-family:'Geist',sans-serif; font-size:12px; color:var(--text-hint);
              margin-bottom:0.75rem; margin-top:-0.25rem;">
        One lab per line &nbsp;·&nbsp;
        format: <code style="font-family:'Geist Mono',monospace; font-size:11px;
        background:var(--bg-elevated); padding:1px 6px; border-radius:4px;
        color:var(--text-muted);">Key: Value</code>
    </p>
    """, unsafe_allow_html=True)

    raw_labs = st.text_area(
        "Lab Values", height=150,
        placeholder=(
            "Hemoglobin: 11.2\nPlatelets: 210\nWBC: 5.3\n"
            "Creatinine: 0.9\nHbA1c: 7.4\nTSH: 2.1"
        ),
        label_visibility="collapsed"
    )

    parsed_labs = parse_kv_labs(raw_labs)
    if parsed_labs:
        patient_data["Lab Values"] = parsed_labs
        patient_data.update(parsed_labs)

    spacer()

    # Treatment History
    sec_header("Treatment History")
    patient_data["Prior Treatments"] = st.text_area(
        "Prior Treatments / Therapy History", height=100,
        placeholder=(
            "e.g. Paclitaxel + carboplatin (2021–2022), "
            "trastuzumab, radiotherapy, metformin"
        )
    )

# =====================================================
# FILE UPLOAD
# =====================================================

else:

    sec_header("Upload Report")

    uploaded_file = st.file_uploader(
    "Upload PDF, image, or text patient report",
    type=["pdf", "txt", "png", "jpg", "jpeg", "webp", "tiff", "docx"],
    label_visibility="collapsed"
    )

    if uploaded_file is not None:
        success_banner(f"{uploaded_file.name} uploaded successfully")
        patient_data, extracted_text = extract_patient_info(uploaded_file)

# =====================================================
# PATIENT DATA PREVIEW
# =====================================================

if patient_data:

    inline_divider()
    sec_header("Patient Summary")

    display_data = {}
    for k, v in patient_data.items():
        if k == "Lab Values" and isinstance(v, dict):
            for lab_k, lab_v in v.items():
                display_data[lab_k] = str(lab_v)
        else:
            display_data[k] = str(v)

    patient_df = pd.DataFrame(
        [(k, v) for k, v in display_data.items()
         if v not in ("", "0", "0.0", "Not Applicable")],
        columns=["Field", "Value"]
    )

    st.dataframe(
        patient_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Field": st.column_config.TextColumn("Field", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="large"),
        }
    )

# =====================================================
# MATCH BUTTON
# =====================================================

inline_divider()

if input_mode == "Upload Report" and uploaded_file is not None:

    if st.button("🔍  Find Matching Trials", use_container_width=True):
        with st.spinner("Running AI Clinical Trial Pipeline…"):
            patient_data, results = run_pipeline(uploaded_file)

        success_banner("Matching completed successfully")
        inline_divider()
        html_output = render_results(results)
        components.html(html_output, height=900, scrolling=True)

elif input_mode == "Manual Entry":

    if st.button("🔍  Find Matching Trials", use_container_width=True):

        if not patient_data.get("Primary Diagnosis"):
            warning_banner("Please enter at least a Primary Diagnosis before matching.")
        else:
            with st.spinner("Running AI Clinical Trial Pipeline…"):
                patient_data, results = run_pipeline(patient_data)

            success_banner("Matching completed successfully")
            inline_divider()
            html_output = render_results(results)
            components.html(html_output, height=900, scrolling=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:4rem; padding-top:1.5rem;
    border-top:1px solid var(--divider);
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:0.5rem;
">
    <span style="font-family:'Geist',sans-serif; font-size:12px; color:var(--text-hint);">
        TrialMatch &nbsp;·&nbsp; AI Clinical Research Platform
    </span>
    <span style="font-family:'Geist',sans-serif; font-size:12px; color:var(--text-hint);">
        Results are AI-generated and for research purposes only.
        Verify with a qualified clinical team.
    </span>
</div>
""", unsafe_allow_html=True)
