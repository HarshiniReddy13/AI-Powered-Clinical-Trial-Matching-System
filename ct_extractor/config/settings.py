# config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# EDIT THIS FILE to control what gets extracted and where output goes.
# Do NOT edit any file inside src/.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Output paths ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "trials_extracted.json"
ERROR_FILE  = OUTPUT_DIR / "errors.json"
LOG_FILE    = OUTPUT_DIR / "run.log"

# ── ClinicalTrials.gov API ────────────────────────────────────────────────────
CT_API_BASE   = "https://clinicaltrials.gov/api/v2"
API_DELAY_SEC = 0.3     # seconds between requests (be polite to the server)
API_RETRIES   = 3
API_TIMEOUT   = 30

# ─────────────────────────────────────────────────────────────────────────────
# TRIAL SOURCE  —  choose ONE option below
# ─────────────────────────────────────────────────────────────────────────────

# OPTION A: Give explicit NCT IDs
# Set NCT_IDS to a non-empty list and the search settings are ignored.
NCT_IDS = []
# Example:
# NCT_IDS = ["NCT00610792", "NCT01026194", "NCT04280848"]

# OPTION B: Search ClinicalTrials.gov (used when NCT_IDS is empty)
SEARCH_QUERY = None

# Optional filters — set any to None to skip it
SEARCH_FILTERS = {
    "phase"       : None,   # "PHASE1" | "PHASE2" | "PHASE3" | "PHASE4"
    "study_type"  : None,   # "INTERVENTIONAL" | "OBSERVATIONAL"
    "status"      : None,   # "COMPLETED" | "RECRUITING" | "ACTIVE_NOT_RECRUITING"
    "condition"   : None,   # e.g. "breast cancer"
    "intervention": None,   # e.g. "drug"
}

# Maximum number of trials to process (0 = no limit / fetch all)
MAX_TRIALS = 20000

# ─────────────────────────────────────────────────────────────────────────────
# NLP MODELS
# ─────────────────────────────────────────────────────────────────────────────

# Which transformer NER models to use:
#   "biobert"   →  dmis-lab/biobert-base-cased-v1.2
#   "clinbert"  →  emilyalsentzer/Bio_ClinicalBERT
#   "both"      →  run both and merge results  (recommended, best coverage)
MODEL_MODE = "both"

# scispaCy model name.
# After setup.py runs, "en_core_sci_lg" will be installed.
# If that fails, it automatically falls back to "en_core_web_sm".
SCISPACY_MODEL = "en_core_sci_lg"

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
SAVE_EVERY_N   = 10      # save output JSON every N trials (resume support)
MAX_NER_CHARS  = 8000    # max characters fed to transformer NER per trial
