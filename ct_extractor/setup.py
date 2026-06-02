# setup.py
# ─────────────────────────────────────────────────────────────────────────────
# Run once before main.py to install all dependencies.
#
#   python setup.py
#
# This installs Python packages AND downloads the NLP model weights.
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
import importlib


def run(cmd, label):
    print(f"\n[setup] {label} ...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[setup] WARNING: '{label}' returned non-zero exit code.")
    else:
        print(f"[setup] OK: {label}")


def check(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


print("=" * 60)
print("  Clinical Trial Extractor — Setup")
print("=" * 60)

pip = f'"{sys.executable}" -m pip install'

# ── Core packages ─────────────────────────────────────────────
run(f'{pip} --upgrade pip', "Upgrade pip")
run(f'{pip} transformers==4.40.0', "transformers")
run(f'{pip} requests==2.31.0 tqdm==4.66.4 numpy==1.26.4 pydantic==2.7.0', "requests / tqdm / numpy / pydantic")

# ── PyTorch (CPU build — works everywhere) ───────────────────
# If you have an NVIDIA GPU and CUDA installed, replace this with:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
run(f'{pip} torch --index-url https://download.pytorch.org/whl/cpu', "PyTorch (CPU)")

# ── spaCy + scispaCy ─────────────────────────────────────────
run(f'{pip} spacy==3.7.4 scispacy==0.5.4', "spaCy + scispaCy")

# ── scispaCy large scientific NER model (~820 MB) ────────────
SCISPACY_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
    "en_core_sci_lg-0.5.4.tar.gz"
)
run(f'{pip} {SCISPACY_URL}', "scispaCy en_core_sci_lg model")

# ── spaCy english fallback model (small, fast) ───────────────
run(f'"{sys.executable}" -m spacy download en_core_web_sm', "spaCy en_core_web_sm")

# ── Verification ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Verification")
print("=" * 60)

checks = [
    ("transformers", "transformers"),
    ("torch",        "torch"),
    ("spacy",        "spacy"),
    ("scispacy",     "scispacy"),
    ("requests",     "requests"),
    ("tqdm",         "tqdm"),
]

all_ok = True
for label, pkg in checks:
    ok = check(pkg)
    status = "✓" if ok else "✗  MISSING"
    print(f"  {status}  {label}")
    if not ok:
        all_ok = False

# Check scispaCy model
try:
    import spacy
    spacy.load("en_core_sci_lg")
    print("  ✓  en_core_sci_lg model")
except OSError:
    print("  ✗  en_core_sci_lg  (will fall back to en_core_web_sm)")

print("\n" + "=" * 60)
if all_ok:
    print("  Setup complete. You can now run:  python main.py")
else:
    print("  Some packages failed. Check the output above.")
print("=" * 60)
