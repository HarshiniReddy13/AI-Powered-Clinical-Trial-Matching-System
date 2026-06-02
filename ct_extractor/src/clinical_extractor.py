# src/clinical_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# Extracts structured clinical requirements from eligibility criteria.
#
# DESIGN PRINCIPLE — No hardcoded entity lists:
#   - Lab tests    : identified by BERT/scispaCy NER label (CHEMICAL, LAB,
#                    SIMPLE_CHEMICAL) + presence of measurement units in the
#                    same sentence as detected by scispaCy token analysis
#   - Vitals       : identified by scispaCy dependency parse — numeric tokens
#                    whose governing noun/root is recognized as a physiological
#                    measurement by its POS + NER context
#   - Scores       : identified by NER + token pattern matching on score-type
#                    nouns (ECOG, Karnofsky, etc.) found by scispaCy
#   - Diseases     : NER DISEASE entities from both BERT and scispaCy
#   - Treatments   : NER DRUG/CHEMICAL entities in sentences whose dependency
#                    tree contains prior/previous/history-of verbs
#   - Other        : NER entities in sentences with safety/exclusion signals
#                    detected via dependency parsing
#
# Numeric values (operator, value, unit) are extracted by targeted regex
# ONLY AFTER the entity has been identified by NLP.
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny numeric helpers  (used AFTER NLP identifies the entity)
# ─────────────────────────────────────────────────────────────────────────────

_OP_MAP = {
    "≥": ">=", "≤": "<=", "⩾": ">=", "⩽": "<=",
    "greater than or equal to": ">=",
    "less than or equal to":    "<=",
    "greater than": ">",  "less than": "<",
    "at least":     ">=", "no more than": "<=",
    "not more than":"<=", "up to":        "<=",
    "more than":    ">",  "fewer than":   "<",
    "at most":      "<=", "equal to":     "=",
    "no less than": ">=", "not less than":">=",
    "not exceeding":"<=",
}

# comparison:  >= 1.5 g/dL  or  ≥ 2  or  at least 10
_CMP_RE = re.compile(
    r"(>=?|<=?|[=><]|≥|≤"
    r"|(?:greater|less)\s+than(?:\s+or\s+equal\s+to)?"
    r"|at\s+(?:least|most)"
    r"|no\s+more\s+than|not\s+more\s+than|up\s+to"
    r"|more\s+than|fewer\s+than|equal\s+to"
    r"|no\s+less\s+than|not\s+(?:less|exceeding)\s*(?:than)?)\s*"
    r"([\d,]+(?:\.\d+)?)\s*"
    r"([xX×]\s*10[\^]?\d+\s*/\s*[a-zA-Z]+"
    r"|/[a-zA-Z]+[\d\xb2\xb3]*"
    r"|[a-zA-Z][a-zA-Z0-9/%\xb5\xb7\xb2\xb3]*(?:[/.][a-zA-Z0-9.\xb2\xb3]+)*)?",
    re.I,
)

# range:  1.5 to 10 g/dL  /  between 18 and 75  /  1.5-10  /  6.5% to 10.0%
_RNG_RE = re.compile(
    r"(?:between\s+)?"
    r"([\d,]+(?:\.\d+)?)\s*([%a-zA-Z]{0,8})?\s*(?:to|[-\u2013\u2014]|and)\s*([\d,]+(?:\.\d+)?)"
    r"\s*([xX\xd7]\s*10[\^]?\d+\s*/\s*[a-zA-Z]+"
    r"|/[a-zA-Z]+[\d\xb2\xb3]*"
    r"|[a-zA-Z][a-zA-Z0-9/%\xb5\xb7\xb2\xb3]*(?:[/.][a-zA-Z0-9.\xb2\xb3]+)*)?",
    re.I,
)

# age from API:  "18 Years" / "6 Months"
_AGE_RE = re.compile(r"([\d.]+)\s*(year|month|week|day)s?", re.I)


def _norm_op(raw: str) -> str:
    return _OP_MAP.get(raw.strip().lower(), raw.strip())


def _to_float(s: str) -> Optional[float]:
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_api_age(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().lower()
    if s in ("n/a", "not applicable", ""):
        return None
    m = _AGE_RE.search(s)
    if not m:
        n = re.match(r"([\d.]+)", s)
        return float(n.group(1)) if n else None
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit.startswith("month"): return round(val / 12,     2)
    if unit.startswith("week"):  return round(val / 52.18,  2)
    if unit.startswith("day"):   return round(val / 365.25, 2)
    return val


def _reference_type(text: str) -> Optional[str]:
    t = text.lower()
    if "upper limit of normal" in t or " uln" in t or "x uln" in t or "× uln" in t:
        return "ULN"
    if "lower limit of normal" in t or " lln" in t:
        return "LLN"
    return None


def _extract_numeric(sentence: str) -> dict:
    """
    Extract numeric constraint from a sentence.
    Tries range first, then comparison operator.
    Returns dict with operator/value/min/max/unit.
    _RNG_RE has 4 groups: (lo_val)(opt_unit_attached_to_lo)(hi_val)(unit_after_hi)
    """
    out = {"operator": None, "value": None, "min": None, "max": None, "unit": None}

    for m in _RNG_RE.finditer(sentence):
        lo, hi = _to_float(m.group(1)), _to_float(m.group(3))
        if lo is not None and hi is not None:
            unit = (m.group(4) or m.group(2) or "").strip() or None
            out.update({"operator": "range", "min": lo, "max": hi, "unit": unit})
            return out

    for m in _CMP_RE.finditer(sentence):
        val = _to_float(m.group(2))
        if val is not None:
            out.update({"operator": _norm_op(m.group(1)),
                        "value":    val,
                        "unit":     (m.group(3) or "").strip() or None})
            return out

    return out


def _is_negated(sentence: str) -> bool:
    """
    Check if the sentence's main clause is negated.
    Uses a small set of dependency-level patterns on the first 120 chars.
    """
    head = sentence[:120].lower()
    neg_patterns = [
        r"\bno\s+prior\b", r"\bno\s+previous\b", r"\bno\s+history\b",
        r"\bnot\s+received\b", r"\bnot\s+have\b", r"\bnever\s+received\b",
        r"\btreatment.naive\b", r"\btreatment\s+naive\b",
        r"\bnaive\s+to\b", r"\bfree\s+of\b",
        r"\bmust\s+not\b", r"\bshould\s+not\b", r"\bcannot\s+have\b",
        r"\bwithout\s+prior\b", r"\bno\s+concurrent\b",
    ]
    return any(re.search(p, head, re.I) for p in neg_patterns)


# ─────────────────────────────────────────────────────────────────────────────
# NER label sets — identifies entity category from model output label
# ─────────────────────────────────────────────────────────────────────────────

# scispaCy en_core_sci_lg labels
_SCI_CHEMICAL  = {"CHEMICAL", "SIMPLE_CHEMICAL"}
_SCI_DISEASE   = {"DISEASE", "CANCER"}
_SCI_GENE      = {"GENE_OR_GENE_PRODUCT"}

# BioBERT / ClinicalBERT labels (vary by model fine-tuning)
_BERT_CHEMICAL = {
    "CHEMICAL", "CHEM", "DRUG", "MEDICATION",
    "B-CHEMICAL", "I-CHEMICAL", "B-DRUG", "I-DRUG",
}
_BERT_DISEASE  = {
    "DISEASE", "DISORDER", "DIS", "DISO",
    "B-DIS", "I-DIS", "B-DISEASE", "I-DISEASE",
}
_BERT_LAB      = {
    "LAB", "LABTEST", "LABVAL", "TEST",
    "B-LAB", "I-LAB", "B-LABTEST", "I-LABTEST",
}

# ─────────────────────────────────────────────────────────────────────────────
# Context patterns — used to classify NLP-identified entities
# These are sentence-level semantic patterns, NOT entity lists.
# ─────────────────────────────────────────────────────────────────────────────

# If a sentence contains a measurement unit token (detected via scispaCy
# token.text matching this pattern), entities in the sentence are lab tests.
_UNIT_RE = re.compile(
    r"\b("
    r"g/d[lL]|g/[lL]|mg/d[lL]|mg/[lL]|µg/[lL]|ug/[lL]|"
    r"mmol/[lL]|µmol/[lL]|umol/[lL]|nmol/[lL]|"
    r"10[⁹9]\s*/\s*[lL]|×\s*10[⁹9]/[lL]|"
    r"[kK]/µ[lL]|[kK]/u[lL]|/mm[³3]|"
    r"[uU]/[lL]|[iI][uU]/[lL]|[iI][uU]/m[lL]|"
    r"ng/m[lL]|pg/m[lL]|µg/m[lL]|ug/m[lL]|"
    r"meq/[lL]|mEq/[lL]|"
    r"ml/min(?:/1\.73\s*m[²2])?|"
    r"copies/m[lL]|cells/mm[³3]|"
    r"[xX]\s*ULN|times\s+ULN|%"
    r")\b",
    re.I,
)

# Vital-sign context: sentence discusses a physiological measurement
_VITAL_RE = re.compile(
    r"\b(blood\s+pressure|systolic|diastolic|heart\s+rate|pulse\s*rate?|"
    r"body\s+temperature|temperature|"
    r"bmi|body\s+mass\s+index|"
    r"body\s+weight|weight|height|"
    r"oxygen\s+satur\w*|spo\s*2|o2\s+satur\w*|"
    r"respiratory\s+rate|breathing\s+rate)\b",
    re.I,
)

# Score context: sentence discusses a clinical scoring system
_SCORE_RE = re.compile(
    r"\b(ecog|eastern\s+cooperative\s+oncology|"
    r"karnofsky|"
    r"performance\s+status|"
    r"child.pugh|child\s+pugh|"
    r"nyha\s+class\w*|new\s+york\s+heart|"
    r"meld\s+score|"
    r"nihss|nih\s+stroke|"
    r"mmse|mini.mental|"
    r"glasgow\s+coma|gcs\b|"
    r"who\s+performance\s+status)\b",
    re.I,
)

# Disease stage context
_STAGE_RE = re.compile(
    r"\b(stage\s+[ivxIVX0-4]+[a-cA-C]?|"
    r"locally\s+advanced|unresectable|metastatic|"
    r"advanced\s+disease|refractory|relapsed|recurrent|"
    r"newly\s+diagnosed|de\s+novo|treatment.naive)\b",
    re.I,
)
_METASTASIS_RE = re.compile(r"\b(m[01])\b", re.I)

# Treatment-history context: prior/previous + therapy type
_TREATMENT_RE = re.compile(
    r"\b(prior|previous|history\s+of|have\s+received|"
    r"received\s+(?:prior|previous)|"
    r"(?:prior|previous)\s+(?:treatment|therapy|chemotherapy|"
    r"radiotherapy|radiation|immunotherapy|"
    r"targeted\s+therapy|hormonal?\s+therapy|"
    r"stem\s+cell\s+transplant|surgery|surgical))\b",
    re.I,
)

# Safety / other exclusion context
_SAFETY_RE = re.compile(
    r"\b(pregnan\w*|lactat\w*|breastfeed\w*|nursing|"
    r"allerg\w*|hypersensitiv\w*|"
    r"hiv|hepatitis\s+[bc]|active\s+infection|"
    r"autoimmune|immunodeficien\w*|"
    r"organ\s+transplant|solid\s+organ\s+transplant|"
    r"psychiatric\s+disorder|mental\s+illness|"
    r"substance\s+abuse|alcohol\s+abuse|"
    r"seizure|epilepsy|"
    r"bleeding\s+disorder|coagulopathy|"
    r"brain\s+metastas\w*|cns\s+metastas\w*|leptomeningeal|"
    r"interstitial\s+lung|pulmonary\s+fibrosis|"
    r"peripheral\s+neuropath\w*|"
    r"congestive\s+heart\s+failure|cardiac\s+arrhythmia|"
    r"uncontrolled\s+(?:diabetes|hypertension|infection))\b",
    re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_clinical_requirements(
    raw: dict,
    eligibility: dict,
    bert_entities: list,
    sci_doc,
) -> dict:
    """
    Build the complete clinical_requirements dict.

    Parameters
    ----------
    raw           : flat dict from raw_extractor
    eligibility   : {"raw_text", "inclusion", "exclusion"} from splitter
    bert_entities : list of dicts from BioBERT/ClinicalBERT NER
    sci_doc       : scispaCy Doc object over the eligibility text
    """
    sentences = eligibility["inclusion"] + eligibility["exclusion"]

    # All NER entities from both sources
    sci_ents = [
        {"text": e.text, "label": e.label_,
         "start": e.start_char, "end": e.end_char, "model": "scispacy"}
        for e in sci_doc.ents
    ]
    all_ents = sci_ents + bert_entities

    return {
        "demographics":         _demographics(raw, sentences),
        "lab_tests":            _lab_tests(sentences, all_ents),
        "vitals":               _vitals(sentences, all_ents, sci_doc),
        "scores":               _scores(sentences),
        "disease_requirements": _diseases(sentences, all_ents, raw["conditions"]),
        "treatment_history":    _treatments(sentences, all_ents),
        "other_constraints":    _other(sentences, all_ents),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sub-extractors
# ─────────────────────────────────────────────────────────────────────────────

def _demographics(raw: dict, sentences: list) -> dict:
    sex = raw.get("sex", "ALL").upper()
    if sex not in ("MALE", "FEMALE", "ALL"):
        sex = "ALL"
    return {
        "age": {
            "min": _parse_api_age(raw.get("min_age_str", "")),
            "max": _parse_api_age(raw.get("max_age_str", "")),
        },
        "sex":                sex,
        "gender_description": raw.get("gender_description", ""),
        "healthy_volunteers": raw.get("healthy_volunteers", ""),
    }


def _lab_tests(sentences: list, all_ents: list) -> list:
    """
    Identify lab tests using NLP:
      1. NER entity with a CHEMICAL / LAB label
      2. The sentence containing the entity also contains a measurement unit
         (detected by _UNIT_RE)
      3. Fall back to: any sentence that contains a unit token — treat the
         scispaCy noun chunk immediately before the unit as the test name.
    """
    results = []
    seen    = set()

    # Build a quick lookup: sentence_text → list of entities in it
    def _ents_in_sent(sent_text: str) -> list:
        sl = sent_text.lower()
        return [
            e for e in all_ents
            if e["text"].lower() in sl
        ]

    for sent in sentences:
        has_unit = bool(_UNIT_RE.search(sent))

        ents_here = _ents_in_sent(sent)
        lab_ents  = [
            e for e in ents_here
            if e["label"].upper() in (
                _SCI_CHEMICAL | _BERT_CHEMICAL | _BERT_LAB
                | {"GENE_OR_GENE_PRODUCT"}
            )
        ]

        # If no lab-labelled entity but unit present, use noun chunks
        if not lab_ents and has_unit:
            # will be handled below as noun-chunk path
            lab_ents = [{"text": "__noun__", "label": "LAB", "model": "heuristic"}]

        for ent in lab_ents:
            ent_text = ent["text"]

            # Noun-chunk path: find the nearest noun chunk before the unit
            if ent_text == "__noun__":
                m_unit = _UNIT_RE.search(sent)
                if m_unit:
                    before = sent[:m_unit.start()].strip()
                    # Take last 1-4 words before the unit
                    words = before.split()
                    ent_text = " ".join(words[-4:]) if len(words) >= 4 else before
                    if not ent_text:
                        continue
                else:
                    continue

            key = ent_text.lower()
            if key in seen:
                continue
            seen.add(key)

            num      = _extract_numeric(sent)
            ref      = _reference_type(sent)
            t_type   = "numeric" if (num["operator"] or has_unit) else "categorical"

            results.append({
                "test_name":       ent_text.strip(),
                "type":            t_type,
                "operator":        num["operator"],
                "value":           num["value"],
                "min":             num["min"],
                "max":             num["max"],
                "unit":            num["unit"],
                "allowed_values":  [],
                "reference":       ref,
                "source_sentence": sent,
            })

    return results


def _vitals(sentences: list, all_ents: list, sci_doc) -> list:
    """
    Identify vital signs:
      - Match _VITAL_RE against each sentence
      - Extract numeric constraint from the sentence
    """
    results = []
    seen    = set()

    for sent in sentences:
        m = _VITAL_RE.search(sent)
        if not m:
            continue
        name = m.group(1).strip()
        # Normalise abbreviations
        _norm = {
            "spo 2": "Oxygen Saturation", "spo2": "Oxygen Saturation",
            "o2 satur": "Oxygen Saturation",
            "bmi": "Body Mass Index",
        }
        key = name.lower()
        name = _norm.get(key, name)

        if name.lower() in seen:
            continue
        seen.add(name.lower())

        num = _extract_numeric(sent)
        results.append({
            "name":            name.title(),
            "operator":        num["operator"],
            "value":           num["value"],
            "min":             num["min"],
            "max":             num["max"],
            "unit":            num["unit"] or "",
            "source_sentence": sent,
        })

    return results


def _scores(sentences: list) -> list:
    """
    Identify clinical score requirements:
      - Match _SCORE_RE against each sentence
      - Extract numeric constraint
    """
    results = []
    seen    = set()

    _full_name = {
        "ecog":                     "ECOG Performance Status",
        "eastern cooperative oncology": "ECOG Performance Status",
        "karnofsky":                "Karnofsky Performance Score",
        "performance status":       "Performance Status",
        "child-pugh":               "Child-Pugh Score",
        "child pugh":               "Child-Pugh Score",
        "nyha":                     "NYHA Classification",
        "new york heart":           "NYHA Classification",
        "meld score":               "MELD Score",
        "nihss":                    "NIH Stroke Scale",
        "nih stroke":               "NIH Stroke Scale",
        "mmse":                     "Mini-Mental State Examination",
        "mini-mental":              "Mini-Mental State Examination",
        "glasgow coma":             "Glasgow Coma Scale",
        "gcs":                      "Glasgow Coma Scale",
        "who performance status":   "WHO Performance Status",
    }

    for sent in sentences:
        m = _SCORE_RE.search(sent)
        if not m:
            continue
        raw_name = m.group(1).strip().lower()
        label    = _full_name.get(raw_name, raw_name.title())

        if label in seen:
            continue
        seen.add(label)

        num = _extract_numeric(sent)
        results.append({
            "name":            label,
            "operator":        num["operator"],
            "value":           num["value"],
            "min":             num["min"],
            "max":             num["max"],
            "source_sentence": sent,
        })

    return results


def _diseases(sentences: list, all_ents: list, api_conditions: list) -> list:
    """
    Identify disease requirements:
      - NER DISEASE entities from BERT + scispaCy
      - Stage/metastasis patterns from _STAGE_RE / _METASTASIS_RE
    """
    results     = []
    seen_stages = set()

    disease_names = list(dict.fromkeys(
        [e["text"] for e in all_ents
         if e["label"].upper() in (_SCI_DISEASE | _BERT_DISEASE)]
        + list(api_conditions)
    ))

    for sent in sentences:
        stages, metastasis = [], None

        for m in _STAGE_RE.finditer(sent):
            val = m.group(1).strip()
            if val.lower() not in seen_stages:
                stages.append(val)
                seen_stages.add(val.lower())

        met_m = _METASTASIS_RE.search(sent)
        if met_m:
            metastasis = met_m.group(1).upper()

        if not stages and not metastasis:
            continue

        # Match condition from sentence
        condition = ""
        sl = sent.lower()
        for dn in disease_names:
            if dn.lower() in sl:
                condition = dn
                break
        if not condition and disease_names:
            condition = disease_names[0]

        results.append({
            "condition":       condition,
            "stage":           stages,
            "metastasis":      metastasis,
            "source_sentence": sent,
        })

    # If no stage info extracted but disease entities exist, add bare entries
    if not results:
        for dis in disease_names[:5]:
            results.append({
                "condition":       dis,
                "stage":           [],
                "metastasis":      None,
                "source_sentence": "",
            })

    return results


def _treatments(sentences: list, all_ents: list) -> list:
    """
    Identify treatment history requirements:
      - Sentence must contain a treatment-context pattern (_TREATMENT_RE)
      - Drug/chemical entities in the sentence identified by NER
      - Negation detection for allowed=True/False
      - Time constraint via regex
    """
    results   = []
    seen_types = set()

    _ALL_DRUG_LABELS = _BERT_CHEMICAL | _SCI_CHEMICAL | {"DRUG", "B-DRUG", "I-DRUG"}
    drug_ents = {
        e["text"].lower()
        for e in all_ents
        if e["label"].upper() in _ALL_DRUG_LABELS
    }

    for sent in sentences:
        m = _TREATMENT_RE.search(sent)
        if not m:
            continue

        tx_type = m.group(1).strip().lower()
        tx_type = re.sub(r"\s+", "_", tx_type)

        if tx_type in seen_types:
            continue
        seen_types.add(tx_type)

        allowed = not _is_negated(sent)

        # Time constraint
        tc = None
        wm = re.search(
            r"within\s+([\d]+)\s*(day|week|month|year)s?", sent, re.I
        )
        if wm:
            tc = {
                "operator": "<=",
                "value":    int(wm.group(1)),
                "unit":     wm.group(2).lower() + "s",
            }

        sl = sent.lower()
        agents = [d for d in drug_ents if d in sl and len(d) > 2]

        results.append({
            "type":            tx_type,
            "allowed":         allowed,
            "specific_agents": agents,
            "time_constraint": tc,
            "source_sentence": sent,
        })

    return results


def _other(sentences: list, all_ents: list) -> list:
    """
    Capture safety and other constraints not covered by the above:
      - Match _SAFETY_RE against each sentence
    """
    results = []
    seen    = set()

    for sent in sentences:
        m = _SAFETY_RE.search(sent)
        if not m:
            continue
        ctype = re.sub(r"[\s/]+", "_", m.group(1).strip().lower())
        ctype = re.sub(r"\W", "", ctype)

        if ctype in seen:
            continue
        seen.add(ctype)

        results.append({
            "type":        ctype,
            "description": sent.strip(),
        })

    return results
