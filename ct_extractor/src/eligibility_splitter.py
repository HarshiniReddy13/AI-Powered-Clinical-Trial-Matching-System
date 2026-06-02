# src/eligibility_splitter.py
# ─────────────────────────────────────────────────────────────────────────────
# Splits the raw eligibility criteria string (as returned by the API) into
# individual inclusion and exclusion criterion sentences.
#
# Approach:
#   1. Find Inclusion/Exclusion section headers with regex
#   2. Within each block, use scispaCy sentence segmentation to split
#      the text into proper sentences (not just newlines)
#   3. Clean away bullet characters and numbering
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging

log = logging.getLogger(__name__)

# Section header patterns
_INC_HDR = re.compile(r"inclusion\s+criteria\s*[:\-]?\s*", re.I)
_EXC_HDR = re.compile(r"exclusion\s+criteria\s*[:\-]?\s*", re.I)

# Leading list markers to strip from each item
_BULLET_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,3}[\.\)]\s*"      # 1. or 1)
    r"|[a-zA-Z][\.\)]\s*"    # a. or a)
    r"|[-•*·▪◦‣⁃]\s*"        # unicode bullet chars
    r")\s*",
    re.UNICODE,
)


def _clean(text: str) -> str:
    return _BULLET_RE.sub("", text).strip()


def _segment_block(block: str, nlp) -> list:
    """
    Turn one criteria block (inclusion or exclusion) into a list of
    individual criterion strings.

    Strategy:
      - Split on blank lines and explicit list markers first
        (handles the numbered list format common in ClinicalTrials.gov)
      - Then run scispaCy sentence segmentation on each chunk to catch
        multi-sentence items
      - Filter out very short fragments (< 10 chars)
    """
    if not block.strip():
        return []

    # Step 1: split on newlines and identify list item boundaries
    lines       = re.split(r"\n+", block)
    raw_items   = []
    buffer      = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer.strip():
                raw_items.append(buffer.strip())
                buffer = ""
            continue

        # Start of a new list item?
        is_new_item = bool(re.match(
            r"^\s*(?:\d{1,3}[\.\)]|[a-zA-Z][\.\)]|[-•*·▪◦‣⁃])\s+\S",
            line,
        ))

        if is_new_item:
            if buffer.strip():
                raw_items.append(buffer.strip())
            buffer = stripped
        else:
            buffer = (buffer + " " + stripped).strip() if buffer else stripped

    if buffer.strip():
        raw_items.append(buffer.strip())

    # Step 2: scispaCy sentence segmentation on each chunk
    criteria = []
    for chunk in raw_items:
        chunk = _clean(chunk)
        if len(chunk) < 10:
            continue
        try:
            doc = nlp(chunk)
            sents = list(doc.sents)
            if len(sents) > 1:
                for sent in sents:
                    s = sent.text.strip()
                    if len(s) >= 10:
                        criteria.append(s)
            else:
                criteria.append(chunk)
        except Exception as exc:
            log.debug(f"scispaCy sentence split error: {exc}")
            criteria.append(chunk)

    return criteria


def split_eligibility(raw_text: str, nlp) -> dict:
    """
    Split raw eligibility criteria text into inclusion and exclusion lists.

    Returns:
        {
            "raw_text":  original string,
            "inclusion": [criterion, ...],
            "exclusion": [criterion, ...]
        }
    """
    if not raw_text:
        return {"raw_text": "", "inclusion": [], "exclusion": []}

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    inc_m = _INC_HDR.search(text)
    exc_m = _EXC_HDR.search(text)

    if inc_m and exc_m:
        if inc_m.start() < exc_m.start():
            inc_block = text[inc_m.end(): exc_m.start()]
            exc_block = text[exc_m.end():]
        else:
            exc_block = text[exc_m.end(): inc_m.start()]
            inc_block = text[inc_m.end():]
        inclusion = _segment_block(inc_block, nlp)
        exclusion = _segment_block(exc_block, nlp)

    elif inc_m:
        inclusion = _segment_block(text[inc_m.end():], nlp)
        exclusion = []

    elif exc_m:
        inclusion = []
        exclusion = _segment_block(text[exc_m.end():], nlp)

    else:
        # No section headers — treat everything as inclusion
        inclusion = _segment_block(text, nlp)
        exclusion = []

    return {
        "raw_text":  raw_text,
        "inclusion": inclusion,
        "exclusion": exclusion,
    }
