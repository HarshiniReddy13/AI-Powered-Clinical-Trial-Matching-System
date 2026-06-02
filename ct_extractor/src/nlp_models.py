# src/nlp_models.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads and holds all NLP models used by the pipeline.
# Models are loaded ONCE at startup and reused for every trial.
#
# Models used:
#   BioBERT     (dmis-lab/biobert-base-cased-v1.2)   — biomedical NER
#   ClinicalBERT(emilyalsentzer/Bio_ClinicalBERT)    — clinical NER
#   scispaCy    (en_core_sci_lg or en_core_web_sm)   — scientific NER
#                                                       + sentence splitting
#                                                       + dependency parsing
# ─────────────────────────────────────────────────────────────────────────────

import logging
import subprocess
import sys

import torch
import spacy
from transformers import pipeline as hf_pipeline

log = logging.getLogger(__name__)

DEVICE = 0 if torch.cuda.is_available() else -1

_BERT_IDS = {
    "biobert":  "dmis-lab/biobert-base-cased-v1.2",
    "clinbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def _load_bert(key: str):
    mid = _BERT_IDS[key]
    log.info(f"Loading {key}  ({mid})  on {'GPU' if DEVICE == 0 else 'CPU'} ...")
    try:
        pipe = hf_pipeline(
            "token-classification",
            model=mid,
            tokenizer=mid,
            aggregation_strategy="simple",
            device=DEVICE,
        )
        log.info(f"  {key} ready.")
        return pipe
    except Exception as exc:
        log.warning(f"  Could not load {key}: {exc}")
        return None


def _load_scispacy(model_name: str):
    """Load scispaCy model; auto-fallback to en_core_web_sm."""
    for name in (model_name, "en_core_web_sm"):
        try:
            nlp = spacy.load(name)
            log.info(f"Loaded spaCy model: {name}")
            return nlp
        except OSError:
            if name == model_name:
                log.warning(
                    f"Model '{model_name}' not found. "
                    "Run setup.py to install it. Trying en_core_web_sm ..."
                )
            else:
                log.warning("en_core_web_sm not found. Downloading now ...")
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                )
                return spacy.load("en_core_web_sm")
    raise RuntimeError("No spaCy model could be loaded.")


class NLPModels:
    """Container for all loaded NLP models. Created once per run."""

    def __init__(self):
        self.bert_pipes: dict = {}
        self.spacy_nlp        = None

    @classmethod
    def load(cls, mode: str, scispacy_model: str) -> "NLPModels":
        obj = cls()
        # BERT pipelines
        for key in (["biobert", "clinbert"] if mode == "both" else [mode]):
            pipe = _load_bert(key)
            if pipe:
                obj.bert_pipes[key] = pipe

        if not obj.bert_pipes:
            log.warning(
                "No BERT NER model loaded. "
                "Extraction will rely on scispaCy only."
            )

        # scispaCy
        obj.spacy_nlp = _load_scispacy(scispacy_model)
        return obj

    # ── NER methods ───────────────────────────────────────────

    def bert_ner(self, text: str, max_chars: int = 8000) -> list:
        """
        Run all loaded BERT models on `text`.
        Chunks at word boundaries to stay within 512-token limit.
        Merges entity spans from all models; deduplicates by (start, end).
        Returns list of dicts with keys: text, label, score, start, end, model.
        """
        if not self.bert_pipes:
            return []

        text     = text[:max_chars]
        entities = []
        seen     = set()

        for model_name, pipe in self.bert_pipes.items():
            pos = 0
            while pos < len(text):
                end = min(pos + 450, len(text))
                if end < len(text):
                    sp = text.rfind(" ", pos, end)
                    if sp > pos:
                        end = sp
                chunk = text[pos:end]

                if chunk.strip():
                    try:
                        for ent in pipe(chunk):
                            span = (pos + ent["start"], pos + ent["end"])
                            key  = (span[0], span[1], model_name)
                            if key not in seen:
                                seen.add(key)
                                entities.append({
                                    "text":   ent["word"],
                                    "label":  ent.get("entity_group",
                                                       ent.get("entity", "")),
                                    "score":  round(float(ent["score"]), 3),
                                    "start":  span[0],
                                    "end":    span[1],
                                    "model":  model_name,
                                })
                    except Exception as exc:
                        log.debug(f"BERT chunk error ({model_name}): {exc}")
                pos = end

        return entities

    def scispacy_doc(self, text: str):
        """Run scispaCy pipeline and return a spacy Doc."""
        return self.spacy_nlp(text[:60_000])
