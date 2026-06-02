# src/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrates the full extraction pipeline.
# Supports resume (skips already-done trials) and incremental JSON saves.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.api_client           import CTAPIClient
from src.raw_extractor        import extract_raw_fields
from src.eligibility_splitter import split_eligibility
from src.nlp_models           import NLPModels
from src.clinical_extractor   import extract_clinical_requirements
from src.schema_builder       import build_record

log = logging.getLogger(__name__)


def _atomic_save(data: list, path: str):
    """Write JSON atomically (tmp file + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    log.debug(f"Saved {len(data)} records → {path}")


class ExtractionPipeline:

    def __init__(self, cfg):
        self.cfg = cfg
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        self.api = CTAPIClient(
            base    = cfg.CT_API_BASE,
            delay   = cfg.API_DELAY_SEC,
            retries = cfg.API_RETRIES,
            timeout = cfg.API_TIMEOUT,
        )
        self.nlp = NLPModels.load(cfg.MODEL_MODE, cfg.SCISPACY_MODEL)

    # ─────────────────────────────────────────────────────────

    def process_one(self, nct_id: str) -> Optional[dict]:
        """Run the full pipeline for one NCT ID. Returns None on failure."""

        # 1. Fetch from API
        study = self.api.fetch_trial(nct_id)
        if not study:
            return None

        # 2. Flatten API response
        raw = extract_raw_fields(study)
        if not raw["nct_id"]:
            raw["nct_id"] = nct_id

        # 3. Split eligibility using scispaCy sentence segmentation
        eligibility = split_eligibility(
            raw["eligibility_criteria_raw"],
            self.nlp.spacy_nlp,
        )

        # 4. Build text for NER
        ner_text = (
            " ".join(eligibility["inclusion"] + eligibility["exclusion"])
            + " " + raw.get("brief_summary", "")
        )

        # 5. Transformer NER (BioBERT + ClinicalBERT)
        bert_ents = self.nlp.bert_ner(ner_text, self.cfg.MAX_NER_CHARS)

        # 6. scispaCy NER + dependency parse
        elig_text = " ".join(
            eligibility["inclusion"] + eligibility["exclusion"]
        )
        sci_doc = self.nlp.scispacy_doc(elig_text)

        # 7. Extract clinical requirements (NLP-driven, no hardcoded lists)
        clinical_req = extract_clinical_requirements(
            raw           = raw,
            eligibility   = eligibility,
            bert_entities = bert_ents,
            sci_doc       = sci_doc,
        )

        # 8. Build output record
        meta = {
            "bert_models":       list(self.nlp.bert_pipes.keys()),
            "scispacy_model":    self.nlp.spacy_nlp.meta.get("name", "?"),
            "bert_entity_count": len(bert_ents),
            "sci_entity_count":  len(sci_doc.ents),
        }
        return build_record(raw, eligibility, clinical_req, meta)

    # ─────────────────────────────────────────────────────────

    def run(self) -> list:
        cfg = self.cfg

        # Resolve trial IDs
        if cfg.NCT_IDS:
            ids = list(cfg.NCT_IDS)
            log.info(f"Processing {len(ids)} explicit NCT IDs.")
        else:
            ids = self.api.search_trials(
                query      = cfg.SEARCH_QUERY,
                filters    = cfg.SEARCH_FILTERS,
                max_trials = cfg.MAX_TRIALS,
            )

        # Resume support: load existing output
        results:  list = []
        done_ids: set  = set()
        errors:   list = []

        out_path = str(cfg.OUTPUT_FILE)
        if Path(out_path).exists():
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["trial_id"] for r in results}
            log.info(f"Resuming — {len(done_ids)} trials already processed.")

        log.info(f"Trials to process: {len(ids) - len(done_ids)} remaining.")

        for nct_id in tqdm(ids, desc="Extracting", unit="trial"):
            if nct_id in done_ids:
                continue

            try:
                record = self.process_one(nct_id)
                if record:
                    results.append(record)
                    done_ids.add(nct_id)
                else:
                    errors.append({"nct_id": nct_id, "error": "no data returned"})
            except Exception as exc:
                log.error(f"{nct_id} failed: {exc}", exc_info=False)
                errors.append({"nct_id": nct_id, "error": str(exc)})

            if len(results) % cfg.SAVE_EVERY_N == 0 and results:
                _atomic_save(results, out_path)

            time.sleep(cfg.API_DELAY_SEC)

        # Final save
        _atomic_save(results, out_path)

        if errors:
            _atomic_save(errors, str(cfg.ERROR_FILE))
            log.warning(f"{len(errors)} errors saved → {cfg.ERROR_FILE}")

        log.info(f"Done. {len(results)} records → {out_path}")
        return results
