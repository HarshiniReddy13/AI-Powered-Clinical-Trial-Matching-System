# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point.
#
#   python main.py
#
# All settings are in config/settings.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
import logging
from pathlib import Path

# Make src/ importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config.settings as cfg
from src.pipeline import ExtractionPipeline


def setup_logging():
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(cfg.LOG_FILE), encoding="utf-8"),
        ],
    )


def print_summary(results: list):
    if not results:
        print("\nNo results.")
        return

    n = len(results)
    inc   = [r["_extraction_meta"]["inclusion_count"]              for r in results]
    exc   = [r["_extraction_meta"]["exclusion_count"]              for r in results]
    labs  = [len(r["clinical_requirements"]["lab_tests"])          for r in results]
    vits  = [len(r["clinical_requirements"]["vitals"])             for r in results]
    scrs  = [len(r["clinical_requirements"]["scores"])             for r in results]
    txs   = [len(r["clinical_requirements"]["treatment_history"])  for r in results]

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"{'─'*55}")
    print(f"  Trials extracted        : {n}")
    print(f"  Avg inclusion criteria  : {sum(inc)/n:.1f}")
    print(f"  Avg exclusion criteria  : {sum(exc)/n:.1f}")
    print(f"  Avg lab tests detected  : {sum(labs)/n:.1f}")
    print(f"  Avg vitals detected     : {sum(vits)/n:.1f}")
    print(f"  Avg scores detected     : {sum(scrs)/n:.1f}")
    print(f"  Avg treatment entries   : {sum(txs)/n:.1f}")
    print(f"{'─'*55}")
    print(f"  Output file  → {cfg.OUTPUT_FILE}")
    print(f"  Error file   → {cfg.ERROR_FILE}")
    print(f"  Log file     → {cfg.LOG_FILE}")
    print(f"{'='*55}\n")

    # Print one full sample record
    sample = results[0]
    print(f"Sample record  ({sample['trial_id']}):\n")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:5000])
    if len(json.dumps(sample)) > 5000:
        print("\n  ... (truncated — open output/trials_extracted.json for full data)")


def main():
    setup_logging()
    log = logging.getLogger("main")

    log.info("=" * 55)
    log.info("  Clinical Trial NLP Extractor")
    log.info("=" * 55)

    if cfg.NCT_IDS:
        log.info(f"  Mode     : explicit IDs ({len(cfg.NCT_IDS)} trials)")
    else:
        log.info(f"  Mode     : search")
        log.info(f"  Query    : {cfg.SEARCH_QUERY!r}")
        log.info(f"  Max      : {cfg.MAX_TRIALS or 'unlimited'}")
        log.info(f"  Filters  : {cfg.SEARCH_FILTERS}")

    log.info(f"  NER      : {cfg.MODEL_MODE}")
    log.info(f"  spaCy    : {cfg.SCISPACY_MODEL}")
    log.info(f"  Output   : {cfg.OUTPUT_FILE}")
    log.info("=" * 55)

    pipeline = ExtractionPipeline(cfg)
    results  = pipeline.run()
    print_summary(results)


if __name__ == "__main__":
    main()
