# src/api_client.py
# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov REST API v2 wrapper.
# Handles single-trial fetch, paginated search, retries, and rate limiting.
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
from typing import Optional, Any, Dict

import requests

log = logging.getLogger(__name__)

# Every section of the v2 API response we want returned
_FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.descriptionModule",
    "protocolSection.conditionsModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.eligibilityModule",
    "protocolSection.statusModule",
    "protocolSection.designModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.outcomesModule",
    "protocolSection.contactsLocationsModule",
    "derivedSection.conditionBrowseModule",
    "derivedSection.interventionBrowseModule",
])


class CTAPIClient:
    """Thin wrapper around the ClinicalTrials.gov v2 REST API."""

    def __init__(self, base: str, delay: float, retries: int, timeout: int):
        self.base    = base.rstrip("/")
        self.delay   = delay
        self.retries = retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ── Internal ──────────────────────────────────────────────

    def _get(self, url: str, params: dict) -> Optional[dict]:
        for attempt in range(self.retries):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 404:
                    log.warning(f"404 Not found: {url}")
                    return None
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as exc:
                log.warning(f"HTTP {exc.response.status_code} on attempt {attempt+1}: {url}")
            except requests.ConnectionError:
                log.warning(f"Connection error on attempt {attempt+1}: {url}")
            except requests.Timeout:
                log.warning(f"Timeout on attempt {attempt+1}: {url}")
            except Exception as exc:
                log.warning(f"Unexpected error on attempt {attempt+1}: {exc}")

            if attempt < self.retries - 1:
                time.sleep(1.5 ** attempt)

        log.error(f"All {self.retries} attempts failed: {url}")
        return None

    # ── Public ────────────────────────────────────────────────

    def fetch_trial(self, nct_id: str) -> Optional[dict]:
        """Fetch a single trial record by NCT ID."""
        return self._get(
            f"{self.base}/studies/{nct_id}",
            {"format": "json", "fields": _FIELDS},
        )

    def search_trials(self, query: str, filters: dict, max_trials: int) -> list:
        """
        Search ClinicalTrials.gov and return a list of NCT ID strings.
        Paginates automatically until max_trials is reached or results
        are exhausted.  max_trials=0 means no limit.
        """
        params: Dict[str, Any] = {
            "format"  : "json",
            "fields"  : "protocolSection.identificationModule.nctId",
            "pageSize": 100,
        }

        if query:
            params["query.term"] = None

        # Build advanced filter string
        adv_parts = []
        if filters.get("phase"):
            adv_parts.append(f"AREA[Phase]{filters['phase']}")
        if filters.get("study_type"):
            adv_parts.append(f"AREA[StudyType]{filters['study_type']}")
        if filters.get("status"):
            adv_parts.append(f"AREA[OverallStatus]{filters['status']}")
        if filters.get("condition"):
            adv_parts.append(f"AREA[Condition]{filters['condition']}")
        if filters.get("intervention"):
            adv_parts.append(f"AREA[InterventionType]{filters['intervention']}")
        if adv_parts:
            params["filter.advanced"] = " AND ".join(adv_parts)

        ids        = []
        page_token = None
        limit      = max_trials if max_trials > 0 else float("inf")

        log.info(f"Searching: query='{query}'  filters={filters}  limit={max_trials or 'unlimited'}")

        while len(ids) < limit:
            if page_token:
                params["pageToken"] = page_token

            data = self._get(f"{self.base}/studies", params)
            if not data:
                break

            for study in data.get("studies", []):
                nid = (
                    (study.get("protocolSection") or {})
                    .get("identificationModule", {})
                    .get("nctId")
                )
                if nid:
                    ids.append(nid)
                if len(ids) >= limit:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(self.delay)

        log.info(f"Discovered {len(ids)} NCT IDs.")
        return ids
