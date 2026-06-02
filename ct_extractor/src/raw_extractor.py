# src/raw_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
# Flattens the nested ClinicalTrials.gov v2 API JSON into a clean, flat dict.
# Every downstream module reads from this dict — nothing touches the raw API
# response directly.
# ─────────────────────────────────────────────────────────────────────────────


def _dig(d: dict, *keys, default=""):
    """Safe deeply-nested dict access. Returns `default` on any miss."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur if cur is not None else default


def extract_raw_fields(study: dict) -> dict:
    """
    Convert one ClinicalTrials.gov v2 study record into a flat dict
    containing every field the pipeline needs.
    """
    ps  = study.get("protocolSection", {}) or {}
    der = study.get("derivedSection",  {}) or {}

    id_m = ps.get("identificationModule",       {}) or {}
    des  = ps.get("descriptionModule",           {}) or {}
    con  = ps.get("conditionsModule",            {}) or {}
    arm  = ps.get("armsInterventionsModule",     {}) or {}
    eli  = ps.get("eligibilityModule",           {}) or {}
    sta  = ps.get("statusModule",                {}) or {}
    des2 = ps.get("designModule",                {}) or {}
    spo  = ps.get("sponsorCollaboratorsModule",  {}) or {}
    out  = ps.get("outcomesModule",              {}) or {}
    loc  = ps.get("contactsLocationsModule",     {}) or {}

    # ── Interventions (structured) ────────────────────────────
    interventions = []
    for iv in arm.get("interventions", []) or []:
        name = (iv.get("name") or "").strip()
        if not name:
            continue
        interventions.append({
            "name":        name,
            "type":        (iv.get("type") or "").strip(),
            "description": (iv.get("description") or "").strip(),
            "arm_labels":  iv.get("armGroupLabels", []) or [],
            "other_names": iv.get("otherNames", []) or [],
        })

    # ── Arms ─────────────────────────────────────────────────
    arms = [
        {
            "label":       ag.get("label", ""),
            "type":        ag.get("type", ""),
            "description": ag.get("description", ""),
        }
        for ag in arm.get("armGroups", []) or []
    ]

    # ── Outcomes ─────────────────────────────────────────────
    def _outcomes(lst):
        return [
            {
                "measure":     o.get("measure", ""),
                "description": o.get("description", ""),
                "time_frame":  o.get("timeFrame", ""),
            }
            for o in (lst or [])
        ]

    # ── Locations (cap at 20) ─────────────────────────────────
    locations = [
        {
            "facility": lc.get("facility", ""),
            "city":     lc.get("city", ""),
            "country":  lc.get("country", ""),
            "status":   lc.get("status", ""),
        }
        for lc in (loc.get("locations") or [])[:20]
    ]

    # ── MeSH terms from derived section ──────────────────────
    cond_browse = der.get("conditionBrowseModule",   {}) or {}
    intv_browse = der.get("interventionBrowseModule",{}) or {}

    cond_mesh      = [m.get("term", "") for m in cond_browse.get("meshes",    []) or []]
    cond_ancestors = [m.get("term", "") for m in cond_browse.get("ancestors", []) or []]
    intv_mesh      = [m.get("term", "") for m in intv_browse.get("meshes",    []) or []]

    # ── Sex normalisation ─────────────────────────────────────
    sex_raw = (eli.get("sex") or "ALL").upper().strip()
    sex     = sex_raw if sex_raw in ("MALE", "FEMALE", "ALL") else "ALL"

    return {
        # identification
        "nct_id":         id_m.get("nctId", ""),
        "org_study_id":   _dig(id_m, "orgStudyIdInfo", "id"),
        "brief_title":    (id_m.get("briefTitle")    or "").strip(),
        "official_title": (id_m.get("officialTitle") or "").strip(),
        "acronym":        (id_m.get("acronym")       or "").strip(),

        # description
        "brief_summary":        (des.get("briefSummary")        or "").strip(),
        "detailed_description": (des.get("detailedDescription") or "").strip(),

        # conditions
        "conditions":          con.get("conditions", []) or [],
        "keywords":            con.get("keywords",   []) or [],
        "condition_mesh":      cond_mesh,
        "condition_ancestors": cond_ancestors,

        # interventions / arms
        "interventions":     interventions,
        "arms":              arms,
        "intervention_mesh": intv_mesh,

        # eligibility
        "eligibility_criteria_raw": (eli.get("eligibilityCriteria") or "").strip(),
        "min_age_str":              eli.get("minimumAge", ""),
        "max_age_str":              eli.get("maximumAge", ""),
        "sex":                      sex,
        "gender_based":             eli.get("genderBased", False),
        "gender_description":       (eli.get("genderDescription") or ""),
        "healthy_volunteers":       (eli.get("healthyVolunteers") or ""),
        "sampling_method":          (eli.get("samplingMethod")    or ""),
        "study_population":         (eli.get("studyPopulation")   or ""),

        # status / design
        "overall_status":           sta.get("overallStatus", ""),
        "start_date":               _dig(sta, "startDateStruct",           "date"),
        "completion_date":          _dig(sta, "completionDateStruct",      "date"),
        "primary_completion_date":  _dig(sta, "primaryCompletionDateStruct","date"),
        "enrollment":               _dig(sta, "enrollmentInfo", "count", default=None),
        "phase":                    des2.get("phases", []) or [],
        "study_type":               des2.get("studyType", ""),
        "allocation":               _dig(des2, "designInfo", "allocation"),
        "intervention_model":       _dig(des2, "designInfo", "interventionModel"),
        "masking":                  _dig(des2, "designInfo", "maskingInfo", "masking"),
        "primary_purpose":          _dig(des2, "designInfo", "primaryPurpose"),

        # sponsor
        "lead_sponsor":   _dig(spo, "leadSponsor", "name"),
        "sponsor_class":  _dig(spo, "leadSponsor", "class"),
        "collaborators":  [c.get("name", "") for c in (spo.get("collaborators") or [])],

        # outcomes
        "primary_outcomes":   _outcomes(out.get("primaryOutcomes")),
        "secondary_outcomes": _outcomes(out.get("secondaryOutcomes")),

        # locations
        "locations": locations,
    }
