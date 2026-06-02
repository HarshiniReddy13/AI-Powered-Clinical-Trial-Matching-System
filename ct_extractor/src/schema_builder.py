# src/schema_builder.py
# ─────────────────────────────────────────────────────────────────────────────
# Assembles the final output JSON record.
# Every field from the target schema is always present; missing values are null.
# ─────────────────────────────────────────────────────────────────────────────


def build_record(
    raw: dict,
    eligibility: dict,
    clinical_req: dict,
    meta: dict,
) -> dict:
    cr = clinical_req
    return {
        # ── Identifiers ───────────────────────────────────────
        "trial_id":   raw["nct_id"],
        "title":      raw["brief_title"] or raw["official_title"],
        "summary":    raw["brief_summary"] or raw["detailed_description"],
        "acronym":    raw.get("acronym", ""),

        # ── Trial metadata ────────────────────────────────────
        "phase":                   raw.get("phase", []),
        "study_type":              raw.get("study_type", ""),
        "overall_status":          raw.get("overall_status", ""),
        "start_date":              raw.get("start_date", ""),
        "completion_date":         raw.get("completion_date", ""),
        "primary_completion_date": raw.get("primary_completion_date", ""),
        "enrollment":              raw.get("enrollment"),
        "allocation":              raw.get("allocation", ""),
        "intervention_model":      raw.get("intervention_model", ""),
        "masking":                 raw.get("masking", ""),
        "primary_purpose":         raw.get("primary_purpose", ""),

        # ── Sponsor ───────────────────────────────────────────
        "lead_sponsor":  raw.get("lead_sponsor", ""),
        "sponsor_class": raw.get("sponsor_class", ""),
        "collaborators": raw.get("collaborators", []),

        # ── Conditions & interventions ────────────────────────
        "conditions":          raw.get("conditions", []),
        "condition_mesh":      raw.get("condition_mesh", []),
        "condition_ancestors": raw.get("condition_ancestors", []),
        "keywords":            raw.get("keywords", []),
        "interventions":       raw.get("interventions", []),
        "intervention_mesh":   raw.get("intervention_mesh", []),
        "arms":                raw.get("arms", []),

        # ── Outcomes ──────────────────────────────────────────
        "primary_outcomes":   raw.get("primary_outcomes", []),
        "secondary_outcomes": raw.get("secondary_outcomes", []),

        # ── Locations ─────────────────────────────────────────
        "locations": raw.get("locations", []),

        # ── Eligibility ───────────────────────────────────────
        "eligibility": {
            "raw_text":  eligibility.get("raw_text", ""),
            "inclusion": eligibility.get("inclusion", []),
            "exclusion": eligibility.get("exclusion", []),
        },

        # ── Clinical requirements ─────────────────────────────
        "clinical_requirements": {
            "demographics": {
                "age": {
                    "min": cr["demographics"]["age"]["min"],
                    "max": cr["demographics"]["age"]["max"],
                },
                "sex":                cr["demographics"]["sex"],
                "gender_description": cr["demographics"].get("gender_description", ""),
                "healthy_volunteers": cr["demographics"].get("healthy_volunteers", ""),
            },
            "lab_tests": [
                {
                    "test_name":       lt["test_name"],
                    "type":            lt["type"],
                    "operator":        lt["operator"],
                    "value":           lt["value"],
                    "min":             lt["min"],
                    "max":             lt["max"],
                    "unit":            lt["unit"],
                    "allowed_values":  lt.get("allowed_values", []),
                    "reference":       lt["reference"],
                    "source_sentence": lt.get("source_sentence", ""),
                }
                for lt in cr.get("lab_tests", [])
            ],
            "vitals": [
                {
                    "name":            v["name"],
                    "operator":        v["operator"],
                    "value":           v["value"],
                    "min":             v["min"],
                    "max":             v["max"],
                    "unit":            v["unit"],
                    "source_sentence": v.get("source_sentence", ""),
                }
                for v in cr.get("vitals", [])
            ],
            "scores": [
                {
                    "name":            s["name"],
                    "operator":        s["operator"],
                    "value":           s["value"],
                    "min":             s["min"],
                    "max":             s["max"],
                    "source_sentence": s.get("source_sentence", ""),
                }
                for s in cr.get("scores", [])
            ],
            "disease_requirements": [
                {
                    "condition":       dr["condition"],
                    "stage":           dr["stage"],
                    "metastasis":      dr["metastasis"],
                    "source_sentence": dr.get("source_sentence", ""),
                }
                for dr in cr.get("disease_requirements", [])
            ],
            "treatment_history": [
                {
                    "type":            th["type"],
                    "allowed":         th["allowed"],
                    "specific_agents": th.get("specific_agents", []),
                    "time_constraint": th["time_constraint"],
                    "source_sentence": th.get("source_sentence", ""),
                }
                for th in cr.get("treatment_history", [])
            ],
            "other_constraints": [
                {
                    "type":        oc["type"],
                    "description": oc["description"],
                }
                for oc in cr.get("other_constraints", [])
            ],
        },

        # ── Extraction metadata ───────────────────────────────
        "_extraction_meta": {
            "bert_models":       meta.get("bert_models", []),
            "scispacy_model":    meta.get("scispacy_model", ""),
            "bert_entity_count": meta.get("bert_entity_count", 0),
            "sci_entity_count":  meta.get("sci_entity_count", 0),
            "inclusion_count":   len(eligibility.get("inclusion", [])),
            "exclusion_count":   len(eligibility.get("exclusion", [])),
        },
    }
