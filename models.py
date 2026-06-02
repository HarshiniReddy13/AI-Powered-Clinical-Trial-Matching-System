"""
models.py — loads the trial_matching_bundle.pkl once and exposes all
             model artefacts + pre-computed sets to the rest of the app.

Import this instead of touching the bundle directly anywhere else.
Uses st.cache_resource so the heavy pickle load only happens once per
Streamlit server process, not on every rerun or page refresh.
"""

import pickle
import streamlit as st


@st.cache_resource(show_spinner="Loading models…")
def load_bundle(path: str = "trial_matching_bundle.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_models(bundle_path: str = "trial_matching_bundle.pkl"):
    """
    Returns a dict with every artefact the backend needs.
    Safe to call multiple times — Streamlit's cache ensures the pkl
    is only unpickled once.
    """
    bundle = load_bundle(bundle_path)

    return {
        "xgb":             bundle["xgb"],
        "dt":              bundle["dt"],
        "scaler":          bundle["scaler"],
        "feature_names":   bundle["feature_names"],
        "trials":          bundle.get("trials", []),
        "selected_labs":   set(bundle.get("selected_labs",   [])),
        "selected_vitals": set(bundle.get("selected_vitals", [])),
    }