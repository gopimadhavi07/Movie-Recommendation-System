"""
app/app_state.py
================
Centralized state and cached data loaders for Streamlit pages.
"""
import ast
import json
import pandas as pd
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.content_based import build_content_based_model
from src.collaborative import build_cf_model

PROC_MOVIES = PROCESSED_DATA_DIR / "processed_movies.csv"
PROC_RATINGS = PROCESSED_DATA_DIR / "processed_ratings.csv"
STATS_JSON = PROCESSED_DATA_DIR / "summary_statistics.json"

@st.cache_data
def load_processed_data():
    if PROC_MOVIES.exists() and PROC_RATINGS.exists():
        movies_df = pd.read_csv(PROC_MOVIES)
        movies_df["genres_list"] = movies_df["genres_list"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
        ratings_df = pd.read_csv(PROC_RATINGS)
        return movies_df, ratings_df
    return None, None

def load_stats():
    if STATS_JSON.exists():
        with open(STATS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_cb_model():
    try:
        return build_content_based_model(force_rebuild=False)
    except Exception:
        return None, None, None

@st.cache_resource
def load_cf_artifacts_cached():
    try:
        return build_cf_model(force_rebuild=False)
    except Exception:
        return None

def check_pipeline_status():
    movies_data, ratings_data = load_processed_data()
    stats = load_stats()
    data_prepared = movies_data is not None and stats is not None

    cb_movies, cb_sim, cb_idx = load_cb_model()
    cb_model_ready = cb_sim is not None

    cf_artifacts = load_cf_artifacts_cached()
    cf_model_ready = cf_artifacts is not None

    eda_plots_ready = all(
        (OUTPUTS_DIR / f).exists()
        for f in ["rating_distribution.png", "user_activity.png", "genre_distribution.png", "movies_by_year.png", "rating_count_vs_mean.png"]
    )
    
    return {
        "data_prepared": data_prepared,
        "eda_plots_ready": eda_plots_ready,
        "cb_model_ready": cb_model_ready,
        "cf_model_ready": cf_model_ready,
        "movies_data": movies_data,
        "ratings_data": ratings_data,
        "stats": stats,
        "cb_artifacts": (cb_movies, cb_sim, cb_idx),
        "cf_artifacts": cf_artifacts
    }
