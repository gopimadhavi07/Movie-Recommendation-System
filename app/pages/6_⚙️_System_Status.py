"""
app/pages/6_⚙️_System_Status.py
"""
import streamlit as st
import pandas as pd
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.config import DATA_DIR, LOGS_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR

st.set_page_config(page_title="System Status", page_icon="⚙️", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>System Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Workspace paths, environment, and model artefacts</p>", unsafe_allow_html=True)

st.subheader("Directory Paths")
st.table(pd.DataFrame({
    "Location": ["Project Root", "Data Root", "Logs", "Processed Data", "EDA Outputs", "Models"],
    "Configured Path": [str(PROJECT_ROOT), str(DATA_DIR), str(LOGS_DIR),
                        str(PROCESSED_DATA_DIR), str(OUTPUTS_DIR), str(MODELS_DIR)],
}))

st.subheader("Model Artefacts")
artefacts = [
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.joblib",
    "cosine_sim_matrix.joblib",
    "content_movies_index.joblib",
    "user_item_matrix.joblib",
    "svd_preds_matrix.joblib",
    "item_knn_model.joblib",
    "cf_user_map.joblib"
]
rows = []
for a in artefacts:
    p = MODELS_DIR / a
    exists = p.exists()
    size = f"{p.stat().st_size / 1024 / 1024:.2f} MB" if exists else "—"
    rows.append({"Artefact": a, "Exists": "✅" if exists else "❌", "Size": size})
st.table(pd.DataFrame(rows))

st.subheader("Python Environment")
st.write(f"**Executable:** `{sys.executable}`")
st.write(f"**Version:** `{sys.version}`")
st.success("✅ Configurations loaded from src.config")
