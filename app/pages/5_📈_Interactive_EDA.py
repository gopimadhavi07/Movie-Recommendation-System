"""
app/pages/5_📈_Interactive_EDA.py
"""
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.config import OUTPUTS_DIR

st.set_page_config(page_title="Interactive EDA", page_icon="📈", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

if not state['eda_plots_ready']:
    st.warning("EDA plots not yet generated. Please run the setup pipeline.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["⭐ Rating Distributions", "👥 User Patterns & Time Trends", "🎭 Genre & Correlation"])

with tab1:
    col_a, col_b = st.columns([3, 2])
    with col_a: st.image(str(OUTPUTS_DIR / "rating_distribution.png"), use_container_width=True)
    with col_b:
        st.markdown("""
        <div class="glass-card" style="margin-top:20px">
          <div class="card-title">⭐ Rating Bias</div>
          <div class="card-text">
            <ul>
              <li><b>4.0★</b> is the mode.</li>
              <li>48% of ratings are ≥ 4.0, confirming positive selection bias.</li>
            </ul>
          </div>
        </div>""", unsafe_allow_html=True)

with tab2:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.image(str(OUTPUTS_DIR / "user_activity.png"), use_container_width=True)
        st.image(str(OUTPUTS_DIR / "movies_by_year.png"), use_container_width=True)
    with col_b:
        st.markdown("""
        <div class="glass-card" style="margin-top:20px">
          <div class="card-title">👥 User Activity</div>
          <div class="card-text">Median user: <b>70 ratings</b>. Long-tail distribution demands regularisation in collaborative models.</div>
        </div>
        <div class="glass-card">
          <div class="card-title">📈 Release Trends</div>
          <div class="card-text">Explosive growth from the 1990s, peaking ~2003.</div>
        </div>""", unsafe_allow_html=True)

with tab3:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.image(str(OUTPUTS_DIR / "genre_distribution.png"), use_container_width=True)
        st.image(str(OUTPUTS_DIR / "rating_count_vs_mean.png"), use_container_width=True)
    with col_b:
        st.markdown("""
        <div class="glass-card" style="margin-top:20px">
          <div class="card-title">🎭 Genre Dominance</div>
          <div class="card-text">Drama 4,361 · Comedy 3,756 · Thriller 1,894.</div>
        </div>
        <div class="glass-card">
          <div class="card-title">🔷 Quality vs Popularity</div>
          <div class="card-text">Correlation coefficient <b>0.127</b> — popular movies tend to be slightly better rated.</div>
        </div>""", unsafe_allow_html=True)
