"""
app/app.py
==========
Main entry point for CineSuggest Multi-Page Application.
Displays the Home & Overview dashboard.
"""
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.content_based import get_recommendations

st.set_page_config(
    page_title="CineSuggest | Home",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS used across all pages
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.main-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #ff007f 0%, #7f00ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { font-size: 1.2rem; color: #a6adc8; font-weight: 300; margin-bottom: 2rem; }
.glass-card {
    background: rgba(255,255,255,0.03); border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,.2); border: 1px solid rgba(255,255,255,.08);
    padding: 22px; margin-bottom: 22px; transition: all .25s ease;
}
.glass-card:hover { transform: translateY(-4px); border-color: rgba(255,0,127,.4); box-shadow: 0 12px 40px rgba(255,0,127,.15); }
.card-title  { font-size: 1.4rem; font-weight: 600; color: #f5c2e7; margin-bottom: 10px; }
.card-text   { font-size: 1rem; color: #cdd6f4; line-height: 1.6; }
.stat-number { font-size: 2.2rem; font-weight: 800; color: #ff007f; }
.stat-label  { font-size: .85rem; color: #a6adc8; text-transform: uppercase; letter-spacing: 1px; }
.badge { background: linear-gradient(135deg, #89b4fa, #b4befe); color: #11111b; padding: 3px 11px; border-radius: 20px; font-size: .82rem; font-weight: 600; display: inline-block; margin: 2px 4px 4px 0; }
.genre-chip { background: rgba(127,0,255,.18); border: 1px solid rgba(127,0,255,.4); color: #cba6f7; padding: 2px 10px; border-radius: 20px; font-size: .78rem; display: inline-block; margin: 2px 3px 2px 0; }
.sim-score { background: linear-gradient(135deg, #ff007f, #7f00ff); color: #fff; padding: 3px 10px; border-radius: 20px; font-size: .82rem; font-weight: 700; display: inline-block; }
.rec-card { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); border-radius: 12px; padding: 16px; margin-bottom: 14px; transition: all .2s ease; }
.rec-card:hover { border-color: rgba(255,0,127,.45); background: rgba(255,0,127,.05); transform: translateX(4px); }
.rec-title { font-size: 1.1rem; font-weight: 700; color: #cdd6f4; }
.rec-meta { font-size: .88rem; color: #a6adc8; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>Welcome to CineSuggest</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>A State-of-the-Art Movie Recommendation Engine</p>", unsafe_allow_html=True)

if state['data_prepared']:
    stats = state['stats']
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{stats['num_movies']:,}", f"{stats['num_users']:,}", f"{stats['num_ratings']:,}", f"{stats['sparsity_pct']}%"],
        ["Unique Movies", "Unique Users", "Total Ratings", "Matrix Sparsity"]
    ):
        with col:
            st.markdown(f"<div class='glass-card' style='text-align:center;padding:15px'><div class='stat-number'>{val}</div><div class='stat-label'>{lbl}</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="glass-card">
      <div class="card-title">🚀 About the Project</div>
      <div class="card-text">CineSuggest uses advanced ML to recommend movies based on content similarity, collaborative signals, and hybrid engines. Explore the sidebar pages to discover trending movies, manage your profile, and get personalized recommendations.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
      <div class="card-title">🛠️ Features</div>
      <div class="card-text">
        <ul>
          <li><b>Hybrid Engine:</b> SVD Matrix Factorization + TF-IDF Vectorization.</li>
          <li><b>Explainable AI:</b> Understand exactly <i>why</i> a movie is recommended.</li>
          <li><b>Mood Based:</b> Filter recommendations by your current feeling.</li>
          <li><b>User Profiles:</b> Create an account, rate movies, and track history.</li>
          <li><b>Trending:</b> Bayesian averages to prevent low-volume skew.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="glass-card">
      <div class="card-title">💡 Core Capabilities</div>
      <div class="card-text" style="margin-bottom:12px;">
        <span class="badge">Hybrid Recommendations</span>
        <span class="badge">SVD Factorization</span>
        <span class="badge">Cosine Similarity</span>
        <span class="badge">Explainable AI</span>
        <span class="badge">Bayesian Trending</span>
        <span class="badge">SQLite Profiles</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='color:#f5c2e7;font-weight:600;'>🎬 Content Similarity Quick Demo</h4>", unsafe_allow_html=True)
    if state['cb_model_ready']:
        demo_titles = ["Toy Story", "The Dark Knight", "Inception", "Forrest Gump", "Pulp Fiction", "The Matrix"]
        sel = st.selectbox("Pick a movie:", demo_titles)
        cb_movies, cb_sim, cb_idx = state['cb_artifacts']
        try:
            demo_recs = get_recommendations(sel, cb_movies, cb_sim, cb_idx, top_n=3)
            for _, row in demo_recs.iterrows():
                st.success(f"🎯 **{row['title']}** — similarity {row['similarity_score']:.3f}")
        except Exception as e:
            st.warning(str(e))
    else:
        st.info("Build the CB model via the sidebar to enable the demo.")
