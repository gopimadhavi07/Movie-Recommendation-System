"""
app/pages/3_🎯_Recommendations.py
"""
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.hybrid import get_hybrid_recommendations
from src.moods import get_all_moods, get_genres_for_mood
from src.explain import generate_explanation
from src.user_profiles import get_user_ratings

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>Personalized Recommendations</h1>", unsafe_allow_html=True)

if not state['cb_model_ready'] or not state['cf_model_ready']:
    st.warning("Please build the Content-Based and Collaborative Filtering models first via the sidebar.")
    st.stop()

if 'current_user_id' not in st.session_state or st.session_state['current_user_id'] is None:
    st.info("👋 Please log in on the **User Profile** page to get personalized hybrid recommendations.")
    st.stop()

user_id = st.session_state['current_user_id']
movies_df = state['movies_data']
ratings_df = state['ratings_data']
cb_movies, cb_sim, cb_idx = state['cb_artifacts']
cf_matrix, cf_item, cf_user, cf_svd, cf_umap, cf_mmap, cf_rumap, cf_rmmap = state['cf_artifacts']

user_ratings = get_user_ratings(user_id)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"### 🎯 For {st.session_state['current_username']}")
    st.write("Powered by Hybrid Collaborative + Content-Based filtering.")
with col2:
    moods = ["Any Mood"] + get_all_moods()
    selected_mood = st.selectbox("How are you feeling?", moods)

if st.button("Generate Picks", use_container_width=True):
    with st.spinner("Crunching the numbers..."):
        mood_genres = None
        if selected_mood != "Any Mood":
            mood_genres = get_genres_for_mood(selected_mood)
            
        recs = get_hybrid_recommendations(
            user_id=user_id,
            user_ratings=user_ratings,
            movies_df=movies_df,
            ratings_df=ratings_df,
            cb_sim_matrix=cb_sim,
            cb_title_idx=cb_idx,
            svd_preds=cf_svd,
            cf_user_map=cf_umap,
            cf_rev_movie_map=cf_rmmap,
            top_n=10,
            mood_genres=mood_genres
        )
        
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            explanation = generate_explanation(row, user_ratings, movies_df)
            genres_html = "".join(f'<span class="genre-chip">{g}</span>' for g in str(row["genres"]).split("|")[:5])
            
            score_col = 'hybrid_score' if 'hybrid_score' in row else 'trending_score'
            
            st.markdown(f"""
            <div class="rec-card">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="rec-title">#{rank} &nbsp; {row['title']}</span>
                <span class="sim-score">Score: {row[score_col]:.2f}</span>
              </div>
              <div class="rec-meta" style="margin-top:6px;">{genres_html}</div>
              <div class="rec-meta" style="margin-top:8px; color:#cba6f7;">
                <strong>💡 Why?</strong> {explanation}
              </div>
            </div>
            """, unsafe_allow_html=True)
