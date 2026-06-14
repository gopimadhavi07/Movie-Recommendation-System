"""
app/pages/1_🔍_Search_&_Trending.py
"""
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.discovery import search_movies, calculate_trending_score

st.set_page_config(page_title="Search & Trending", page_icon="🔍", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>Search & Trending</h1>", unsafe_allow_html=True)

if not state['data_prepared']:
    st.warning("Please run the setup pipeline first.")
    st.stop()

movies_df = state['movies_data']

tab1, tab2 = st.tabs(["🔥 Trending Movies", "🔍 Advanced Search"])

with tab1:
    st.markdown("### 🔥 Trending Movies")
    st.write("Ranked by Bayesian Average to balance high ratings with high vote volume.")
    
    trending = calculate_trending_score(movies_df)
    
    for rank, (_, row) in enumerate(trending.head(20).iterrows(), 1):
        genres_html = "".join(f'<span class="genre-chip">{g}</span>' for g in str(row["genres"]).split("|")[:5])
        st.markdown(f"""
        <div class="rec-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="rec-title">#{rank} &nbsp; {row['title']}</span>
            <span class="sim-score">Trending Score: {row['trending_score']:.2f}</span>
          </div>
          <div class="rec-meta" style="margin-top:6px;">{genres_html}</div>
          <div class="rec-meta" style="margin-top:6px;">⭐ {row['rating_mean']:.2f} &nbsp;·&nbsp; 🗳️ {int(row['rating_count']):,} votes</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🔍 Search Movies")
    
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Title Search:")
    with col2:
        all_genres = set()
        for g_list in movies_df['genres'].str.split('|'):
            if isinstance(g_list, list):
                all_genres.update(g_list)
        if "(no genres listed)" in all_genres: all_genres.remove("(no genres listed)")
        genres = st.multiselect("Filter by Genres:", sorted(list(all_genres)))
        
    if st.button("Search", use_container_width=True):
        results = search_movies(movies_df, title_query=query, genres=genres)
        if results.empty:
            st.info("No movies found.")
        else:
            st.success(f"Found {len(results)} movies.")
            for _, row in results.head(50).iterrows():
                genres_html = "".join(f'<span class="genre-chip">{g}</span>' for g in str(row["genres"]).split("|")[:5])
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:10px; padding: 12px;">
                  <span class="rec-title">{row['title']}</span> <br>
                  <div class="rec-meta" style="margin-top:4px;">{genres_html}</div>
                  <div class="rec-meta">⭐ {row['rating_mean']:.2f} ({int(row['rating_count'])} votes)</div>
                </div>
                """, unsafe_allow_html=True)
