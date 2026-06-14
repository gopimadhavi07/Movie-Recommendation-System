"""
app/pages/4_📊_Dataset_Explorer.py
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

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>Dataset Explorer</h1>", unsafe_allow_html=True)

if not state['data_prepared']:
    st.info("Dataset not prepared. Use the sidebar to run the setup pipeline.")
    st.stop()

movies_data = state['movies_data']
stats = state['stats']

search = st.text_input("🔍 Search by title:", "")
df_show = movies_data.copy()
if search:
    df_show = df_show[df_show["title"].str.contains(search, case=False, na=False)]

st.dataframe(
    df_show[["movieId","title","genres","release_year","rating_mean","rating_count"]].head(100),
    use_container_width=True,
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("🔥 Top 5 Most Rated")
    st.table(pd.DataFrame(stats["most_rated_movies"]))
with c2:
    st.subheader("🏆 Top 5 Highest Rated (≥50 ratings)")
    st.table(pd.DataFrame(stats["highest_rated_movies"]))
