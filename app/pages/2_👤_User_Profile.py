"""
app/pages/2_👤_User_Profile.py
"""
import streamlit as st
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app_state import check_pipeline_status
from app.sidebar import render_sidebar
from src.user_profiles import get_all_users, create_user, add_rating, get_user_ratings, get_watch_history

st.set_page_config(page_title="User Profile", page_icon="👤", layout="wide")

state = check_pipeline_status()
render_sidebar(state['data_prepared'], state['eda_plots_ready'], state['cb_model_ready'], state['cf_model_ready'])

st.markdown("<h1 class='main-title'>User Profile</h1>", unsafe_allow_html=True)

if not state['data_prepared']:
    st.warning("Please run the setup pipeline first.")
    st.stop()

movies_df = state['movies_data']

# Initialize session state for user
if 'current_user_id' not in st.session_state:
    st.session_state['current_user_id'] = None
if 'current_username' not in st.session_state:
    st.session_state['current_username'] = None

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🔑 Account")
    
    users_df = get_all_users()
    if not users_df.empty:
        user_list = users_df['username'].tolist()
        selected_user = st.selectbox("Select Existing User:", ["-- Select --"] + user_list)
        if selected_user != "-- Select --":
            if st.button("Login"):
                u_id = users_df[users_df['username'] == selected_user]['user_id'].values[0]
                st.session_state['current_user_id'] = int(u_id)
                st.session_state['current_username'] = selected_user
                st.rerun()
                
    st.markdown("---")
    new_username = st.text_input("Create New User:")
    if st.button("Register"):
        if new_username:
            u_id = create_user(new_username)
            st.session_state['current_user_id'] = u_id
            st.session_state['current_username'] = new_username
            st.success(f"Registered {new_username}!")
            st.rerun()

with col2:
    if st.session_state['current_user_id']:
        st.markdown(f"### 👋 Welcome, {st.session_state['current_username']}")
        st.write(f"User ID: `{st.session_state['current_user_id']}`")
        
        tab1, tab2 = st.tabs(["⭐ Rate a Movie", "📜 History"])
        
        with tab1:
            st.write("Rate movies to improve your personalized recommendations.")
            search_q = st.text_input("Search movie to rate:")
            if search_q:
                matches = movies_df[movies_df['title'].str.contains(search_q, case=False, na=False)].head(10)
                if not matches.empty:
                    movie_sel = st.selectbox("Select movie:", matches['title'].tolist())
                    m_id = matches[matches['title'] == movie_sel]['movieId'].values[0]
                    rating = st.slider("Rating", 0.5, 5.0, 4.0, 0.5)
                    if st.button("Submit Rating"):
                        add_rating(st.session_state['current_user_id'], m_id, rating)
                        st.success(f"Rated {movie_sel} - {rating}⭐")
                else:
                    st.write("No matches.")
                    
        with tab2:
            ratings = get_user_ratings(st.session_state['current_user_id'])
            if not ratings.empty:
                # Merge with movie titles
                merged = ratings.merge(movies_df[['movieId', 'title', 'genres']], left_on='movie_id', right_on='movieId')
                merged = merged.sort_values('timestamp', ascending=False)
                
                for _, row in merged.iterrows():
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px; margin-bottom:10px;">
                        <strong>{row['title']}</strong> — ⭐ {row['rating']} <br>
                        <small>{row['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No ratings yet.")
    else:
        st.info("Please login or register to manage your profile.")
