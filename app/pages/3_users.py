"""
Users Page - View what mock users like
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.recommender import get_cbf_data, _cover
from utils.user_data import get_all_user_likes
from components.navbar import render_navbar
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Users - Spotipai",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

# Load CSS based on theme
def load_css():
    """Load CSS file based on current theme"""
    theme = st.session_state.get("theme", "dark")
    css_file = f"styles/{theme}_theme.css"
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), css_file)

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Check authentication
if not is_authenticated():
    st.warning("Please login to access this page")
    st.switch_page("app.py")
    st.stop()

# Render navbar and sidebar
render_navbar()
render_sidebar()

# Page Header
st.markdown('<div class="greeting-hero">👥 Search Users</div>', unsafe_allow_html=True)
st.markdown('<div class="greeting-sub">See what other users are listening to</div>', unsafe_allow_html=True)

with st.spinner("Loading users..."):
    data = get_cbf_data()
    df_spotify = data['df']
    all_likes = get_all_user_likes()
    users = list(all_likes.keys())
    
# Filter UI
selected_user = st.selectbox(
    "Select a User:", 
    options=sorted(users, key=lambda x: (0, int(x.split('_')[1])) if x.startswith('user_') and len(x.split('_')) > 1 and x.split('_')[1].isdigit() else (1, x))
)

if selected_user:
    st.markdown(f"### Songs liked by {selected_user}")
    
    # Get user's ratings directly from json data
    user_songs = all_likes.get(selected_user, [])
    
    if not user_songs:
        st.info(f"{selected_user} hasn't rated any songs yet.")
    else:
        # Sort by rating descending
        user_songs = sorted(user_songs, key=lambda x: x.get('rating', 0), reverse=True)
        
        # We display them using native columns
        cards_per_row = 5
        
        for i in range(0, len(user_songs), cards_per_row):
            row_songs = user_songs[i : i + cards_per_row]
            cols = st.columns(cards_per_row)

            for col, song in zip(cols, row_songs):
                with col:
                    st.markdown('<div class="song-card-wrap">', unsafe_allow_html=True)
                    st.image(song.get("cover_url", "https://picsum.photos/300/300"), use_container_width=True)
                    st.markdown(f'<div class="song-title">{song["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="song-artist">{song["artist"]}</div>', unsafe_allow_html=True)
                    
                    # Display stars
                    rating_val = int(song.get("rating", 5))
                    stars = "⭐" * rating_val
                    st.markdown(f'<div style="color: gold; margin-top: 5px; margin-bottom: 5px;">{stars}</div>', unsafe_allow_html=True)
                    
                    if st.button("ℹ️ Details", key=f"user_sim_detail_{song['id']}_{i}", use_container_width=True):
                        st.session_state["selected_track_id"] = song["id"]
                        st.switch_page("pages/6_song_detail.py")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
