"""
Liked Songs Page
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.user_data import get_liked_songs, remove_liked_song, update_song_rating
from components.navbar import render_navbar
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Liked Songs - Spotipai",
    page_icon="❤️",
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
username = st.session_state.get("username", "User")
st.markdown('<div class="greeting-hero">❤️ Liked Songs</div>', unsafe_allow_html=True)
st.markdown(f'<div class="greeting-sub">Your personal music library, {username}</div>', unsafe_allow_html=True)

# Fetch liked songs
liked_songs = get_liked_songs(username)

if not liked_songs:
    st.info("You haven't liked any songs yet. Go to the Home page and discover some music!")
    st.button("Go to Home", on_click=lambda: st.switch_page("pages/1_home.py"))
else:
    # We display them using native columns, similarly to recommendation_row but wrapped
    cards_per_row = 5
    for i in range(0, len(liked_songs), cards_per_row):
        row_songs = liked_songs[i : i + cards_per_row]
        cols = st.columns(cards_per_row)

        for col, song in zip(cols, row_songs):
            with col:
                st.markdown('<div class="song-card-wrap">', unsafe_allow_html=True)
                st.image(song.get("cover_url", "https://picsum.photos/300/300"), use_container_width=True)
                st.markdown(f'<div class="song-title">{song["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="song-artist">{song["artist"]}</div>', unsafe_allow_html=True)
                
                # Feedback stars
                fb_key = f"rating_{song['id']}"
                if fb_key not in st.session_state:
                    st.session_state[fb_key] = song.get("rating", 5) - 1
                
                def update_rating_callback(sid=song["id"], key=fb_key):
                    new_val = st.session_state[key]
                    if new_val is not None:
                        update_song_rating(username, sid, new_val + 1)
                        
                st.feedback("stars", key=fb_key, on_change=update_rating_callback)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Unlike button
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("💔 Remove", key=f"unlike_btn_{song['id']}", use_container_width=True):
                        remove_liked_song(username, song["id"])
                        st.rerun()
                with btn_col2:
                    if st.button("ℹ️ Details", key=f"detail_btn_{song['id']}", use_container_width=True):
                        st.session_state["selected_track_id"] = song["id"]
                        st.switch_page("pages/6_song_detail.py")
