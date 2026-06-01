"""
Profile Page - View user statistics
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.user_data import get_liked_songs
from utils.recommender import get_cbf_data
from components.navbar import render_navbar
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Profile - Spotipai",
    page_icon="👤",
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

username = st.session_state.get("username")

st.markdown(f'<div class="greeting-hero">👤 {username}\'s Profile</div>', unsafe_allow_html=True)

with st.spinner("Loading your profile..."):
    liked_songs = get_liked_songs(username)
    
    if not liked_songs:
        st.info("You haven't liked any songs yet! Go to the Content-Based tab to start liking songs.")
        st.stop()
        
    data = get_cbf_data()
    df_spotify = data['df']
    
    # Get all liked track ids
    liked_ids = [s['id'] for s in liked_songs]
    user_df = df_spotify[df_spotify['track_id'].isin(liked_ids)]
    
    if user_df.empty:
        st.warning("Could not find metadata for your liked songs.")
        st.stop()

    # Basic stats
    total_songs = len(liked_songs)
    st.markdown(f'<div class="greeting-sub">You have {total_songs} liked songs.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Artists")
        artists = user_df['artists'].value_counts().head(5)
        for artist, count in artists.items():
            st.markdown(f"- **{artist}** ({count} songs)")
            
    with col2:
        st.markdown("### 🎸 Top Genres")
        genres = user_df['track_genre'].value_counts().head(5)
        for genre, count in genres.items():
            st.markdown(f"- **{genre.title()}** ({count} songs)")
            
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("### 🎧 Your Music Vibe")
    
    # Audio Features Averages
    avg_danceability = user_df['danceability'].mean()
    avg_energy = user_df['energy'].mean()
    avg_valence = user_df['valence'].mean()
    avg_acousticness = user_df['acousticness'].mean()
    
    vibe_cols = st.columns(4)
    with vibe_cols[0]:
        st.metric(label="💃 Danceability", value=f"{avg_danceability*100:.0f}%")
        st.caption("How suitable your tracks are for dancing.")
    with vibe_cols[1]:
        st.metric(label="⚡ Energy", value=f"{avg_energy*100:.0f}%")
        st.caption("Intensity and activity measure.")
    with vibe_cols[2]:
        st.metric(label="😊 Positivity (Valence)", value=f"{avg_valence*100:.0f}%")
        st.caption("Musical positiveness conveyed.")
    with vibe_cols[3]:
        st.metric(label="🎸 Acousticness", value=f"{avg_acousticness*100:.0f}%")
        st.caption("Confidence that your tracks are acoustic.")
