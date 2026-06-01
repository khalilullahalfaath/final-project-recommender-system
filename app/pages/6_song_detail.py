"""
Song Detail Page - View metadata and similar songs for a specific track.
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.user_data import is_song_liked, add_liked_song, remove_liked_song, get_liked_songs, update_song_rating
from utils.recommender import get_cbf_data, _cover, recommend_from_seeds
from components.navbar import render_navbar
from components.sidebar import render_sidebar
from components.recommendation_row import render_recommendation_row

# Page configuration
st.set_page_config(
    page_title="Song Details - Spotipai",
    page_icon="🎵",
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

# Check if a track is selected
if "selected_track_id" not in st.session_state or not st.session_state["selected_track_id"]:
    st.warning("No song selected. Please select a song from the Home or Liked Songs page.")
    st.stop()
    
track_id = st.session_state["selected_track_id"]
username = st.session_state.get("username")

with st.spinner("Loading song details..."):
    data = get_cbf_data()
    df_spotify = data['df']
    
    match = df_spotify[df_spotify['track_id'] == track_id]
    
    if match.empty:
        st.error("Song not found in the database.")
        st.stop()
        
    song = match.iloc[0]
    
    col_cover, col_info = st.columns([1, 2])
    
    with col_cover:
        cover_url = _cover(track_id)
        st.image(cover_url, use_container_width=True)
        
    with col_info:
        st.markdown(f'<h1 style="margin-bottom: 0px;">{song["track_name"]}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h3 style="color: gray; margin-top: 0px;">{song["artists"]}</h3>', unsafe_allow_html=True)
        
        st.markdown(f"**Album:** {song['album_name']}")
        st.markdown(f"**Genre:** {str(song['track_genre']).title()}")
        st.markdown(f"**Popularity:** {song['popularity']}/100")
        
        explicit_badge = "🔞 Explicit" if song['explicit'] else "✅ Clean"
        st.markdown(f"**Content:** {explicit_badge}")
        
        # Format duration
        duration_sec = int(song['duration_ms'] / 1000)
        st.markdown(f"**Duration:** {duration_sec // 60}:{duration_sec % 60:02d}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        is_liked = is_song_liked(username, track_id)
        
        if is_liked:
            # Find the current rating
            user_likes = get_liked_songs(username)
            current_rating = 5
            for ls in user_likes:
                if ls["id"] == track_id:
                    current_rating = ls.get("rating", 5)
                    break
                    
            fb_key = f"detail_rating_{track_id}"
            if fb_key not in st.session_state:
                st.session_state[fb_key] = current_rating - 1
                
            def detail_update_rating():
                new_val = st.session_state[fb_key]
                if new_val is not None:
                    update_song_rating(username, track_id, new_val + 1)
                    
            st.markdown("**Your Rating:**")
            st.feedback("stars", key=fb_key, on_change=detail_update_rating)
            st.markdown("<br>", unsafe_allow_html=True)
            
        button_label = "❤️ Remove from Liked" if is_liked else "🤍 Add to Liked"
        
        if st.button(button_label, use_container_width=False, type="primary" if not is_liked else "secondary"):
            if is_liked:
                remove_liked_song(username, track_id)
            else:
                clean_song = {
                    "id": track_id,
                    "title": str(song['track_name']),
                    "artist": str(song['artists']),
                    "genre": str(song['track_genre']),
                    "cover_url": cover_url,
                    "rating": 5
                }
                add_liked_song(username, clean_song)
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.markdown("### 📊 Audio Features")
    features = ['danceability', 'energy', 'valence', 'acousticness', 'liveness', 'speechiness']
    
    for feat in features:
        val = song[feat]
        st.markdown(f"**{feat.title()}**: {val:.2f}")
        st.progress(float(val))

with col_feat2:
    st.markdown("### 🔍 Similar Songs")
    st.caption("Based on Content-Based Filtering using audio features and genres.")
    
    label = f"{song['track_name']} - {song['artists']}"
    similar_songs, not_found = recommend_from_seeds([label], top_n=5)
    
    if similar_songs:
        for sim_song in similar_songs:
            col_img, col_text, col_btn = st.columns([1, 4, 2])
            with col_img:
                st.image(sim_song["cover_url"], width=50)
            with col_text:
                st.markdown(f"**{sim_song['title']}**")
                st.caption(sim_song['artist'])
            with col_btn:
                if st.button("Details", key=f"sim_detail_{sim_song['id']}"):
                    st.session_state["selected_track_id"] = sim_song["id"]
                    st.rerun()
            st.markdown("<hr style='margin: 5px 0px;'>", unsafe_allow_html=True)
    else:
        st.info("No similar songs found.")
