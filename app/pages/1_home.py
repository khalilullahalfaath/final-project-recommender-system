"""
Home page - Music Recommendations
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.mock_data import (
    get_collaborative_recommendations,
)
from utils.recommender import (
    recommend_from_criteria,
    recommend_from_seeds,
    recommend_collaborative,
    get_song_list,
)
from utils.user_data import get_liked_songs
from components.navbar import render_navbar
from components.sidebar import render_sidebar
from components.recommendation_row import render_recommendation_row

# Page configuration
st.set_page_config(
    page_title="Home - Spotipai",
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

# Render navbar
render_navbar()

# Render sidebar
render_sidebar()

# Greeting hero
username = st.session_state.get("username", "User")
hour = datetime.now().hour

if hour < 12:
    greeting = "Good morning"
elif hour < 18:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

st.markdown(
    f'<div class="greeting-hero">{greeting}, {username}</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="greeting-sub">Discover music tailored to your taste</div>',
    unsafe_allow_html=True,
)

# Tabs for different recommendation types
tab2, tab1 = st.tabs(["Content-Based", "Collaborative Filtering"])

with tab1:
    st.markdown(
        '<div class="section-subtitle">Based on what users similar to you love</div>',
        unsafe_allow_html=True,
    )
    
    # Check if user has liked any songs
    user_liked_songs = get_liked_songs(username)
    if not user_liked_songs:
        st.warning("Please like some songs using the Content-Based tab first before using Collaborative Filtering!")
        st.info("Collaborative Filtering needs to know what you like to find users with similar taste.")
    else:
        if "cf_results" not in st.session_state:
            with st.spinner("Finding similar users based on your Liked Songs..."):
                results, not_found, similar_users = recommend_collaborative(username, top_n=5)
                st.session_state["cf_results"] = results
                st.session_state["cf_not_found"] = not_found
                st.session_state["cf_similar_users"] = similar_users

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Found {len(user_liked_songs)} liked songs in your library.**")
        with col2:
            if st.button("🔄 Refresh Recommendations"):
                with st.spinner("Recalculating..."):
                    results, not_found, similar_users = recommend_collaborative(username, top_n=5)
                    st.session_state["cf_results"] = results
                    st.session_state["cf_not_found"] = not_found
                    st.session_state["cf_similar_users"] = similar_users
                st.rerun()

        if st.session_state.get("cf_results") is not None:
            not_found = st.session_state.get("cf_not_found", [])
            for label in not_found:
                st.warning(f"Song '{label}' was not found in the database and was ignored.")
                
            similar_users = st.session_state.get("cf_similar_users", [])
            if similar_users:
                users_text = ", ".join([f"{u['user_id']} ({u['similarity']*100:.0f}% match)" for u in similar_users])
                st.success(f"We found these users with similar taste: **{users_text}**")
                
            if st.session_state["cf_results"]:
                render_recommendation_row("Recommended by Users Like You", st.session_state["cf_results"])
            else:
                if not not_found: # Only show warning if we didn't already show not_found warnings
                    st.warning("Could not find any matches or recommendations.")

with tab2:
    st.markdown(
        '<div class="section-subtitle">Discover music based on attributes or your favorite songs</div>',
        unsafe_allow_html=True,
    )

    cbf_mode = st.radio("Choose recommendation mode:", ["Find by Criteria", "Find Similar to Favorite Songs"], horizontal=True)

    if "form_key_suffix" not in st.session_state:
        st.session_state["form_key_suffix"] = 0

    if cbf_mode == "Find by Criteria":
        with st.form(f"criteria_form_{st.session_state['form_key_suffix']}"):
            col1, col2 = st.columns(2)
            with col1:
                artist = st.text_input("Artist (Optional)", placeholder="e.g. Coldplay", key="artist_crit")
                genre = st.text_input("Genre (Optional)", placeholder="e.g. pop, rock", key="genre_crit")
            with col2:
                use_energy = st.checkbox("Use Target Energy Filter", value=True)
                target_energy = st.slider("Target Energy (0: Acoustic/Calm, 1: Intense)", 0.0, 1.0, 0.5, key="energy_crit")
                
                use_valence = st.checkbox("Use Target Valence Filter", value=True)
                target_valence = st.slider("Target Valence (0: Sad/Depressing, 1: Happy/Cheerful)", 0.0, 1.0, 0.5, key="valence_crit")
            
            top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, key="top_n_crit")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit_criteria = st.form_submit_button("Get Recommendations", type="primary")
            with col_btn2:
                clear_criteria = st.form_submit_button("🔄 Clear Fields")

        if clear_criteria:
            st.session_state["form_key_suffix"] += 1
            st.session_state["criteria_results"] = None
            st.rerun()

        if submit_criteria:
            st.session_state["criteria_results"] = None
            if not artist and not genre and not use_energy and not use_valence:
                st.warning("Please specify at least 1 criteria (Artist, Genre, Energy, or Valence).")
            else:
                with st.spinner("Finding matches..."):
                    results = recommend_from_criteria(
                        artist=artist, 
                        genre=genre, 
                        target_energy=target_energy, 
                        target_valence=target_valence, 
                        use_energy=use_energy,
                        use_valence=use_valence,
                        top_n=top_n
                    )
                    st.session_state["criteria_results"] = results
                    
                    # Create a friendly title
                    used_crit = []
                    if artist: used_crit.append(f"Artist '{artist}'")
                    if genre: used_crit.append(f"Genre '{genre}'")
                    if use_energy: used_crit.append("Energy")
                    if use_valence: used_crit.append("Valence")
                    st.session_state["criteria_artist_genre"] = " + ".join(used_crit)

        if st.session_state.get("criteria_results") is not None:
            if st.session_state["criteria_results"]:
                title = f"Recommendations for {st.session_state.get('criteria_artist_genre', '')}"
                render_recommendation_row(title, st.session_state["criteria_results"])
            else:
                st.warning("No recommendations found.")

    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Search and select your favorite songs:")
        with col2:
            if st.button("❤️ Use Liked Songs", key="populate_cbf"):
                liked = get_liked_songs(username)
                if liked:
                    labels = [f"{s['title']} - {s['artist']}" for s in liked]
                    valid_labels = [l for l in get_song_list() if l in labels]
                    st.session_state[f"cbf_multi_{st.session_state['form_key_suffix']}"] = valid_labels[:5]
                    st.rerun()
                else:
                    st.toast("No liked songs found!")
        
        if "form_key_suffix" not in st.session_state:
            st.session_state["form_key_suffix"] = 0

        with st.form(f"seeds_form_{st.session_state['form_key_suffix']}"):
            all_songs = get_song_list()
            default_val = st.session_state.get(f"cbf_multi_{st.session_state['form_key_suffix']}", [])
            
            selected_songs = st.multiselect(
                "Select up to 5 songs (Type to search)", 
                options=all_songs, 
                default=default_val,
                max_selections=5, 
                placeholder="e.g. Baby - Justin Bieber"
            )
            top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, key="top_n_seeds")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit_seeds = st.form_submit_button("Get Recommendations", type="primary")
            with col_btn2:
                clear_seeds = st.form_submit_button("🔄 Clear Fields")

        if clear_seeds:
            if "top_n_seeds" in st.session_state:
                del st.session_state["top_n_seeds"]
            st.session_state["form_key_suffix"] += 1
            st.session_state["seeds_results"] = None
            st.rerun()

        if submit_seeds:
            st.session_state["seeds_results"] = None
            if selected_songs:
                with st.spinner("Finding similar songs..."):
                    results, not_found = recommend_from_seeds(selected_songs, top_n)
                    st.session_state["seeds_results"] = results
                    st.session_state["seeds_not_found"] = not_found
            else:
                st.warning("Please search and select at least one song.")

        if st.session_state.get("seeds_results") is not None:
            not_found = st.session_state.get("seeds_not_found", [])
            for label in not_found:
                st.warning(f"Song '{label}' was not found in the database and was ignored.")
                
            if st.session_state["seeds_results"]:
                render_recommendation_row("Because you liked those songs", st.session_state["seeds_results"])
            else:
                if not not_found:
                    st.warning("Could not find any matches for the valid songs provided.")

