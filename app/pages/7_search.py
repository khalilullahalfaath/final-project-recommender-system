"""
Search Page - View search results for songs.
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from utils.recommender import get_cbf_data, _cover, recommend_from_seeds
from components.navbar import render_navbar
from components.sidebar import render_sidebar
from components.recommendation_row import render_recommendation_row

# Page configuration
st.set_page_config(
    page_title="Search Results - Spotipai",
    page_icon="🔍",
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

query = st.session_state.get("search_query", "")

st.markdown(f'<div class="greeting-hero">🔍 Search Results</div>', unsafe_allow_html=True)

if not query:
    st.info("Please enter a search term in the search bar above.")
    st.stop()
    
st.markdown(f'<div class="greeting-sub">Showing results for: "{query}"</div>', unsafe_allow_html=True)

with st.spinner("Searching for songs..."):
    data = get_cbf_data()
    df_spotify = data['df']
    
    labels = df_spotify['track_name'].astype(str) + " - " + df_spotify['artists'].astype(str)
    
    is_exact_match = False
    if query in labels.values:
        top_match = df_spotify[labels == query].iloc[0]
        is_exact_match = True
        results_df = pd.DataFrame()
    else:
        # Match by words against a combined corpus of track name and artists
        search_corpus = (df_spotify['track_name'].fillna('') + " " + df_spotify['artists'].fillna('')).str.lower()
        q_lower = query.lower()
        
        exact_combined_mask = search_corpus == q_lower
        exact_track_mask = df_spotify['track_name'].fillna('').str.lower() == q_lower
        exact_artist_mask = df_spotify['artists'].fillna('').str.lower() == q_lower
        
        words = q_lower.split()
        partial_mask = pd.Series([True] * len(df_spotify), index=df_spotify.index)
        for word in words:
            partial_mask = partial_mask & search_corpus.str.contains(word, regex=False)
            
        final_mask = exact_combined_mask | exact_track_mask | exact_artist_mask | partial_mask
        results_df = df_spotify[final_mask].copy()
        
        results_df['match_score'] = 0
        results_df.loc[partial_mask, 'match_score'] = 1
        results_df.loc[exact_artist_mask, 'match_score'] = 2
        results_df.loc[exact_track_mask, 'match_score'] = 3
        results_df.loc[exact_combined_mask, 'match_score'] = 4
        
        # Limit to top 50 sorted by match score then popularity
        results_df = results_df.sort_values(by=["match_score", "popularity"], ascending=[False, False]).head(50)
        
    if not is_exact_match and results_df.empty:
        st.warning(f"No songs found matching '{query}'.")
    else:
        if not is_exact_match:
            top_match = results_df.iloc[0]
            
        label = f"{top_match['track_name']} - {top_match['artists']}"
        
        st.markdown(f"### 🎯 Match for '{query}'")
        top_res = [{
            "id": top_match['track_id'],
            "title": top_match['track_name'],
            "artist": top_match['artists'],
            "genre": top_match['track_genre'],
            "cover_url": _cover(top_match['track_id'])
        }]
        render_recommendation_row("", top_res, key_prefix="top_match")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🔮 Recommended based on '{top_match['track_name']}'")
        st.caption("Content-Based recommendations automatically generated from your search")
        
        similar_songs, _ = recommend_from_seeds([label], top_n=10)
        if similar_songs:
            render_recommendation_row("", similar_songs, key_prefix="recommendations")
        else:
            st.info("No recommendations found.")
            
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("### 📄 Other Search Results")
        
        other_results = []
        for _, row in results_df.iloc[1:].iterrows():
            other_results.append({
                "id": row['track_id'],
                "title": row['track_name'],
                "artist": row['artists'],
                "genre": row['track_genre'],
                "cover_url": _cover(row['track_id'])
            })
            
        if other_results:
            render_recommendation_row("", other_results, key_prefix="other_results")
