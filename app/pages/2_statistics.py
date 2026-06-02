"""
System Statistics Page
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.session import init_session_state, is_authenticated
from components.navbar import render_navbar
from components.sidebar import render_sidebar
from utils.recommender import get_cbf_data, get_w2v_embeddings, get_gnn_embeddings

# Page configuration
st.set_page_config(
    page_title="System Statistics - Spotipai",
    page_icon="📊",
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

# Render navbar & sidebar
render_navbar()
render_sidebar()

# Main page content
st.markdown('<div class="greeting-text">📊 System Statistics</div>', unsafe_allow_html=True)
st.markdown('<div class="greeting-sub">Engine specifications for demonstration purposes</div>', unsafe_allow_html=True)
st.write("---")

col1, col2, col3 = st.columns(3)

try:
    data = get_cbf_data()
    df = data['df']
    cleaned_songs_count = len(df)
except Exception:
    cleaned_songs_count = 0

# 1. TF-IDF
with col1:
    st.subheader("1. TF-IDF Engine")
    try:
        st.metric("Active Songs", f"{cleaned_songs_count:,}")
        st.write(f"- **Title Vector**: `{len(data['tfidf_title'].vocabulary_):,} dims`")
        st.write(f"- **Artist Vector**: `{len(data['tfidf_artists'].vocabulary_):,} dims`")
        st.write(f"- **Genre Vector**: `{len(data['genre_vec'].vocabulary_):,} dims`")
        st.write(f"- **Audio Features**: `11 dims`")
        st.write(f"- **Data Structure**: `Sparse CSR Matrix`")
    except Exception as e:
        st.error(f"Error loading TF-IDF data: {e}")

# 2. Word2Vec
with col2:
    st.subheader("2. Word2Vec AI")
    try:
        w2v_emb, _ = get_w2v_embeddings()
        st.metric("Active Songs", f"{cleaned_songs_count:,}")
        st.write(f"- **Base Model**: `Google News 300d`")
        st.write(f"- **Vector Size**: `{w2v_emb.shape[1]} dims`")
        st.write(f"- **Architecture**: `Skip-gram/CBOW`")
        st.write(f"- **Alignment**: `Mapped to TF-IDF Index`")
    except Exception as e:
        st.warning("Word2Vec cache not found.")

# 3. GNN
with col3:
    st.subheader("3. GNN Collab Filtering")
    try:
        u_emb, s_emb = get_gnn_embeddings()
        dim = len(next(iter(s_emb.values())))
        st.metric("Active Songs", f"{cleaned_songs_count:,}")
        st.write(f"- **Trained Users**: `{len(u_emb):,}`")
        st.write(f"- **Architecture**: `Heterogeneous SAGEConv`")
        st.write(f"- **Embedding Size**: `{dim} dims`")
        st.write(f"- **Loss Function**: `BPR (Bayesian Personalized Ranking)`")
        
        # Load and display training stats if available
        import json
        import pandas as pd
        stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gnn_training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            st.write("---")
            st.markdown("##### Training Loss History")
            history = stats.get("history", [])
            if history:
                chart_data = pd.DataFrame(history).set_index("epoch")
                st.line_chart(chart_data)
                
    except Exception as e:
        st.warning("GNN not trained yet.")
