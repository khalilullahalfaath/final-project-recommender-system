"""
Top navigation bar component
"""
import streamlit as st
from components.theme_toggle import render_theme_toggle
from utils.auth import logout_user


def render_navbar():
    """Render the top navigation bar with flat 4-column layout."""
    # Flat layout: logo | search | theme | logout
    col_logo, col_search, col_theme, col_logout = st.columns([2, 5, 1.2, 1.2])

    with col_logo:
        st.markdown('<p class="navbar-logo">🎵 Spotipai</p>', unsafe_allow_html=True)

    with col_search:
        st.markdown('<div class="navbar-search">', unsafe_allow_html=True)
        def handle_search():
            if st.session_state.get("search_bar"):
                st.session_state["search_query"] = st.session_state["search_bar"]
                st.session_state["navigate_to_search"] = True
                
        st.text_input(
            "Search",
            placeholder="Search for songs, artists...",
            key="search_bar",
            label_visibility="collapsed",
            on_change=handle_search
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.get("navigate_to_search"):
            st.session_state["navigate_to_search"] = False
            st.switch_page("pages/7_search.py")

    with col_theme:
        st.markdown('<div class="navbar-action">', unsafe_allow_html=True)
        render_theme_toggle()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_logout:
        st.markdown('<div class="navbar-action">', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn"):
            logout_user()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
