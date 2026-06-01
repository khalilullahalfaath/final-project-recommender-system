"""
Theme toggle component for switching between dark and light modes
"""
import streamlit as st


def render_theme_toggle():
    """Render compact icon-only theme toggle button."""
    current_theme = st.session_state.get("theme", "dark")

    if current_theme == "dark":
        icon = "☀️"
        tooltip = "Switch to light mode"
    else:
        icon = "🌙"
        tooltip = "Switch to dark mode"

    if st.button(icon, key="theme_toggle", help=tooltip):
        st.session_state["theme"] = "light" if current_theme == "dark" else "dark"
        st.rerun()
