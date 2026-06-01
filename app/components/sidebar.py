"""
Sidebar navigation component
"""
import streamlit as st


def render_sidebar():
    """Render the sidebar navigation with clean menu structure."""
    with st.sidebar:
        st.markdown("## 🎵 Spotipai")
        st.divider()

        st.page_link("pages/1_home.py", label="Home", icon="🏠")
        st.page_link("pages/3_users.py", label="Search Users", icon="👥")
        st.page_link("pages/4_liked_songs.py", label="Liked Songs", icon="❤️")
        st.page_link("pages/5_profile.py", label="My Profile", icon="👤")

        # Spacer
        st.markdown("<br>" * 2, unsafe_allow_html=True)

        # User info at bottom
        st.divider()
        username = st.session_state.get("username", "User")
        st.markdown(
            f"""
            <div class="sidebar-user">
                <div class="user-name">Hello, {username}!</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
