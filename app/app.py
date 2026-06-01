"""
Main Streamlit app - Login and Registration
"""

import streamlit as st
from utils.session import init_session_state, is_authenticated
from utils.auth import login_user, register_user
from components.sidebar import render_sidebar
import os

# Page configuration
st.set_page_config(
    page_title="Spotipai - Your Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

render_sidebar()


# Load CSS based on theme
def load_css():
    """Load CSS file based on current theme"""
    theme = st.session_state.get("theme", "dark")
    css_file = f"styles/{theme}_theme.css"
    css_path = os.path.join(os.path.dirname(__file__), css_file)

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Check if user is already logged in
if is_authenticated():
    st.switch_page("pages/1_home.py")

# Toggle between login and register
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

# Centered layout via columns
_, center, _ = st.columns([1, 1, 1])

with center:
    # Hero section
    st.markdown('<div class="auth-hero">🎵 Spotipai</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="auth-tagline">Your music recommendation system</p>',
        unsafe_allow_html=True,
    )

    # Form
    with st.form(key="auth_form"):
        if st.session_state["auth_mode"] == "login":
            st.markdown(
                '<div class="auth-form-title">Welcome back</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="auth-form-title">Create your account</div>',
                unsafe_allow_html=True,
            )

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )

        if st.session_state["auth_mode"] == "login":
            submit_button = st.form_submit_button(
                "Login", use_container_width=True, type="primary"
            )
        else:
            submit_button = st.form_submit_button(
                "Register", use_container_width=True, type="primary"
            )

        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                if st.session_state["auth_mode"] == "login":
                    # Login logic
                    if login_user(username, password):
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    # Register logic
                    if register_user(username, password):
                        st.success("Registration successful! Please login.")
                        st.session_state["auth_mode"] = "login"
                        st.rerun()
                    else:
                        st.error("Username already exists")

    # Demo divider + button
    st.markdown(
        '<div class="auth-divider"><span>or</span></div>', unsafe_allow_html=True
    )

    col_demo1, col_demo2 = st.columns([1, 2])
    with col_demo1:
        if st.button("🚀 Run Demo", use_container_width=True, type="secondary"):
            demo_username = "demo"
            demo_password = "demo123"

            # Register demo account if it doesn't exist (idempotent)
            if demo_username not in st.session_state["users"]:
                register_user(demo_username, demo_password)

            # Auto-login
            login_user(demo_username, demo_password)
            st.success("Welcome to the demo!")
            st.rerun()
            
    with col_demo2:
        col_sel, col_btn = st.columns([2, 3])
        with col_sel:
            mock_user = st.selectbox("Mock User", options=[f"user_{i}" for i in range(1, 101)], label_visibility="collapsed")
        with col_btn:
            if st.button(f"👤 Login as Mock User", use_container_width=True, type="secondary"):
                if mock_user not in st.session_state["users"]:
                    register_user(mock_user, "mock123")
                    
                login_user(mock_user, "mock123")
                st.success(f"Welcome {mock_user}!")
                st.rerun()

    # Toggle link
    st.markdown('<div class="toggle-link-btn">', unsafe_allow_html=True)
    if st.session_state["auth_mode"] == "login":
        if st.button("Don't have an account? Register here", use_container_width=True):
            st.session_state["auth_mode"] = "register"
            st.rerun()
    else:
        if st.button("Already have an account? Login here", use_container_width=True):
            st.session_state["auth_mode"] = "login"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
