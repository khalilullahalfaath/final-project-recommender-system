"""
Session state management for the Streamlit app
"""
import streamlit as st


def init_session_state():
    """Initialize all session state variables with defaults"""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    
    if 'users' not in st.session_state:
        try:
            from utils.auth import _load_users
            st.session_state['users'] = _load_users()
        except:
            st.session_state['users'] = {}
    
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state['sidebar_collapsed'] = False
    
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = 'Collaborative Filtering'


def is_authenticated():
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)


def logout():
    """Clear authentication session state"""
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
