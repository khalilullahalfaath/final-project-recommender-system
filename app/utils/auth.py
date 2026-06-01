"""
Authentication helpers for user login and registration
"""
import streamlit as st
import json
import os

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.json")

def _load_users():
    """Load users from local JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def _save_users(users_data):
    """Save users to local JSON file."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)


def register_user(username, password):
    """
    Register a new user
    
    Args:
        username: Username to register
        password: Password for the user
    
    Returns:
        bool: True if registration successful, False if username already exists
    """
    users = _load_users()
    
    if username in users:
        return False
    
    users[username] = password
    _save_users(users)
    
    # Update session state as well
    if 'users' not in st.session_state:
        st.session_state['users'] = {}
    st.session_state['users'][username] = password
    
    return True


def login_user(username, password):
    """
    Authenticate and login a user
    
    Args:
        username: Username to login
        password: Password to verify
    
    Returns:
        bool: True if login successful, False otherwise
    """
    users = _load_users()
    
    if username not in users:
        return False
    
    if users[username] != password:
        return False
    
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    
    # Ensure it's in session state
    if 'users' not in st.session_state:
        st.session_state['users'] = users
        
    return True


def logout_user():
    """Logout the current user"""
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
