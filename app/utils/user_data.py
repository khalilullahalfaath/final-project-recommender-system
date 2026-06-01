import json
import os
import streamlit as st

# Path to the JSON file where user likes are stored
# Use the data directory at the root of the project
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
USER_LIKES_FILE = os.path.join(DATA_DIR, "user_likes.json")

def _ensure_likes_file():
    """Ensure the user likes JSON file exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(USER_LIKES_FILE):
        with open(USER_LIKES_FILE, 'w') as f:
            json.dump({}, f)

def _load_all_likes():
    """Load all likes from the JSON file."""
    _ensure_likes_file()
    try:
        with open(USER_LIKES_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_all_likes(data):
    """Save all likes to the JSON file."""
    _ensure_likes_file()
    with open(USER_LIKES_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_all_user_likes():
    """Return the entire dictionary of user likes."""
    return _load_all_likes()

def save_bulk_likes(data):
    """Save the entire dictionary of user likes."""
    _save_all_likes(data)

def get_liked_songs(username):
    """
    Get the list of liked songs for a specific user.
    
    Args:
        username (str): The username of the user.
        
    Returns:
        list: A list of song dictionaries representing the user's liked songs.
    """
    if not username:
        return []
    
    data = _load_all_likes()
    return data.get(username, [])

def add_liked_song(username, song_data):
    """
    Add a song to a user's liked songs.
    
    Args:
        username (str): The username of the user.
        song_data (dict): The dictionary containing song info (id, title, artist, cover_url).
        
    Returns:
        bool: True if added successfully, False if already existed.
    """
    if not username or not song_data or "id" not in song_data:
        return False
        
    data = _load_all_likes()
    
    if username not in data:
        data[username] = []
        
    user_likes = data[username]
    
    # Check if song already exists in likes
    song_id = song_data["id"]
    if any(song["id"] == song_id for song in user_likes):
        return False # Already liked
        
    # Set default rating if not present
    if "rating" not in song_data:
        song_data["rating"] = 5
        
    user_likes.append(song_data)
    _save_all_likes(data)
    
    # Also update session state if it exists
    cache_key = f'liked_songs_cache_{username}'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    st.session_state[cache_key][song_id] = True
        
    return True

def remove_liked_song(username, song_id):
    """
    Remove a song from a user's liked songs.
    
    Args:
        username (str): The username of the user.
        song_id (str/int): The ID of the song to remove.
        
    Returns:
        bool: True if removed successfully, False if not found.
    """
    if not username or not song_id:
        return False
        
    data = _load_all_likes()
    
    if username not in data:
        return False
        
    user_likes = data[username]
    original_length = len(user_likes)
    
    # Filter out the song to remove
    data[username] = [song for song in user_likes if song["id"] != song_id]
    
    if len(data[username]) < original_length:
        _save_all_likes(data)
        
        # Update session state cache
        cache_key = f'liked_songs_cache_{username}'
        if cache_key in st.session_state and song_id in st.session_state[cache_key]:
            del st.session_state[cache_key][song_id]
            
        return True
        
    return False

def update_song_rating(username, song_id, new_rating):
    """
    Update the rating of a liked song.
    
    Args:
        username (str): The username of the user.
        song_id (str/int): The ID of the song to update.
        new_rating (int): The new rating (1-5).
        
    Returns:
        bool: True if updated successfully, False if not found.
    """
    if not username or not song_id:
        return False
        
    data = _load_all_likes()
    
    if username not in data:
        return False
        
    user_likes = data[username]
    updated = False
    
    for song in user_likes:
        if song["id"] == song_id:
            song["rating"] = new_rating
            updated = True
            break
            
    if updated:
        _save_all_likes(data)
        return True
        
    return False

def is_song_liked(username, song_id):
    """
    Check if a user has liked a specific song.
    Uses session_state cache for faster lookups during rendering.
    """
    if not username or not song_id:
        return False
        
    cache_key = f'liked_songs_cache_{username}'
    
    # Check cache first for faster rendering
    if cache_key in st.session_state:
        if song_id in st.session_state[cache_key]:
            return True
            
    # Fallback to loading from file
    likes = get_liked_songs(username)
    is_liked = any(song["id"] == song_id for song in likes)
    
    # Update cache
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    
    if is_liked:
        st.session_state[cache_key][song_id] = True
        
    return is_liked

def initialize_likes_cache(username):
    """Initialize the session state cache of liked songs for faster lookups."""
    if not username:
        return
        
    cache_key = f'liked_songs_cache_{username}'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
        
        likes = get_liked_songs(username)
        for song in likes:
            st.session_state[cache_key][song["id"]] = True
