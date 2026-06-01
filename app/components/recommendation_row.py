"""
Horizontal scrollable recommendation row component using st.columns + st.image
"""
import streamlit as st
import sys
import os

# Add parent directory to path to import from app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.user_data import is_song_liked, add_liked_song, remove_liked_song
def render_recommendation_row(title, songs, key_prefix=None):
    """
    Render a grid of song recommendations using Streamlit native columns.

    Args:
        title: Section title (e.g., "Users Like You Enjoyed")
        songs: List of song dictionaries with id, title, artist, cover_url
        key_prefix: Optional string to ensure unique keys when rendering multiple rows with empty titles
    """
    if title:
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        
    prefix = key_prefix if key_prefix else title.replace(' ', '_')

    # Split songs into rows of 5 cards each
    cards_per_row = 5
    for i in range(0, len(songs), cards_per_row):
        row_songs = songs[i : i + cards_per_row]
        cols = st.columns(cards_per_row)

        for col, song in zip(cols, row_songs):
            with col:
                # Wrap card in a div for hover styling
                st.markdown('<div class="song-card-wrap">', unsafe_allow_html=True)

                # Cover image
                st.image(
                    song["cover_url"],
                    use_container_width=True,
                )

                # Title and artist
                st.markdown(
                    f'<div class="song-title">{song["title"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="song-artist">{song["artist"]}</div>',
                    unsafe_allow_html=True,
                )
                
                if "similarity" in song:
                    sim_percent = f"{song['similarity'] * 100:.1f}% Match"
                    st.markdown(
                        f'<div class="song-similarity" style="font-size: 0.8rem; color: #1DB954; font-weight: bold; margin-top: 4px;">{sim_percent}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add Like/Unlike button
                username = st.session_state.get("username")
                if username:
                    is_liked = is_song_liked(username, song["id"])
                    
                    button_label = "❤️ Liked" if is_liked else "🤍 Like"
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(button_label, key=f"like_btn_{prefix}_{song['id']}_{i}", use_container_width=True):
                            if is_liked:
                                remove_liked_song(username, song["id"])
                            else:
                                # Pass full song data without the similarity key so it stores cleanly
                                clean_song = {k:v for k,v in song.items() if k != "similarity"}
                                add_liked_song(username, clean_song)
                            st.rerun()
                    with btn_col2:
                        if st.button("ℹ️ Details", key=f"detail_btn_{prefix}_{song['id']}_{i}", use_container_width=True):
                            st.session_state["selected_track_id"] = song["id"]
                            st.switch_page("pages/6_song_detail.py")
