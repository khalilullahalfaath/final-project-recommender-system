import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import streamlit as st

@st.cache_resource(show_spinner="Loading and preprocessing dataset for Recommender Engine...")
def get_cbf_data():
    """
    Load dataset and build vectors once per application run.
    """
    # Build absolute path to data folder
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset
    df_spotify = pd.read_csv(data_path)
    
    # Preprocessing
    df_spotify = df_spotify.dropna(subset=['artists', 'album_name', 'track_name'])
    df_spotify = df_spotify.drop_duplicates(subset=['track_id'], keep='first')
    
    # Text preprocessing
    df_spotify['artists_clean'] = df_spotify['artists'].str.lower().str.replace(r'[^\w\s&]', '', regex=True)
    df_spotify['track_name_clean'] = df_spotify['track_name'].str.lower()
    
    patterns_to_remove = [
        r'\s*-\s*remastered.*$',
        r'\s*-\s*live.*$',
        r'\s*-\s*radio edit.*$',
        r'\s*-\s*stereo.*$',
        r'\s*-\s*mono.*$'
    ]
    for pattern in patterns_to_remove:
        df_spotify['track_name_clean'] = df_spotify['track_name_clean'].str.replace(pattern, '', regex=True)

    # Gabung genre jika ada yg duplikat lagu+artis (seperti di notebook)
    agg_rules = {
        'track_genre': lambda x: ', '.join(x.dropna().unique()),
        'popularity': 'max',
        'track_id': 'first',
        'artists': 'first',
        'track_name': 'first',
        'album_name': 'first',
        'duration_ms': 'first',
        'explicit': 'first',
        'danceability': 'mean',
        'energy': 'mean',
        'key': 'first',
        'loudness': 'mean',
        'mode': 'first',
        'speechiness': 'mean',
        'acousticness': 'mean',
        'instrumentalness': 'mean',
        'liveness': 'mean',
        'valence': 'mean',
        'tempo': 'mean',
        'time_signature': 'first'
    }
    df_spotify = df_spotify.groupby(['artists_clean', 'track_name_clean']).agg(agg_rules).reset_index()

    # TF-IDF Artists
    tfidf_artists = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english')
    artists_features = tfidf_artists.fit_transform(df_spotify['artists_clean'])

    # TF-IDF Title
    tfidf_title = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english')
    title_features = tfidf_title.fit_transform(df_spotify['track_name_clean'])

    # Genre Vectorizer
    genre_vec = CountVectorizer(tokenizer=lambda x: x.split(', '))
    genre_features = genre_vec.fit_transform(df_spotify['track_genre'])

    # Audio features normalization
    cols_audio_to_normalize = ['loudness', 'tempo']
    audio_scaler = MinMaxScaler()
    df_spotify[cols_audio_to_normalize] = audio_scaler.fit_transform(df_spotify[cols_audio_to_normalize])
    
    # For Combined Features 1 (Energy, Valence)
    audio_features_sparse1 = csr_matrix(df_spotify[['energy', 'valence']])
    
    # For Combined Features 2 (Energy, Valence, Tempo, Loudness)
    audio_features_sparse2 = csr_matrix(df_spotify[['energy', 'valence', 'tempo', 'loudness']])

    combined_features1 = hstack([artists_features, genre_features, audio_features_sparse1]).tocsr()
    combined_features2 = hstack([artists_features, title_features, genre_features, audio_features_sparse2]).tocsr()
    
    # Collaborative Filtering Mock Data
    df_ratings = _load_cf_ratings()
    user_profiles_matrix, valid_user_ids = _build_mock_user_profiles(df_ratings, combined_features2, df_spotify)
    
    return {
        'df': df_spotify,
        'tfidf_artists': tfidf_artists,
        'genre_vec': genre_vec,
        'combined_features1': combined_features1,
        'combined_features2': combined_features2,
        'df_ratings': df_ratings,
        'user_profiles_matrix': user_profiles_matrix,
        'valid_user_ids': valid_user_ids
    }


def get_song_list():
    """Returns a list of 'Track Name - Artists' for the UI autocomplete."""
    data = get_cbf_data()
    df = data['df']
    if 'display_name' not in df.columns:
        df['display_name'] = df['track_name'] + " - " + df['artists']
    return df['display_name'].tolist()

def _load_cf_ratings():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.user_data import get_all_user_likes
    
    all_likes = get_all_user_likes()
    
    # Convert all users in JSON to df_ratings
    mock_data = []
    for user_id, songs in all_likes.items():
        for song in songs:
            # Default rating to 5 if not explicitly generated
            rating = song.get('rating', 5)
            mock_data.append({
                "user_id": user_id,
                "track_id": song["id"],
                "rating": rating
            })
            
    df_ratings = pd.DataFrame(mock_data) if mock_data else pd.DataFrame(columns=["user_id", "track_id", "rating"])
    return df_ratings

def _build_mock_user_profiles(df_ratings, combined_features2, df_spotify):
    """Build profile vectors for mock users based on their rated songs."""
    track_id_to_index = pd.Series(df_spotify.index, index=df_spotify['track_id']).to_dict()
    all_users = df_ratings['user_id'].unique()
    
    user_profiles_list = []
    valid_user_ids = []
    
    for id_user in all_users:
        data_user = df_ratings[df_ratings['user_id'] == id_user]
        lagu_user = data_user['track_id'].tolist()
        rating_user = data_user['rating'].values
        
        indeks_matriks = []
        rating_valid = []
        
        for lagu, rating in zip(lagu_user, rating_user):
            if lagu in track_id_to_index:
                indeks_matriks.append(track_id_to_index[lagu])
                rating_valid.append(rating)
                
        if len(indeks_matriks) == 0:
            continue
            
        rating_valid = np.array(rating_valid)
        fitur_lagu_pilihan = combined_features2[indeks_matriks]
        
        # Calculate weighted average based on ratings
        sum_rating = np.sum(rating_valid)
        if sum_rating > 0:
            user_profile_vector = fitur_lagu_pilihan.T.dot(rating_valid) / sum_rating
        else:
            user_profile_vector = np.zeros(combined_features2.shape[1])
            
        from scipy.sparse import vstack
        user_profile_sparse = csr_matrix(user_profile_vector)
        user_profiles_list.append(user_profile_sparse)
        valid_user_ids.append(id_user)
        
    from scipy.sparse import vstack
    if user_profiles_list:
        user_profiles_matrix = vstack(user_profiles_list)
    else:
        user_profiles_matrix = csr_matrix((0, combined_features2.shape[1]))
        
    return user_profiles_matrix, valid_user_ids


def _cover(song_id: str) -> str:
    """Mock a cover URL based on track_id string."""
    import hashlib
    # Create deterministic number from track_id string
    seed = int(hashlib.sha256(song_id.encode('utf-8')).hexdigest(), 16) % 10000
    return f"https://picsum.photos/seed/spotipai{seed}/300/300"


def recommend_from_criteria(artist="", genre="", target_energy=0.5, target_valence=0.5, use_energy=True, use_valence=True, top_n=5):
    """
    Scenario 1: Find songs based on specific criteria.
    """
    data = get_cbf_data()
    df_spotify = data['df']
    tfidf_artists = data['tfidf_artists']
    genre_vec = data['genre_vec']
    combined_features1 = data['combined_features1']

    # Audio array
    energy_val = target_energy if use_energy else 0.0
    valence_val = target_valence if use_valence else 0.0
    
    # Vectorize
    q_text1 = tfidf_artists.transform([artist if artist else ""])
    q_genre = genre_vec.transform([genre if genre else ""])
    q_audio = csr_matrix([[energy_val, valence_val]])

    # Combine
    q_vector = hstack([q_text1, q_genre, q_audio]).tocsr()
    
    # Adjust database features if ignoring audio
    from scipy.sparse import diags
    import numpy as np
    
    n_cols = combined_features1.shape[1]
    weights = np.ones(n_cols)
    if not use_energy:
        weights[-2] = 0.0
    if not use_valence:
        weights[-1] = 0.0
        
    adjusted_features = combined_features1.dot(diags(weights))

    # Similarity
    sim_scores = cosine_similarity(q_vector, adjusted_features)[0]
    top_indices = sim_scores.argsort()[::-1][:top_n]

    hasil = df_spotify.iloc[top_indices].copy()
    
    # Format to match UI
    results = []
    for _, row in hasil.iterrows():
        results.append({
            "id": row['track_id'],
            "title": row['track_name'],
            "artist": row['artists'],
            "genre": row['track_genre'],
            "cover_url": _cover(row['track_id']),
            "similarity": float(sim_scores[row.name])
        })
        
    return results


def recommend_from_seeds(seed_song_labels, top_n=5):
    """
    Scenario 2: Find songs similar to user's favorite tracks.
    Format input seed_song_labels: ["Yellow - Coldplay", "Baby - Justin Bieber;Ludacris"]
    """
    data = get_cbf_data()
    df_spotify = data['df']
    combined_features2 = data['combined_features2']

    if 'display_name' not in df_spotify.columns:
        df_spotify['display_name'] = df_spotify['track_name'] + " - " + df_spotify['artists']

    song_indices = []
    not_found = []
    
    for label in seed_song_labels:
        match = df_spotify[df_spotify['display_name'] == label]
        if not match.empty:
            song_indices.append(match.index[0])
        else:
            not_found.append(label)

    if not song_indices:
        return [], not_found

    seed_vectors = combined_features2[song_indices]
    user_profile_vector = seed_vectors.mean(axis=0)
    user_profile_vector = np.asarray(user_profile_vector).flatten()

    sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), combined_features2)[0]
    sorted_indices = sim_scores.argsort()[::-1]
    
    # Exclude input songs
    recommended_indices = [idx for idx in sorted_indices if idx not in song_indices][:top_n]

    hasil = df_spotify.iloc[recommended_indices].copy()
    
    # Format to match UI
    results = []
    for _, row in hasil.iterrows():
        results.append({
            "id": row['track_id'],
            "title": row['track_name'],
            "artist": row['artists'],
            "genre": row['track_genre'],
            "cover_url": _cover(row['track_id']),
            "similarity": float(sim_scores[row.name])
        })
        
    return results, not_found


def recommend_collaborative(username, top_n=5, top_users=3):
    """
    Scenario 3: Find songs using Collaborative Filtering (User-User).
    Matches current user profile against mock users.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from user_data import get_liked_songs
    
    data = get_cbf_data()
    df_spotify = data['df']
    combined_features2 = data['combined_features2']
    df_ratings = data['df_ratings']
    user_profiles_matrix = data['user_profiles_matrix']
    valid_user_ids = data['valid_user_ids']

    # Get user's liked songs
    liked_songs = get_liked_songs(username)
    if not liked_songs:
        return [], [], []

    song_indices = []
    not_found = []
    
    for song in liked_songs:
        track_id = song['id']
        match = df_spotify[df_spotify['track_id'] == track_id]
        if not match.empty:
            song_indices.append(match.index[0])
        else:
            not_found.append(f"{song['title']} - {song['artist']}")

    if not song_indices:
        return [], not_found, []

    # 1. Build profile vector for the CURRENT USER based on selected songs
    # Assume rating 5 for all selected liked songs
    seed_vectors = combined_features2[song_indices]
    current_user_profile = seed_vectors.mean(axis=0)
    current_user_sparse = csr_matrix(current_user_profile)
    
    # 2. Compute Cosine Similarity between current user and the mock users
    user_similarities = cosine_similarity(current_user_sparse, user_profiles_matrix)[0]
    
    # 3. Find top N similar users
    similar_user_indices = np.argsort(user_similarities)[::-1][:top_users]
    
    similar_users_info = []
    
    target_track_ids = df_spotify.iloc[song_indices]['track_id'].tolist()
    
    # 4. Generate candidate scores based on what similar users liked
    candidate_scores = {}
    candidate_sim_total = {}
    
    for idx in similar_user_indices:
        sim_score = user_similarities[idx]
        if sim_score <= 0:
            continue
            
        similar_user_id = valid_user_ids[idx]
        similar_users_info.append({"user_id": similar_user_id, "similarity": sim_score})
            
        similar_user_id = valid_user_ids[idx]
        # Get songs liked by this similar user
        user_data = df_ratings[df_ratings['user_id'] == similar_user_id]
        
        for _, row in user_data.iterrows():
            lagu = row['track_id']
            rating = row['rating']
            
            # Skip if current user already selected this song
            if lagu in target_track_ids:
                continue
                
            bobot = sim_score * rating
            
            if lagu in candidate_scores:
                candidate_scores[lagu] += bobot
                candidate_sim_total[lagu] += sim_score
            else:
                candidate_scores[lagu] = bobot
                candidate_sim_total[lagu] = sim_score
                
    # Normalize scores
    final_scores = {}
    for lagu, total_bobot in candidate_scores.items():
        total_sim = candidate_sim_total[lagu]
        final_scores[lagu] = total_bobot / total_sim if total_sim > 0 else 0
        
    # Sort and pick top N
    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Format to match UI
    results = []
    for track_id, score in sorted_candidates:
        match = df_spotify[df_spotify['track_id'] == track_id]
        if not match.empty:
            row = match.iloc[0]
            # Normalize score to a percentage scale (0-1) for UI if it's based on 1-5 ratings
            normalized_sim = min(score / 5.0, 1.0)
            
            results.append({
                "id": row['track_id'],
                "title": row['track_name'],
                "artist": row['artists'],
                "genre": row['track_genre'],
                "cover_url": _cover(row['track_id']),
                "similarity": normalized_sim
            })
            
    return results, not_found, similar_users_info

def get_mock_users_data():
    """Retrieve the generated mock users and their liked songs for the UI."""
    data = get_cbf_data()
    df_ratings = data['df_ratings']
    df_spotify = data['df']
    valid_user_ids = data['valid_user_ids']
    
    return {
        'df_ratings': df_ratings,
        'df_spotify': df_spotify,
        'users': valid_user_ids
    }
