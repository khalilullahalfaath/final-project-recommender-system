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
        'tfidf_title': tfidf_title,
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


def recommend_from_criteria(artist="", genre="", target_energy=0.5, target_valence=0.5, use_energy=True, use_valence=True, top_n=5, use_w2v=False):
    """
    Scenario 1: Find songs based on specific criteria.
    """
    import numpy as np
    from scipy.sparse import diags, hstack, csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    
    data = get_cbf_data()
    df_spotify = data['df']
    
    # Audio array
    energy_val = target_energy if use_energy else 0.0
    valence_val = target_valence if use_valence else 0.0

    # Fallback: if no text provided, ignore Word2Vec and fallback to TF-IDF logic
    if use_w2v and not artist.strip() and not genre.strip():
        use_w2v = False

    if not use_w2v:
        tfidf_artists = data['tfidf_artists']
        genre_vec = data['genre_vec']
        combined_features1 = data['combined_features1']
        
        # Vectorize
        q_text1 = tfidf_artists.transform([artist if artist else ""])
        q_genre = genre_vec.transform([genre if genre else ""])
        q_audio = csr_matrix([[energy_val, valence_val]])
    
        # Combine
        q_vector = hstack([q_text1, q_genre, q_audio]).tocsr()
        
        # Adjust database features if ignoring audio
        n_cols = combined_features1.shape[1]
        weights = np.ones(n_cols)
        if not use_energy: weights[-2] = 0.0
        if not use_valence: weights[-1] = 0.0
            
        adjusted_features = combined_features1.dot(diags(weights))
    else:
        w2v_model = get_w2v_model()
        aligned_w2v = get_aligned_w2v_matrix()
        
        audio_features = df_spotify[['energy', 'valence']].values
        adjusted_features = np.hstack([aligned_w2v, audio_features])
        
        query_text = f"{artist} {genre}".strip()
        tokens = clean_and_tokenize(query_text)
        valid_vectors = []
        for token in tokens:
            if token in w2v_model:
                valid_vectors.append(w2v_model[token])
            elif token.capitalize() in w2v_model:
                valid_vectors.append(w2v_model[token.capitalize()])
                
        if not valid_vectors:
            q_text_vec = np.zeros(w2v_emb.shape[1])
        else:
            q_text_vec = np.mean(valid_vectors, axis=0)
            
        q_vector = np.concatenate([q_text_vec, [energy_val, valence_val]]).reshape(1, -1)
        
        weights = np.ones(adjusted_features.shape[1])
        if not use_energy: weights[-2] = 0.0
        if not use_valence: weights[-1] = 0.0
        
        adjusted_features = adjusted_features * weights

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


def recommend_from_seeds(seed_song_labels, top_n=5, use_w2v=False):
    """
    Scenario 2: Find songs similar to user's favorite tracks.
    Format input seed_song_labels: ["Yellow - Coldplay", "Baby - Justin Bieber;Ludacris"]
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    data = get_cbf_data()
    df_spotify = data['df']

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

    if not use_w2v:
        combined_features2 = data['combined_features2']
        seed_vectors = combined_features2[song_indices]
        user_profile_vector = seed_vectors.mean(axis=0)
        user_profile_vector = np.asarray(user_profile_vector).flatten()
    
        sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), combined_features2)[0]
    else:
        aligned_w2v = get_aligned_w2v_matrix()
        audio_features = df_spotify[['energy', 'valence']].values
        adjusted_features = np.hstack([aligned_w2v, audio_features])
        
        seed_vectors = adjusted_features[song_indices]
        user_profile_vector = seed_vectors.mean(axis=0)
        
        sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), adjusted_features)[0]

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

# ==========================================
# Word2Vec Search Functions (Content-Based)
# ==========================================
import re

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [word for word in text.split() if word.strip()]

@st.cache_resource(show_spinner="Loading Google News Word2Vec Model (3.6GB)...")
def get_w2v_model():
    import os
    from gensim.models import KeyedVectors
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "data", "word2vec", "GoogleNews-vectors-negative300-1.bin")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    return KeyedVectors.load_word2vec_format(model_path, binary=True)

@st.cache_resource(show_spinner="Loading Word2Vec Song Embeddings...")
def get_w2v_embeddings():
    import os
    import numpy as np
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    embeddings_path = os.path.join(base_dir, "data", "w2v_embeddings.npy")
    track_ids_path = os.path.join(base_dir, "data", "w2v_track_ids.npy")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(track_ids_path):
        raise FileNotFoundError("Word2Vec embeddings cache not found. Please run scripts/precompute_w2v.py first.")
        
    embeddings = np.load(embeddings_path)
    track_ids = np.load(track_ids_path)
    return embeddings, track_ids

@st.cache_resource(show_spinner="Aligning Word2Vec Embeddings with Dataset...")
def get_aligned_w2v_matrix():
    import numpy as np
    data = get_cbf_data()
    df = data['df']
    w2v_emb, track_ids = get_w2v_embeddings()
    
    track_id_to_idx = {tid: idx for idx, tid in enumerate(track_ids)}
    
    aligned_w2v = []
    for tid in df['track_id']:
        idx = track_id_to_idx[tid]
        aligned_w2v.append(w2v_emb[idx])
        
    return np.array(aligned_w2v)

def recommend_word2vec(query_text, top_n=5):
    """
    Recommend songs based on word2vec cosine similarity.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = get_w2v_model()
    aligned_w2v = get_aligned_w2v_matrix()
    data = get_cbf_data()
    df = data['df']
    track_ids = df['track_id'].values
    
    # Process query
    tokens = clean_and_tokenize(query_text)
    valid_vectors = []
    for token in tokens:
        if token in model:
            valid_vectors.append(model[token])
        elif token.capitalize() in model:
            valid_vectors.append(model[token.capitalize()])
            
    if not valid_vectors:
        return [] # No valid words found
        
    query_vector = np.mean(valid_vectors, axis=0).reshape(1, -1)
    
    # Calculate similarity
    similarities = cosine_similarity(query_vector, aligned_w2v)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_track_ids = track_ids[top_indices]
    
    recommendations = []
    for i, t_id in enumerate(top_track_ids):
        match = df[df['track_id'] == t_id]
        if not match.empty:
            song = match.iloc[0]
            recommendations.append({
                "id": t_id,
                "title": song['track_name'],
                "artist": song['artists'],
                "genre": song['track_genre'],
                "cover_url": _cover(t_id),
                "similarity": float(similarities[top_indices[i]])
            })
            
    return recommendations

# ==========================================
# GNN Collaborative Filtering
# ==========================================

@st.cache_resource(show_spinner="Loading GNN Embeddings...")
def get_gnn_embeddings():
    import os
    import pickle
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    emb_path = os.path.join(base_dir, "data", "gnn_embeddings.pkl")
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError("GNN embeddings not found. Please run scripts/train_gnn.py first.")
        
    with open(emb_path, "rb") as f:
        data = pickle.load(f)
        
    return data['user_embeddings'], data['song_embeddings']

def recommend_gnn(user_id, top_n=5):
    """
    Recommend songs using GNN Collaborative Filtering.
    """
    import numpy as np
    from utils.user_data import get_liked_songs
    
    user_embeddings, song_embeddings = get_gnn_embeddings()
    data = get_cbf_data()
    df = data['df']
    
    likes = get_liked_songs(user_id)
    liked_ids = {s['id'] for s in likes}
    
    # 1. Compute User Embedding (Always Inductive/Dynamic for real-time updates)
    if not likes:
        raise ValueError(f"User '{user_id}' has no ratings. Please rate some songs first in 'Set Your Profile'!")
        
    valid_embs = []
    for song in likes:
        tid = song['id']
        if tid in song_embeddings:
            valid_embs.append(song_embeddings[tid])
            
    if not valid_embs:
        raise ValueError("None of your liked songs are recognized by the GNN model.")
        
    u_emb = np.mean(valid_embs, axis=0)
        
    # 2. Compute Dot Product with all songs
    track_ids = list(song_embeddings.keys())
    s_embs = np.array(list(song_embeddings.values()))
    
    scores = np.dot(s_embs, u_emb)
    
    # 3. Sort and format
    top_indices = np.argsort(scores)[::-1]
    
    recommendations = []
    for idx in top_indices:
        tid = track_ids[idx]
        if tid in liked_ids:
            continue
            
        match = df[df['track_id'] == tid]
        if not match.empty:
            song = match.iloc[0]
            # Convert raw dot product to probability using sigmoid for UI consistency
            prob = 1 / (1 + np.exp(-scores[idx]))
            
            recommendations.append({
                "id": tid,
                "title": song['track_name'],
                "artist": song['artists'],
                "genre": song['track_genre'],
                "cover_url": _cover(tid),
                "similarity": float(prob)
            })
            if len(recommendations) >= top_n:
                break
                
    return recommendations
