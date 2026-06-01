import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

markdown_intro = """# Spotipai - Graph Neural Network (GNN) Exploration
This notebook explores the creation of a Heterogeneous Graph for the Spotify dataset and demonstrates how to prepare it for a Graph Neural Network (e.g., LightGCN or HeteroGNN) using `PyTorch Geometric`.

### Graph Schema
- **Nodes**: `User`, `Song`, `Artist`, `Genre`, `Album`
- **Edges**:
  - `User` -- [RATED] --> `Song` (edge weight: positive for rating >= 3, penalty for < 3)
  - `Song` -- [SUNG_BY] --> `Artist`
  - `Song` -- [HAS_GENRE] --> `Genre`
  - `Song` -- [BELONGS_TO] --> `Album`
"""

code_setup = """import pandas as pd
import numpy as np
import json
import torch
# Install PyTorch Geometric if not present:
# !pip install torch_geometric
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
"""

code_load_data = """# Load the dataset
data_path = "data/dataset.csv"
df_spotify = pd.read_csv(data_path)
df_spotify = df_spotify.dropna(subset=['artists', 'album_name', 'track_name'])
df_spotify = df_spotify.drop_duplicates(subset=['track_id'], keep='first')
print(f"Loaded {len(df_spotify)} unique songs.")

# Load User Likes (Ratings)
try:
    with open("data/user_likes.json", "r") as f:
        user_likes = json.load(f)
except FileNotFoundError:
    user_likes = {}
    
print(f"Loaded {len(user_likes)} users from user_likes.json.")
"""

code_build_mappings = """# Create unique IDs mapping for each node type
# 1. Songs
song_ids = df_spotify['track_id'].unique()
song_mapping = {id: i for i, id in enumerate(song_ids)}

# 2. Artists (Handling multiple artists separated by ';')
all_artists = set()
for artists_str in df_spotify['artists']:
    for artist in artists_str.split(';'):
        all_artists.add(artist.strip())
artist_mapping = {name: i for i, name in enumerate(all_artists)}

# 3. Genres
all_genres = set()
for genre_str in df_spotify['track_genre'].dropna():
    for genre in genre_str.split(','):
        all_genres.add(genre.strip())
genre_mapping = {name: i for i, name in enumerate(all_genres)}

# 4. Albums
album_names = df_spotify['album_name'].unique()
album_mapping = {name: i for i, name in enumerate(album_names)}

# 5. Users
user_mapping = {username: i for i, username in enumerate(user_likes.keys())}

print(f"Nodes count: Songs={len(song_mapping)}, Artists={len(artist_mapping)}, Genres={len(genre_mapping)}, Albums={len(album_mapping)}, Users={len(user_mapping)}")
"""

code_node_features = """# Build Node Features for Songs (Audio features)
audio_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_spotify[audio_cols] = scaler.fit_transform(df_spotify[audio_cols])

# Create feature tensor
song_features = torch.zeros((len(song_mapping), len(audio_cols)), dtype=torch.float)
for _, row in df_spotify.iterrows():
    idx = song_mapping[row['track_id']]
    song_features[idx] = torch.tensor(row[audio_cols].values, dtype=torch.float)
    
print(f"Song features shape: {song_features.shape}")
"""

code_edges = """# Build Edges
edge_song_artist = [[], []]
edge_song_genre = [[], []]
edge_song_album = [[], []]

for _, row in df_spotify.iterrows():
    song_idx = song_mapping[row['track_id']]
    
    # Artist edges
    for artist in row['artists'].split(';'):
        if artist.strip() in artist_mapping:
            edge_song_artist[0].append(song_idx)
            edge_song_artist[1].append(artist_mapping[artist.strip()])
            
    # Genre edges
    for genre in str(row['track_genre']).split(','):
        if genre.strip() in genre_mapping:
            edge_song_genre[0].append(song_idx)
            edge_song_genre[1].append(genre_mapping[genre.strip()])
            
    # Album edges
    album_idx = album_mapping[row['album_name']]
    edge_song_album[0].append(song_idx)
    edge_song_album[1].append(album_idx)

# User-Song Edges
edge_user_song = [[], []]
edge_user_song_weights = []

for username, songs in user_likes.items():
    u_idx = user_mapping[username]
    for song in songs:
        if song['id'] in song_mapping:
            s_idx = song_mapping[song['id']]
            rating = song.get('rating', 5)
            
            # PENALTY LOGIC:
            # Rating >= 3 is positive. Rating < 3 is negative (penalty).
            if rating >= 3:
                weight = 1.0 # Positive interaction
            else:
                weight = -1.0 # Negative interaction penalty
                
            edge_user_song[0].append(u_idx)
            edge_user_song[1].append(s_idx)
            edge_user_song_weights.append(weight)

# Convert to tensors
edge_index_song_artist = torch.tensor(edge_song_artist, dtype=torch.long)
edge_index_song_genre = torch.tensor(edge_song_genre, dtype=torch.long)
edge_index_song_album = torch.tensor(edge_song_album, dtype=torch.long)
edge_index_user_song = torch.tensor(edge_user_song, dtype=torch.long)
edge_attr_user_song = torch.tensor(edge_user_song_weights, dtype=torch.float)

print(f"User->Song edges: {edge_index_user_song.size(1)}")
"""

code_pyg = """# Construct the Heterogeneous Graph using PyTorch Geometric
data = HeteroData()

# Add nodes
data['song'].x = song_features # (num_songs, num_features)
data['user'].num_nodes = len(user_mapping)
data['artist'].num_nodes = len(artist_mapping)
data['genre'].num_nodes = len(genre_mapping)
data['album'].num_nodes = len(album_mapping)

# Add edges
data['user', 'rates', 'song'].edge_index = edge_index_user_song
data['user', 'rates', 'song'].edge_attr = edge_attr_user_song

data['song', 'sung_by', 'artist'].edge_index = edge_index_song_artist
data['song', 'has_genre', 'genre'].edge_index = edge_index_song_genre
data['song', 'belongs_to', 'album'].edge_index = edge_index_song_album

# Also add reverse edges (useful for message passing from item back to user)
import torch_geometric.transforms as T
data = T.ToUndirected()(data)

print(data)
"""

code_train_placeholder = """# Next Steps for Training:
# 1. Define a Heterogeneous GNN (e.g. using `SAGEConv` or `HeteroConv`).
# 2. Split the 'user, rates, song' edges into train/val/test.
# 3. Use Link Prediction (dot product between user embeddings and song embeddings).
# 4. Train the model using BPR (Bayesian Personalized Ranking) Loss or MSE on the edge weights.

print("Graph is ready for GNN Training!")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_code_cell(code_build_mappings),
    nbf.v4.new_code_cell(code_node_features),
    nbf.v4.new_code_cell(code_edges),
    nbf.v4.new_code_cell(code_pyg),
    nbf.v4.new_code_cell(code_train_placeholder)
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/gnn_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
