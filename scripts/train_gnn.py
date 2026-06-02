import os
import sys
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, HeteroConv
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

DATA_DIR = os.path.join(base_dir, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
USER_LIKES_PATH = os.path.join(DATA_DIR, "user_likes.json")
EMB_PATH = os.path.join(DATA_DIR, "gnn_embeddings.pkl")
STATS_PATH = os.path.join(DATA_DIR, "gnn_training_stats.json")

def load_data():
    df_spotify = pd.read_csv(DATASET_PATH)
    df_spotify = df_spotify.dropna(subset=['artists', 'album_name', 'track_name'])
    df_spotify = df_spotify.drop_duplicates(subset=['track_id'], keep='first')
    
    try:
        with open(USER_LIKES_PATH, "r") as f:
            user_likes = json.load(f)
    except FileNotFoundError:
        user_likes = {}
        
    return df_spotify, user_likes

def build_graph(df_spotify, user_likes):
    song_ids = df_spotify['track_id'].unique()
    song_mapping = {id: i for i, id in enumerate(song_ids)}

    all_artists = set()
    for artists_str in df_spotify['artists']:
        for artist in artists_str.split(';'):
            all_artists.add(artist.strip())
    artist_mapping = {name: i for i, name in enumerate(all_artists)}

    all_genres = set()
    for genre_str in df_spotify['track_genre'].dropna():
        for genre in genre_str.split(','):
            all_genres.add(genre.strip())
    genre_mapping = {name: i for i, name in enumerate(all_genres)}

    album_names = df_spotify['album_name'].unique()
    album_mapping = {name: i for i, name in enumerate(album_names)}

    user_mapping = {username: i for i, username in enumerate(user_likes.keys())}
    
    print(f"Graph nodes: Songs={len(song_mapping)}, Artists={len(artist_mapping)}, Users={len(user_mapping)}")

    audio_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_spotify[audio_cols] = scaler.fit_transform(df_spotify[audio_cols])

    song_features = torch.zeros((len(song_mapping), len(audio_cols)), dtype=torch.float)
    for _, row in df_spotify.iterrows():
        idx = song_mapping[row['track_id']]
        song_features[idx] = torch.tensor(row[audio_cols].values.astype(float), dtype=torch.float)

    edge_song_artist = [[], []]
    edge_song_genre = [[], []]
    edge_song_album = [[], []]

    for _, row in df_spotify.iterrows():
        song_idx = song_mapping[row['track_id']]
        for artist in row['artists'].split(';'):
            if artist.strip() in artist_mapping:
                edge_song_artist[0].append(song_idx)
                edge_song_artist[1].append(artist_mapping[artist.strip()])
                
        for genre in str(row['track_genre']).split(','):
            if genre.strip() in genre_mapping:
                edge_song_genre[0].append(song_idx)
                edge_song_genre[1].append(genre_mapping[genre.strip()])
                
        album_idx = album_mapping[row['album_name']]
        edge_song_album[0].append(song_idx)
        edge_song_album[1].append(album_idx)

    edge_user_song = [[], []]
    edge_user_song_weights = []

    for username, songs in user_likes.items():
        u_idx = user_mapping[username]
        for song in songs:
            if song['id'] in song_mapping:
                s_idx = song_mapping[song['id']]
                rating = song.get('rating', 5)
                weight = 1.0 if rating >= 3 else -1.0
                edge_user_song[0].append(u_idx)
                edge_user_song[1].append(s_idx)
                edge_user_song_weights.append(weight)

    data = HeteroData()
    data['song'].x = song_features
    data['user'].num_nodes = len(user_mapping)
    data['artist'].num_nodes = len(artist_mapping)
    data['genre'].num_nodes = len(genre_mapping)
    data['album'].num_nodes = len(album_mapping)

    data['user', 'rates', 'song'].edge_index = torch.tensor(edge_user_song, dtype=torch.long)
    data['song', 'sung_by', 'artist'].edge_index = torch.tensor(edge_song_artist, dtype=torch.long)
    data['song', 'has_genre', 'genre'].edge_index = torch.tensor(edge_song_genre, dtype=torch.long)
    data['song', 'belongs_to', 'album'].edge_index = torch.tensor(edge_song_album, dtype=torch.long)

    # Add reverse edges automatically
    data = T.ToUndirected()(data)
    
    return data, song_mapping, user_mapping

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, data):
        super().__init__()
        self.user_emb = torch.nn.Embedding(data['user'].num_nodes, hidden_channels)
        self.artist_emb = torch.nn.Embedding(data['artist'].num_nodes, hidden_channels)
        self.genre_emb = torch.nn.Embedding(data['genre'].num_nodes, hidden_channels)
        self.album_emb = torch.nn.Embedding(data['album'].num_nodes, hidden_channels)
        
        self.song_proj = torch.nn.Linear(data['song'].x.size(1), hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'rates', 'song'): SAGEConv((-1, -1), hidden_channels),
                ('song', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
                ('song', 'sung_by', 'artist'): SAGEConv((-1, -1), hidden_channels),
                ('artist', 'rev_sung_by', 'song'): SAGEConv((-1, -1), hidden_channels),
                ('song', 'has_genre', 'genre'): SAGEConv((-1, -1), hidden_channels),
                ('genre', 'rev_has_genre', 'song'): SAGEConv((-1, -1), hidden_channels),
                ('song', 'belongs_to', 'album'): SAGEConv((-1, -1), hidden_channels),
                ('album', 'rev_belongs_to', 'song'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'user': self.user_emb.weight,
            'artist': self.artist_emb.weight,
            'genre': self.genre_emb.weight,
            'album': self.album_emb.weight,
            'song': self.song_proj(x_dict['song'])
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

def sample_negative_edges(edge_index, num_nodes_u, num_nodes_v, num_samples):
    u = torch.randint(0, num_nodes_u, (num_samples,))
    v = torch.randint(0, num_nodes_v, (num_samples,))
    return torch.stack([u, v], dim=0)

def main():
    print("Loading data...")
    df_spotify, user_likes = load_data()
    if not user_likes:
        print("No users found. Creating a dummy user for training.")
        user_likes["dummy_user"] = [{"id": df_spotify.iloc[0]['track_id'], "rating": 5}]
        
    print("Building Heterogeneous Graph...")
    data, song_mapping, user_mapping = build_graph(df_spotify, user_likes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    hidden_channels = 64
    model = HeteroGNN(hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=2, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting Training (BPR Loss)...")
    model.train()
    epochs = 30
    edge_index = data['user', 'rates', 'song'].edge_index
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        user_emb = x_dict['user']
        song_emb = x_dict['song']
        
        pos_scores = (user_emb[edge_index[0]] * song_emb[edge_index[1]]).sum(dim=-1)
        
        neg_edge_index = sample_negative_edges(edge_index, data['user'].num_nodes, data['song'].num_nodes, edge_index.size(1))
        neg_edge_index = neg_edge_index.to(device)
        
        neg_scores = (user_emb[neg_edge_index[0]] * song_emb[neg_edge_index[1]]).sum(dim=-1)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append({"epoch": epoch + 1, "loss": loss_val})
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}, Loss: {loss_val:.4f}")

    print("Training finished. Extracting final embeddings...")
    model.eval()
    with torch.no_grad():
        final_x_dict = model(data.x_dict, data.edge_index_dict)
        u_emb = final_x_dict['user'].cpu().numpy()
        s_emb = final_x_dict['song'].cpu().numpy()
        
    # Invert mappings for easy lookup
    idx_to_song = {idx: sid for sid, idx in song_mapping.items()}
    idx_to_user = {idx: uid for uid, idx in user_mapping.items()}
    
    # Save as a dictionary matching ID to embedding vector
    user_embeddings = {idx_to_user[i]: u_emb[i] for i in range(len(idx_to_user))}
    song_embeddings = {idx_to_song[i]: s_emb[i] for i in range(len(idx_to_song))}
    
    output_data = {
        "user_embeddings": user_embeddings,
        "song_embeddings": song_embeddings
    }
    
    with open(EMB_PATH, "wb") as f:
        pickle.dump(output_data, f)
        
    stats_data = {
        "epochs": epochs,
        "final_loss": loss_history[-1]["loss"],
        "history": loss_history
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats_data, f, indent=4)
        
    print(f"Embeddings saved to {EMB_PATH}")
    print(f"Stats saved to {STATS_PATH}")
    print(f"Saved {len(user_embeddings)} user embeddings and {len(song_embeddings)} song embeddings.")

if __name__ == "__main__":
    main()
