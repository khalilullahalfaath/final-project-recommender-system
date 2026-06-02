import os
import sys
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add parent dir to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

DATA_DIR = os.path.join(base_dir, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "semantic_embeddings.npy")
TRACK_IDS_PATH = os.path.join(DATA_DIR, "semantic_track_ids.npy")

def main():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    
    # Create combined text for each song
    # Example: "SongName by ArtistName in GenreName genre"
    print("Preparing text for embedding...")
    df['text_for_embedding'] = df.apply(
        lambda row: f"{row['track_name']} by {row['artists']}. Genre: {row['track_genre']}", 
        axis=1
    )
    
    texts = df['text_for_embedding'].tolist()
    track_ids = df['track_id'].tolist()
    
    print(f"Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Computing embeddings for {len(texts)} songs... This may take 1-3 minutes.")
    # Show progress bar if possible, batch size 256 for speed
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    print(f"Saving to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(TRACK_IDS_PATH, np.array(track_ids))
    
    print("Done! Semantic embeddings successfully cached.")

if __name__ == "__main__":
    main()
