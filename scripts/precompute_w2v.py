import os
import sys
import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors

# Add parent dir to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

DATA_DIR = os.path.join(base_dir, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "word2vec", "GoogleNews-vectors-negative300-1.bin")

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "w2v_embeddings.npy")
TRACK_IDS_PATH = os.path.join(DATA_DIR, "w2v_track_ids.npy")

def clean_and_tokenize(text):
    # Remove non-alphanumeric characters, convert to lowercase, split by space
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [word for word in text.split() if word.strip()]

def get_mean_vector(tokens, model, vector_size=300):
    valid_vectors = []
    for token in tokens:
        # Check both lowercase and original title case (GoogleNews is case sensitive)
        if token in model:
            valid_vectors.append(model[token])
        elif token.capitalize() in model:
            valid_vectors.append(model[token.capitalize()])
            
    if len(valid_vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(valid_vectors, axis=0)

def main():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])
    df = df.drop_duplicates(subset=['track_id'], keep='first')
    
    print(f"Loading Google News Word2Vec model from {MODEL_PATH}...")
    print("This requires ~3.6GB RAM and may take a minute.")
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    vector_size = model.vector_size
    
    print("Preparing text for embedding...")
    # Combine track name, artists and genre
    df['text_for_embedding'] = df['track_name'] + " " + df['artists'] + " " + df['track_genre']
    
    track_ids = df['track_id'].tolist()
    texts = df['text_for_embedding'].tolist()
    
    print(f"Computing embeddings for {len(texts)} songs via Mean Pooling...")
    embeddings = []
    
    for i, text in enumerate(texts):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} / {len(texts)} songs...")
            
        tokens = clean_and_tokenize(text)
        vec = get_mean_vector(tokens, model, vector_size)
        embeddings.append(vec)
        
    embeddings_array = np.array(embeddings)
    
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    print(f"Saving to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings_array)
    np.save(TRACK_IDS_PATH, np.array(track_ids))
    
    print("Done! Word2Vec embeddings successfully cached.")

if __name__ == "__main__":
    main()
