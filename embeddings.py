import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')
# Define paths
chunks_dir = '/content/drive/MyDrive/chunks'
embeddings_dir = '/content/drive/MyDrive/embeddings'
os.makedirs(embeddings_dir, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
print("allMiniLM6v2 loaded")

def embed_chunks(chunks, batch_size=32):
    embeddings = []
    for i in range(0, len(chunks),32):#loops through the list of chunksin batches of 32 so that memory doesnt get overloaded
        batch = chunks[i:i+batch_size]#grabs a small portion from the full list
        batch_embeddings = model.encode(batch, convert_to_tensor=False)#MAIN thing-converting batch of text into vector
        embeddings.extend(batch_embeddings)#add those batch results in full embedding list
    return np.array(embeddings)

chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('_chunks.json')]#getting all chunk files

# Process each chunk file
all_embeddings=[]
all_chunks=[]
chunk_metadata=[]

for chunk_file in chunk_files:
    chunk_path = os.path.join(chunks_dir, chunk_file)
    with open(chunk_path, "r", encoding="utf-8") as f:#loading chunks
        chunks = json.load(f)
      
    embeddings = embed_chunks(chunks)#generating embeddings for all
    
    # Store data
    filename = chunk_file.replace('_chunks.json', '')
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        all_chunks.append(chunk)
        all_embeddings.append(embedding)
        chunk_metadata.append({
            'source_file': filename,
            'chunk_id': i,
            'chunk_text': chunk[:100] + "..." if len(chunk) > 100 else chunk  
        })
    print(f"Generated {len(embeddings)} embeddings")

all_embeddings = np.array(all_embeddings)#conversion in numpy array
print(f"{len(all_chunks)}")#total chunks processed
print(f"{all_embeddings.shape}")#embedding shape

embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
chunks_path = os.path.join(embeddings_dir, 'chunks.json')
metadata_path = os.path.join(embeddings_dir, 'metadata.json')

#saving embeddings
np.save(embeddings_path, all_embeddings)

#saving chunks
with open(chunks_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

with open(metadata_path, "w", encoding="utf-8") as f:#saving metaddata
    json.dump(chunk_metadata, f, indent=2)

print(f"\nEmbedding process completed!")

pickle_path = os.path.join(embeddings_dir, 'embedding_data.pkl')
embedding_data = {
    'embeddings': all_embeddings,
    'chunks': all_chunks,
    'metadata': chunk_metadata
}

with open(pickle_path, 'wb') as f:
    pickle.dump(embedding_data, f)

print(f"- Pickle file: {pickle_path}")
