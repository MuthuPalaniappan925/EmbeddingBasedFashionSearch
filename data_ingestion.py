## Importing Packages
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

## Loading Data
data = pd.read_csv("Dataset/Myntra Fasion Clothing.csv")

## Loading Sentence Transformers
encoder = SentenceTransformer("all-mpnet-base-v2")  ## Change the Model according to your needs

def create_embedding(data):
    """
    Create embeddings for product descriptions using a pre-trained SentenceTransformer model.

    Parameters:
    - data: DataFrame containing product data, including a 'Description' column.

    Returns:
    - vector_embeddings: Numpy array containing the generated embeddings.
    """
    desc = data["Description"]
    vector_embeddings = encoder.encode(desc)
    return vector_embeddings

def build_faiss(vector_embeddings):
    """
    Build a Faiss index for efficient similarity search.

    Parameters:
    - vector_embeddings: Numpy array containing the embeddings.

    Returns:
    - index: Faiss index built on the vector embeddings.
    """
    vector_dim = vector_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dim)
    faiss.normalize_L2(vector_embeddings)
    index.add(vector_embeddings)
    return index

def store_vector_store(index):
    """
    Store the Faiss index to a file.

    Parameters:
    - index: Faiss index to be stored.

    Prints a message indicating whether the index storage was successful or not.
    """
    try:
        index_file_path = "vector_store/myntra_embedding_vector_store.index"
        faiss.write_index(index, index_file_path)
        print(f"Index File Stored at {index_file_path}")
    except:
        print("Failed to Store the Index File")

# Create embeddings, build Faiss index, and store the index
vector_embeddings = create_embedding(data)
index = build_faiss(vector_embeddings)
store_vector_store(index)
