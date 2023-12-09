## Importing Packages
import streamlit as st
import pandas as pd
import numpy as np
from credit_app import credit_page
import faiss
from sentence_transformers import SentenceTransformer

st.title("EmbeddingBasedFashionSearch")

##Loading Samples of data
try:
    data = pd.read_csv("Dataset/Myntra Fasion Clothing.csv")
    data = data[:500]
    loaded_placeholader = st.empty()
    loaded_placeholader.success("Data Loaded")
    loaded_placeholader.empty()
except:
    loaded_placeholader.error("Data Loading Exception")
    print("Data Loading Exception")

## Encoder Loading
def encoder_search(text):
    """
    Function to encode the input text using SentenceTransformer.

    Parameters:
    - text (str): Input text to be encoded.

    Returns:
    - np.array: Encoded vector for the input text.
    """
    enocder = SentenceTransformer("all-mpnet-base-v2")
    search_vector = enocder.encode(text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    return _vector

## Results Loader
def get_results(vector,k,index):
    """
    Function to retrieve search results based on the input vector.

    Parameters:
    - vector (np.array): Input vector for search.
    - k (int): Number of top results to retrieve.
    - index: Faiss index for similarity search.

    Returns:
    - pd.DataFrame: DataFrame containing search results.
    """
    distances, ann = index.search(vector, k=k)
    sim_scores = 1 / (1 + distances[0])
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0],'Score': sim_scores})
    merge_results_data = pd.merge(results,data,left_on='ann',right_index=True)
    return merge_results_data
    

## Reading the vector Index
try:  
    index = faiss.read_index("vector_store/myntra_embedding_vector_store.index")
    print("Index File Loaded")
    loaded_placeholader = st.empty()
    loaded_placeholader.success("Index File Loaded")
    loaded_placeholader.empty()
except:
    print("Index not loaded properly")
    
## Creating Streamlit UI
with st.sidebar:
    st.subheader("Search Options")
    search_text = st.text_input("Go Here...")
    k = st.slider("Top Results", min_value=1, max_value=10, value=3)
    gender_filter = st.radio("Filter by Gender", ["All", "Men", "Women"])
    sbub = st.sidebar.button("Explore")
    
    cred_page = st.button("About This Project")

# Main section to display results
if sbub:
    # Perform the search and get results
    _vector = encoder_search(search_text)
    results = get_results(_vector, k, index)
    
    if gender_filter == "Men":
        results = results[results["category_by_Gender"] == "Men"]
    
    elif gender_filter == "Women":
        results = results[results["category_by_Gender"] == "Women"]

    # Display results in the main section
    st.markdown("## Style Finds for You")
    st.markdown("---")
    for sample in results.itertuples():
        st.write(f"**{sample.BrandName}**")
        st.write(f"Product Rating: {sample.Ratings}")
        st.write(f"Product URL: {sample.URL}")
        st.write(f"Product ID: {sample.Product_id}")
        st.write(f"Product Category: {sample.Category}")
        st.write(f"Gender: {sample.category_by_Gender}")
        st.write(f"Available Size: {sample.SizeOption}")
        st.write(f"Description: {sample.Description}")
        st.markdown("---")

if cred_page:
    credit_page()