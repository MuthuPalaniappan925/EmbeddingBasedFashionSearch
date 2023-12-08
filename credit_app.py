import streamlit as st

def credit_page():

    ## Display Project Overview
    st.header("Overview:")
    st.write("""
    The objective of the Embedding-Based Fashion Search project is to create an advanced search and recommendation system for fashion utilizing sophisticated techniques in natural language processing and similarity search. The system harnesses the power of embeddings produced by the SentenceTransformer model to depict product descriptions and utilizes the Faiss library to conduct efficient similarity searches.
    """)
    
    ## Technologies Used
    st.header("Technologies Used:")
    st.write("""
    - **Streamlit:** for building the interactive user interface.
    - **Pandas:** for data manipulation and loading.
    - **NumPy:** for numerical operations.
    - **Faiss:** for similarity search and efficient vector indexing.
    - **SentenceTransformer:** for generating embeddings from product descriptions.
    """)