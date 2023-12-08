# Embedding-Based Fashion Search Documentation

## Overview
The Embedding-Based Fashion Search project is designed to provide an advanced search and recommendation system for fashion using state-of-the-art natural language processing techniques and similarity search. The system leverages SentenceTransformer models to generate embeddings from product descriptions and utilizes the Faiss library for efficient vector indexing and similarity searches.

## Table of Contents
1. [Installation](#installation)
2. [Data Ingestion](#data-ingestion)
3. [Streamlit Application](#streamlit-application)
4. [Credit Page](#credit-page)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Technologies Used](#technologies-used)


## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/MuthuPalaniappan925/EmbeddingBasedFashionSearch.git
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Data Ingestion
Run the `data_ingestion.py` script to load the fashion dataset, create embeddings for product descriptions, build a Faiss index, and store the index for later use.

```bash
python data_ingestion.py
```

## Streamlit Application
Run the Streamlit application `app.py` to interact with the fashion search and recommendation system.

```bash
streamlit run app.py
```

The Streamlit UI allows users to enter search queries, specify the number of top results to retrieve, and filter results based on gender.

## Credit Page
The `credit_page.py` module contains information about the project overview and the technologies used. To view the credits, click the "About This Project" button in the Streamlit application.

## Project Structure
- `app.py`: Streamlit application for fashion search.
- `credit_page.py`: Module containing project overview and credits.
- `data_ingestion.py`: Script for loading data, creating embeddings, building Faiss index, and storing the index.
- `Dataset/`: Folder containing the fashion dataset.
- `vector_store/`: Folder to store the Faiss index file.

## Usage
1. Run `data_ingestion.py` to prepare the Faiss index.
2. Run `app.py` to launch the Streamlit application.
3. Enter search queries and explore fashion recommendations.

## Technologies Used
- **Streamlit:** Interactive web application development.
- **Pandas:** Data manipulation and loading.
- **NumPy:** Numerical operations.
- **Faiss:** Similarity search and efficient vector indexing.
- **SentenceTransformer:** Generating embeddings from product descriptions.

## Demo

Will update soon...
