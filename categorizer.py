import pandas as pd
import numpy as np
import os
from data_connector import preprocess_text
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F 

#Configuration
RAW_DATA_FILE = 'cleaned_ticket_data.csv'
CLEANED_DATA_FILE = 'knowledge_base_with_embeddings.csv'
EMBEDDINGS_FILE = 'kb_embeddings.npy'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


# Load the Sentence Transformer model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

def get_embedding(text):
    """Generates an embedding for a piece of text."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze().numpy().astype(np.float64) 

def load_and_process_data():
    """Loads raw data, preprocesses, generates embeddings, and saves the new KB."""
    print("Loading raw data...")
    try:
        articles_df = pd.read_csv(RAW_DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {RAW_DATA_FILE}. Please ensure it exists.")
        return None
    except Exception as e:
        print(f"ERROR loading raw data: {e}. If the column names 'Subject' or 'Full_Ticket_Text' are wrong, rename them in your CSV.")
        return None

    # Preprocess text and generate the combined text column
    print("Preprocessing text and combining columns...")
    try:
        articles_df['Preprocessed_Text'] = articles_df.apply(
            lambda row: preprocess_text(str(row['Subject']) + " " + str(row['Full_Ticket_Text'])), 
            axis=1
        )
    except KeyError as e:
        print(f"\nFATAL ERROR: Column {e} not found in your CSV file. Please check the header names in {RAW_DATA_FILE} and ensure they exactly match 'Subject' and 'Full_Ticket_Text'.")
        return None


    # Generate embeddings for all articles
    print(f"Generating embeddings for {len(articles_df)} articles...")
    
    embeddings_list = [get_embedding(text) for text in articles_df['Preprocessed_Text']]
    
    # Store embeddings as a NumPy array
    embeddings_array = np.array(embeddings_list)
    np.save(EMBEDDINGS_FILE, embeddings_array)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")


    # Save the cleaned knowledge base with embeddings
    articles_df.to_csv(CLEANED_DATA_FILE, sep='|', index=False)
    print(f"Cleaned Knowledge Base saved to {CLEANED_DATA_FILE} using '|' separator.")
    
    return articles_df

if __name__ == "__main__":
    kb_df = load_and_process_data()
    if kb_df is not None:
        print("Knowledge Base generation complete!")