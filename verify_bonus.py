import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_split_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify():
    try:
        print("Checking imports...")
        import umap
        import openai
        print("Imports successful.")
        
        print("Loading data...")
        splits = load_and_split_data()
        test_df = splits['test'].head(20)
        
        print("Generating dummy embeddings...")
        # Dummy embeddings for speed
        embeddings = np.random.rand(20, 768)
        
        print("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        print(f"UMAP output shape: {embedding_2d.shape}")
        
        print("Checking OpenAI API Key...")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("API Key found.")
        else:
            print("API Key not found (expected).")
            
        print("Bonus verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
