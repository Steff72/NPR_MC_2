import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_split_data
from models import SentimentClassifier

def verify():
    try:
        print("Loading data...")
        splits = load_and_split_data()
        train_df = splits['train_100']
        val_df = splits['val']
        test_df = splits['test']
        
        print("Initializing model...")
        # Use a small model or just run for 1 epoch
        classifier = SentimentClassifier(model_name="distilbert-base-uncased", output_dir="models/verify_baseline")
        
        print("Starting training (1 epoch)...")
        classifier.train(train_df, val_df, epochs=1, batch_size=8)
        
        print("Evaluating...")
        metrics = classifier.evaluate(test_df)
        print(f"Metrics: {metrics}")
        
        print("Baseline verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
