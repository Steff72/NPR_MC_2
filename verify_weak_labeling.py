import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_split_data
from weak_labeling import WeakLabeler
from sklearn.metrics import accuracy_score

def verify():
    try:
        print("Loading data...")
        splits = load_and_split_data()
        train_df = splits['train_100']
        unlabeled_df = splits['unlabeled_100'].head(50) # Use a small subset for speed
        
        print("Initializing WeakLabeler...")
        labeler = WeakLabeler(model_name="all-mpnet-base-v2")
        
        print("Training k-NN...")
        knn = labeler.train_knn(train_df, n_neighbors=5)
        
        print("Predicting weak labels...")
        weak_labeled_df = labeler.predict(knn, unlabeled_df)
        
        print("Evaluating...")
        true_labels = unlabeled_df['label']
        predicted_labels = weak_labeled_df['label']
        
        acc = accuracy_score(true_labels, predicted_labels)
        print(f"Weak Label Accuracy (on 50 samples): {acc:.4f}")
        
        print("Weak labeling verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
