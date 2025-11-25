import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_split_data
from models import SentimentClassifier
from weak_labeling import WeakLabeler

def verify():
    try:
        print("Loading data...")
        splits = load_and_split_data()
        train_df = splits['train_100']
        unlabeled_df = splits['unlabeled_100'].head(50) # Use small subset for speed
        val_df = splits['val']
        test_df = splits['test']
        
        print("Generating weak labels...")
        labeler = WeakLabeler(model_name="all-mpnet-base-v2")
        knn = labeler.train_knn(train_df, n_neighbors=5)
        weak_labeled_df = labeler.predict(knn, unlabeled_df)
        
        print("Combining data...")
        combined_df = pd.concat([train_df, weak_labeled_df[['sentence', 'label']]]).reset_index(drop=True)
        print(f"Combined size: {len(combined_df)}")
        
        print("Training semi-supervised model (1 epoch)...")
        classifier = SentimentClassifier(model_name="distilbert-base-uncased", output_dir="models/verify_semi")
        classifier.train(combined_df, val_df, epochs=1, batch_size=8)
        
        print("Evaluating...")
        metrics = classifier.evaluate(test_df)
        print(f"Metrics: {metrics}")
        
        print("Semi-supervised verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
