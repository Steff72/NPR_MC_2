from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from typing import List, Tuple

class WeakLabeler:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, show_progress_bar=True)
    
    def train_knn(self, train_df: pd.DataFrame, n_neighbors: int = 5) -> KNeighborsClassifier:
        """
        Trains a k-NN classifier on the labeled training data embeddings.
        """
        embeddings = self.encode(train_df['sentence'].tolist())
        labels = train_df['label'].tolist()
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(embeddings, labels)
        return knn
    
    def predict(self, knn: KNeighborsClassifier, unlabeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts labels for unlabeled data using the trained k-NN.
        Returns the unlabeled dataframe with a new 'weak_label' column.
        """
        sentences = unlabeled_df['sentence'].tolist()
        embeddings = self.encode(sentences)
        
        weak_labels = knn.predict(embeddings)
        
        result_df = unlabeled_df.copy()
        result_df['label'] = weak_labels # Assign weak labels to 'label' column for compatibility
        result_df['is_weak'] = True
        
        return result_df

if __name__ == "__main__":
    print("WeakLabeler class defined.")
