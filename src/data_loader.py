import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

def load_and_split_data(
    dataset_name: str = "takala/financial_phrasebank",
    config_name: str = "sentences_allagree",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1, # Fraction of the remaining train data
    train_sizes: List[int] = [100, 250, 500, 1000]
) -> Dict[str, pd.DataFrame]:
    """
    Loads the Financial Phrasebank dataset and creates hierarchically nested splits.
    
    Args:
        dataset_name: Hugging Face dataset name.
        config_name: Dataset configuration (e.g., 'sentences_allagree').
        seed: Random seed for reproducibility.
        test_size: Fraction of data for the test set.
        val_size: Fraction of the *remaining* data for the validation set.
        train_sizes: List of sizes for the hierarchical training sets.
        
    Returns:
        A dictionary containing:
        - 'test': Test DataFrame
        - 'val': Validation DataFrame
        - 'train_{size}': Training DataFrame for each size in train_sizes
        - 'unlabeled_{size}': Unlabeled DataFrame for each size (remainder of train pool)
    """
    
    # Load dataset
    print(f"Loading dataset: {dataset_name} ({config_name})")
    dataset = load_dataset(dataset_name, config_name, split="train", trust_remote_code=True)
    df = dataset.to_pandas()
    
    # Initial Split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    
    # Split Train+Val into Train_Pool and Val
    train_pool_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=seed, stratify=train_val_df['label']
    )
    
    print(f"Total samples: {len(df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Training pool size: {len(train_pool_df)}")
    
    splits = {
        'test': test_df,
        'val': val_df
    }
    
    # Create Hierarchical Splits
    # We want train_100 subset of train_250 subset of train_500 ...
    # To do this, we can shuffle the pool once and take the first N samples.
    
    shuffled_pool = train_pool_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    for size in train_sizes:
        if size > len(shuffled_pool):
            raise ValueError(f"Requested training size {size} is larger than available pool {len(shuffled_pool)}")
            
        # Select the first 'size' samples as the labeled training set
        train_subset = shuffled_pool.iloc[:size].copy()
        
        # The rest are considered 'unlabeled' for this scenario
        unlabeled_subset = shuffled_pool.iloc[size:].copy()
        # Remove the label column from unlabeled set to simulate reality (optional, but good practice)
        # unlabeled_subset = unlabeled_subset.drop(columns=['label']) 
        
        splits[f'train_{size}'] = train_subset
        splits[f'unlabeled_{size}'] = unlabeled_subset
        
        print(f"Created split 'train_{size}': {len(train_subset)} samples")
        print(f"Created split 'unlabeled_{size}': {len(unlabeled_subset)} samples")
        
    return splits

if __name__ == "__main__":
    # Simple test
    data = load_and_split_data()
    print("Keys:", data.keys())
