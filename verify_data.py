import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_split_data

def verify():
    try:
        splits = load_and_split_data()
        print("Keys:", splits.keys())
        
        train_100 = splits['train_100']
        train_250 = splits['train_250']
        train_500 = splits['train_500']
        train_1000 = splits['train_1000']
        
        # Check if indices of smaller sets are contained in larger sets
        assert set(train_100.index).issubset(set(train_250.index)), "train_100 not subset of train_250"
        assert set(train_250.index).issubset(set(train_500.index)), "train_250 not subset of train_500"
        assert set(train_500.index).issubset(set(train_1000.index)), "train_500 not subset of train_1000"
        
        print("Hierarchical property verified!")
        print("Data loading successful.")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
