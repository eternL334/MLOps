import pickle
import os
import sys

from .data_loader import FiQADataset

def main():
    raw_path = "data/fiqa" 
    print("Loading dataset...")
    ds = FiQADataset("fiqa", os.path.dirname(raw_path))
    ds.load_data()
    
    print("Building map...")
    corpus_map = {did: d['title'] + ' ' + d['text'] for did, d in ds.corpus.items()}
    
    output_path = "data/corpus_map.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(corpus_map, f)
    
    print(f"Saved {len(corpus_map)} docs to {output_path}")

if __name__ == "__main__":
    main()
