import argparse
import os
import json
import logging
from .data_loader import FiQADataset 

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True, help="Путь к сырой папке")
    parser.add_argument("--output_file", type=str, required=True, help="Куда сохранить jsonl")
    args = parser.parse_args()

    
    loader = FiQADataset(dataset_name="fiqa", cache_dir=os.path.dirname(args.raw_data_path))
    loader.load_data() 

    train_samples = loader.prepare_training_data(num_negatives=1)
    
    print(f"Saving prepared data to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            json.dump(sample, f)
            f.write('\n')

if __name__ == "__main__":
    main()
