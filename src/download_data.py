import argparse
import os
import sys

# Импортируем твой класс (предположим, он лежит в файле src/dataset_loader.py)
from data_loader import FiQADataset 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Куда скачивать данные")
    args = parser.parse_args()

    # DVC требует, чтобы мы явно указывали пути
    dataset_name = "fiqa"
    cache_path = args.output_dir
    
    # Инициализируем твой класс
    print(f"Downloading data to {cache_path}...")
    loader = FiQADataset(dataset_name=dataset_name, cache_dir=cache_path)
    
    # Этот метод внутри себя вызывает util.download_and_unzip
    # Beir создаст папку cache_path/fiqa
    loader.load_data()
    
    print("Download complete.")

if __name__ == "__main__":
    main()
