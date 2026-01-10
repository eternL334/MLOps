import argparse
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.getcwd())

from src.model import DenseRetriever

def main():
    parser = argparse.ArgumentParser(description="Offline Inference inside Docker")
    parser.add_argument("--input_path", type=str, required=True, help="input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="output CSV file")
    parser.add_argument("--model_path", type=str, default="outputs/final_model", help="path to model")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"Starting inference...")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model_path}")

    try:
        df = pd.read_csv(args.input_path)
        if 'query' not in df.columns:
            raise ValueError("input CSV must contain column 'query'")
        data = df['query'].tolist()
        ids = df['id'].tolist() if 'id' in df.columns else list(range(len(data)))
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    data_with_prefix = ["query: " + text for text in data]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    try:
        model = DenseRetriever.load_model(args.model_path)
        model.model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Processing {len(data)} queries...")
    all_embeddings = []
    
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data_with_prefix), batch_size)):
            batch_texts = data_with_prefix[i : i + batch_size]
            
            inputs = model.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = model.mean_pooling(outputs, attention_mask)
            
            if model.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.extend(embeddings.cpu().numpy().tolist())

    print("Saving results...")
    output_df = pd.DataFrame({
        'id': ids,
        'query': data,
        'embedding': all_embeddings
    })
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    output_df.to_csv(args.output_path, index=False)
    print(f"Done! Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
