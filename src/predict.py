import argparse
import os
import sys
import pandas as pd
import logging

from .model import DenseRetriever
from .evaluate import VectorIndex
from .data_loader import FiQADataset 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="outputs/final_model")
    parser.add_argument("--index_path", type=str, required=True, help="Путь к файлу .index")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Путь к корпусу (для текстов)")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    logger.info("Loading model...")
    model = DenseRetriever.load_model(args.model_path)

    logger.info(f"Loading index from {args.index_path}...")
    index = VectorIndex.load(args.index_path)

    logger.info("Loading corpus map...")
    dataset = FiQADataset(dataset_name="fiqa", cache_dir=os.path.dirname(args.raw_data_path))
    dataset.load_data()
    corpus_map = {did: d['title'] + ' ' + d['text'] for did, d in dataset.corpus.items()}

    df = pd.read_csv(args.input_path)
    queries = df['query'].tolist()
    query_ids = df['id'].tolist() if 'id' in df.columns else list(range(len(queries)))
    
    queries_prefix = ["query: " + q for q in queries]
    
    logger.info(f"Encoding {len(queries)} queries...")
    q_embeddings = model.encode(queries_prefix, batch_size=32, show_progress=True)
    
    logger.info("Searching...")
    scores, indices = index.search(q_embeddings, top_k=args.top_k)

    results = []
    for i, q_id in enumerate(query_ids):
        for rank, idx in enumerate(indices[i]):
            if idx == -1: continue 
            
            doc_id = index.get_doc_id(idx) 
            doc_text = corpus_map.get(doc_id, "")
            
            results.append({
                'query_id': q_id,
                'rank': rank + 1,
                'doc_id': doc_id,
                'score': float(scores[i][rank]),
                'text': doc_text[:500] 
            })
            
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_path, index=False)
    logger.info(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
