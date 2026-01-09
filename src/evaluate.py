"""Evaluation with vector index and metrics"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
import faiss
from collections import defaultdict

import argparse
import json
import os
import sys

from .config import load_config
from .utils import setup_logging, set_seed
from .data_loader import FiQADataset
from .model import DenseRetriever

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-based vector index for retrieval"""
    
    def __init__(self, dimension: int, metric: str = "cosine", use_gpu: bool = False):
        """
        Initialize vector index
        
        Args:
            dimension: Embedding dimension
            metric: Similarity metric (cosine, dot, l2)
            use_gpu: Use GPU for index
        """
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu
        
        # Create index
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        elif metric == "dot":
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS index")
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
        
        self.doc_ids = []
        logger.info(f"Vector index initialized (metric={metric}, gpu={use_gpu})")
    
    def add_documents(self, embeddings: np.ndarray, doc_ids: List[str]):
        """
        Add documents to index
        
        Args:
            embeddings: Document embeddings [num_docs, dim]
            doc_ids: Document IDs
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        embeddings = embeddings.astype('float32')
        
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        
        logger.info(f"Added {len(doc_ids)} documents to index (total: {len(self.doc_ids)})")
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar documents
        
        Args:
            query_embeddings: Query embeddings [num_queries, dim]
            top_k: Number of results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()
        
        query_embeddings = query_embeddings.astype('float32')
        
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices
    
    def get_doc_id(self, idx: int) -> str:
        """Get document ID by index"""
        return self.doc_ids[idx]


class RetrievalEvaluator:
    """Evaluator for retrieval metrics"""
    
    def __init__(self, top_k_values: List[int] = [1, 3, 5, 10, 20, 100]):
        """
        Initialize evaluator
        
        Args:
            top_k_values: List of k values for metrics
        """
        self.top_k_values = top_k_values
    
    def evaluate(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        query_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics
        
        Args:
            qrels: Ground truth relevance {query_id: {doc_id: relevance}}
            results: Retrieved results {query_id: {doc_id: score}}
            query_ids: List of query IDs to evaluate
            
        Returns:
            Dictionary of metrics
        """
        metrics = defaultdict(list)
        
        for query_id in query_ids:
            if query_id not in qrels or query_id not in results:
                continue
            
            # Get relevant documents
            relevant_docs = set(qrels[query_id].keys())
            
            if not relevant_docs:
                continue
            
            # Get retrieved documents (sorted by score)
            retrieved = sorted(
                results[query_id].items(),
                key=lambda x: x[1],
                reverse=True
            )
            retrieved_ids = [doc_id for doc_id, _ in retrieved]
            
            # Calculate metrics at different k
            for k in self.top_k_values:
                retrieved_at_k = retrieved_ids[:k]
                
                # Recall@k
                recall = len(set(retrieved_at_k) & relevant_docs) / len(relevant_docs)
                metrics[f'recall@{k}'].append(recall)
                
                # Precision@k
                if len(retrieved_at_k) > 0:
                    precision = len(set(retrieved_at_k) & relevant_docs) / len(retrieved_at_k)
                    metrics[f'precision@{k}'].append(precision)
                
                # NDCG@k
                ndcg = self.compute_ndcg(retrieved_at_k, relevant_docs, k)
                metrics[f'ndcg@{k}'].append(ndcg)
            
            # MRR (Mean Reciprocal Rank)
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_docs:
                    metrics['mrr'].append(1.0 / i)
                    break
            else:
                metrics['mrr'].append(0.0)
        
        # Average metrics
        avg_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in metrics.items()
        }
        
        return avg_metrics
    
    def compute_ndcg(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Compute NDCG@k"""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model,
    dataset,
    config,
    split: str = "dev"
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set
    
    Args:
        model: Dense retriever model
        dataset: FiQA dataset
        config: Configuration
        split: 'dev' or 'test'
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating model on {split} set...")
    
    # Get corpus
    doc_texts, doc_ids = dataset.get_corpus_list()
    doc_texts = ["passage: " + text for text in doc_texts]  # Add E5 prefix
    
    # Encode documents
    logger.info("Encoding documents...")
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=config.evaluation.batch_size,
        max_length=config.data.max_doc_length,
        show_progress=True
    )
    
    # Build index
    logger.info("Building vector index...")
    index = VectorIndex(
        dimension=config.model.embedding_dim,
        metric=config.index.metric,
        use_gpu=config.index.use_gpu
    )
    index.add_documents(doc_embeddings, doc_ids)
    
    # Get validation data
    queries, qrels = dataset.get_validation_data(split)
    query_ids = list(queries.keys())
    query_texts = ["query: " + queries[qid] for qid in query_ids]  # Add E5 prefix
    
    # Encode queries
    logger.info(f"Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=config.evaluation.batch_size,
        max_length=config.data.max_query_length,
        show_progress=True
    )
    
    # Search
    logger.info("Searching...")
    max_k = max(config.evaluation.top_k)
    scores, indices = index.search(query_embeddings, top_k=max_k)
    
    # Build results dictionary
    results = {}
    for i, query_id in enumerate(query_ids):
        results[query_id] = {}
        for j, idx in enumerate(indices[i]):
            doc_id = index.get_doc_id(idx)
            results[query_id][doc_id] = float(scores[i][j])
    
    # Compute metrics
    logger.info("Computing metrics...")
    evaluator = RetrievalEvaluator(top_k_values=config.evaluation.top_k)
    metrics = evaluator.evaluate(qrels, results, query_ids)
    
    # Log metrics
    logger.info(f"Evaluation results on {split} set:")
    for metric_name, value in sorted(metrics.items()):
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Dense Retriever (DVC Stage)")
    
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигу")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к папке с моделью (output от train)")
    parser.add_argument("--raw_data_path", type=str, required=True, help="Путь папки с сырыми данными (из dvc add)")
    parser.add_argument("--metrics_file", type=str, required=True, help="Куда сохранить JSON с метриками")
    
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config.experiment.log_dir, log_file="evaluate_dvc.log")
    set_seed(config.experiment.random_seed)

    logger.info("DVC Evaluation Stage Started")

    logger.info(f"Loading data from {args.raw_data_path}")
    dataset = FiQADataset(
        dataset_name=config.data.dataset_name,
        cache_dir=os.path.dirname(args.raw_data_path) 
    )
    dataset.load_data()

    logger.info(f"Loading model from {args.model_path}")
    model = DenseRetriever.load_model(model_path=args.model_path)

    metrics = {}
    
    logger.info("Evaluating on DEV split")
    metrics['dev'] = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        split="dev"
    )

    logger.info("Evaluating on TEST split")
    metrics['test'] = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        split="test"
    )

    logger.info(f"Saving metrics to {args.metrics_file}")
    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation completed successfully.")