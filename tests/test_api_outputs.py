"""Tests for API output format and predictions"""

import pytest
import numpy as np
import torch
from typing import Dict, List


class TestAPIOutputFormat:
    """Test output format for potential API"""
    
    def test_embedding_output_format(self):
        """Test that embeddings have correct format for API"""
        from src.model import DenseRetriever
        
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = ["Test sentence"]
        embeddings = model.encode(texts, batch_size=1, max_length=64)
        
        # Convert to API format (list of lists)
        api_embeddings = embeddings.tolist()
        
        assert isinstance(api_embeddings, list)
        assert isinstance(api_embeddings[0], list)
        assert all(isinstance(x, float) for x in api_embeddings[0])
    
    def test_search_results_format(self):
        """Test that search results have correct format for API"""
        from src.evaluate import VectorIndex
        
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        # Add documents
        doc_embeddings = np.random.randn(10, 128).astype('float32')
        doc_ids = [f"doc_{i}" for i in range(10)]
        index.add_documents(doc_embeddings, doc_ids)
        
        # Search
        query_embeddings = np.random.randn(1, 128).astype('float32')
        scores, indices = index.search(query_embeddings, top_k=5)
        
        # Convert to API format
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                'doc_id': index.get_doc_id(int(idx)),
                'score': float(score)
            })
        
        # Validate format
        assert isinstance(results, list)
        assert len(results) == 5
        
        for result in results:
            assert 'doc_id' in result
            assert 'score' in result
            assert isinstance(result['doc_id'], str)
            assert isinstance(result['score'], float)
    
    def test_ranked_results_descending_order(self):
        """Test that results are ranked in descending order of score"""
        from src.evaluate import VectorIndex
        
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        doc_embeddings = np.random.randn(20, 128).astype('float32')
        doc_ids = [f"doc_{i}" for i in range(20)]
        index.add_documents(doc_embeddings, doc_ids)
        
        query_embeddings = np.random.randn(1, 128).astype('float32')
        scores, indices = index.search(query_embeddings, top_k=10)
        
        # Convert to API format with ranking
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            results.append({
                'rank': rank,
                'doc_id': index.get_doc_id(int(idx)),
                'score': float(score)
            })
        
        # Check descending order
        scores_list = [r['score'] for r in results]
        assert scores_list == sorted(scores_list, reverse=True)
        
        # Check ranks
        ranks = [r['rank'] for r in results]
        assert ranks == list(range(1, 11))
    
    def test_metrics_output_format(self):
        """Test that evaluation metrics have correct format for API"""
        from src.evaluate import RetrievalEvaluator
        
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        qrels = {'q1': {'doc1': 1, 'doc2': 1}}
        results = {'q1': {'doc1': 1.0, 'doc2': 0.9, 'doc3': 0.5}}
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        # Convert to API format
        api_metrics = {
            'metrics': {k: float(v) for k, v in metrics.items()},
            'num_queries': 1
        }
        
        assert isinstance(api_metrics, dict)
        assert 'metrics' in api_metrics
        assert 'num_queries' in api_metrics
        
        for metric_name, value in api_metrics['metrics'].items():
            assert isinstance(metric_name, str)
            assert isinstance(value, float)


class TestPredictionValidation:
    """Test prediction validation and conversion"""
    
    def test_similarity_score_range(self):
        """Test that similarity scores are in valid range"""
        from src.model import DenseRetriever
        
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        query = ["What is Python?"]
        docs = ["Python is a programming language.", "Java is also popular."]
        
        query_emb = model.encode(query, batch_size=1, max_length=64)
        doc_emb = model.encode(docs, batch_size=2, max_length=64)
        
        # Compute cosine similarity
        similarities = torch.mm(query_emb, doc_emb.T)
        
        # For normalized embeddings, cosine similarity is in [-1, 1]
        assert (similarities >= -1.0).all()
        assert (similarities <= 1.0).all()
    
    def test_raw_scores_to_probabilities(self):
        """Test conversion of raw scores to probabilities"""
        # Raw similarity scores
        raw_scores = np.array([0.8, 0.6, 0.4, 0.2])
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(raw_scores)
        probabilities = exp_scores / exp_scores.sum()
        
        # Validate probabilities
        assert probabilities.sum() == pytest.approx(1.0, rel=1e-5)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
        assert probabilities[0] > probabilities[1]  # Higher score → higher probability
    
    def test_relevance_classification(self):
        """Test binary relevance classification from scores"""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        threshold = 0.6
        
        # Classify as relevant/not relevant
        relevance = (scores >= threshold).astype(int)
        
        assert relevance[0] == 1  # Relevant
        assert relevance[1] == 1  # Relevant
        assert relevance[2] == 0  # Not relevant
        assert relevance[3] == 0  # Not relevant
    
    def test_top_k_filtering(self):
        """Test filtering to top-k results"""
        results = {
            'doc1': 0.95,
            'doc2': 0.85,
            'doc3': 0.75,
            'doc4': 0.65,
            'doc5': 0.55
        }
        
        k = 3
        
        # Get top-k
        top_k_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:k])
        
        assert len(top_k_results) == k
        assert 'doc1' in top_k_results
        assert 'doc2' in top_k_results
        assert 'doc3' in top_k_results
        assert 'doc4' not in top_k_results
