"""Tests for evaluation components"""

import pytest
import numpy as np
import torch
from src.evaluate import VectorIndex, RetrievalEvaluator


class TestVectorIndex:
    """Test VectorIndex class"""
    
    def test_index_initialization(self):
        """Test index initialization"""
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        assert index.dimension == 128
        assert index.metric == "cosine"
        assert index.use_gpu is False
        assert len(index.doc_ids) == 0
    
    def test_index_add_documents(self):
        """Test adding documents to index"""
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        embeddings = np.random.randn(10, 128).astype('float32')
        doc_ids = [f"doc{i}" for i in range(10)]
        
        index.add_documents(embeddings, doc_ids)
        
        assert len(index.doc_ids) == 10
        assert index.doc_ids == doc_ids
    
    def test_index_search(self):
        """Test search functionality"""
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        # Add documents
        doc_embeddings = np.random.randn(100, 128).astype('float32')
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_ids = [f"doc{i}" for i in range(100)]
        
        index.add_documents(doc_embeddings, doc_ids)
        
        # Search
        query_embeddings = np.random.randn(5, 128).astype('float32')
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        scores, indices = index.search(query_embeddings, top_k=10)
        
        # Check output shapes
        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)
        
        # Check value ranges
        assert indices.min() >= 0
        assert indices.max() < 100
    
    def test_index_search_returns_most_similar(self):
        """Test that search returns most similar documents"""
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        # Create documents
        doc_embeddings = np.random.randn(10, 128).astype('float32')
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_ids = [f"doc{i}" for i in range(10)]
        
        index.add_documents(doc_embeddings, doc_ids)
        
        # Query with first document (should return itself as top result)
        query_embeddings = doc_embeddings[0:1]
        
        scores, indices = index.search(query_embeddings, top_k=5)
        
        # First result should be the document itself
        assert indices[0, 0] == 0
        assert scores[0, 0] > 0.99  # Very high similarity
    
    def test_index_get_doc_id(self):
        """Test retrieving document ID by index"""
        index = VectorIndex(dimension=128, metric="cosine", use_gpu=False)
        
        embeddings = np.random.randn(5, 128).astype('float32')
        doc_ids = [f"doc{i}" for i in range(5)]
        
        index.add_documents(embeddings, doc_ids)
        
        assert index.get_doc_id(0) == "doc0"
        assert index.get_doc_id(2) == "doc2"
        assert index.get_doc_id(4) == "doc4"


class TestRetrievalEvaluator:
    """Test RetrievalEvaluator class"""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        assert evaluator.top_k_values == [1, 5, 10]
    
    def test_evaluate_perfect_retrieval(self):
        """Test metrics with perfect retrieval"""
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        # Perfect retrieval: all relevant docs ranked first
        qrels = {
            'q1': {'doc1': 1, 'doc2': 1}
        }
        
        results = {
            'q1': {
                'doc1': 1.0,
                'doc2': 0.9,
                'doc3': 0.5,
                'doc4': 0.3
            }
        }
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        # Perfect recall and NDCG at all k
        assert metrics['recall@10'] == 1.0
        assert metrics['recall@5'] == 1.0
        assert metrics['ndcg@10'] == 1.0
        assert metrics['mrr'] == 1.0
    
    def test_evaluate_partial_retrieval(self):
        """Test metrics with partial retrieval"""
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        qrels = {
            'q1': {'doc1': 1, 'doc2': 1, 'doc3': 1}
        }
        
        results = {
            'q1': {
                'doc1': 1.0,
                'doc4': 0.9,
                'doc2': 0.8,
                'doc5': 0.7
            }
        }
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        # Should find 2 out of 3 relevant docs
        assert 0 < metrics['recall@10'] < 1.0
        assert metrics['recall@10'] == pytest.approx(2.0/3.0, rel=1e-2)
    
    def test_evaluate_no_relevant_retrieved(self):
        """Test metrics when no relevant docs retrieved"""
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        qrels = {
            'q1': {'doc1': 1, 'doc2': 1}
        }
        
        results = {
            'q1': {
                'doc3': 1.0,
                'doc4': 0.9
            }
        }
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        # All metrics should be 0
        assert metrics['recall@10'] == 0.0
        assert metrics['ndcg@10'] == 0.0
        assert metrics['mrr'] == 0.0
    
    def test_compute_ndcg(self):
        """Test NDCG computation"""
        evaluator = RetrievalEvaluator(top_k_values=[5])
        
        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = {'doc1', 'doc3'}
        
        ndcg = evaluator.compute_ndcg(retrieved, relevant, k=3)
        
        # NDCG should be between 0 and 1
        assert 0 <= ndcg <= 1.0
    
    def test_evaluate_multiple_queries(self):
        """Test evaluation with multiple queries"""
        evaluator = RetrievalEvaluator(top_k_values=[5, 10])
        
        qrels = {
            'q1': {'doc1': 1},
            'q2': {'doc2': 1},
            'q3': {'doc3': 1}
        }
        
        results = {
            'q1': {'doc1': 1.0, 'doc4': 0.5},
            'q2': {'doc2': 1.0, 'doc5': 0.5},
            'q3': {'doc6': 1.0, 'doc3': 0.5}  # Relevant not at top
        }
        
        metrics = evaluator.evaluate(qrels, results, ['q1', 'q2', 'q3'])
        
        # Should average across queries
        assert 'recall@5' in metrics
        assert 'ndcg@10' in metrics
        assert 0 <= metrics['recall@5'] <= 1.0


class TestMetricsValidation:
    """Test metrics validation"""
    
    def test_metrics_value_ranges(self):
        """Test that all metrics are in valid ranges"""
        evaluator = RetrievalEvaluator(top_k_values=[1, 5, 10])
        
        qrels = {'q1': {'doc1': 1, 'doc2': 1}}
        results = {'q1': {'doc1': 1.0, 'doc3': 0.5}}
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1.0, f"{metric_name} out of range: {value}"
            assert not np.isnan(value), f"{metric_name} is NaN"
            assert not np.isinf(value), f"{metric_name} is Inf"
    
    def test_metrics_types(self):
        """Test that metrics have correct types"""
        evaluator = RetrievalEvaluator(top_k_values=[5, 10])
        
        qrels = {'q1': {'doc1': 1}}
        results = {'q1': {'doc1': 1.0}}
        
        metrics = evaluator.evaluate(qrels, results, ['q1'])
        
        for metric_name, value in metrics.items():
            assert isinstance(value, (float, np.floating)), \
                f"{metric_name} is not float: {type(value)}"
