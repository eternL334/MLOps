"""Pytest configuration and shared fixtures"""

import pytest
import torch
import tempfile
import os
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def sample_config(temp_dir):
    """Create sample configuration for testing"""
    config = {
        'experiment': {
            'name': 'test-experiment',
            'output_dir': os.path.join(temp_dir, 'outputs'),
            'log_dir': os.path.join(temp_dir, 'logs'),
            'random_seed': 42
        },
        'data': {
            'dataset_name': 'fiqa',
            'max_query_length': 64,
            'max_doc_length': 128,
            'cache_dir': os.path.join(temp_dir, 'cache')
        },
        'model': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'pooling': 'mean',
            'normalize_embeddings': True
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 4,
            'learning_rate': 5.0e-6,
            'warmup_steps': 10,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'fp16': False,
            'dataloader_num_workers': 0,
            'loss_type': 'contrastive',
            'temperature': 0.05,
            'num_negatives': 2
        },
        'evaluation': {
            'batch_size': 8,
            'top_k': [1, 3, 5, 10],
            'metrics': ['ndcg@10', 'recall@10', 'mrr@10'],
            'eval_steps': 50,
            'save_steps': 50
        },
        'index': {
            'type': 'faiss',
            'metric': 'cosine',
            'use_gpu': False
        }
    }
    
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing"""
    return {
        'doc1': {'title': 'Python Programming', 'text': 'Python is a high-level programming language.'},
        'doc2': {'title': 'Machine Learning', 'text': 'Machine learning is a subset of artificial intelligence.'},
        'doc3': {'title': 'Deep Learning', 'text': 'Deep learning uses neural networks with multiple layers.'},
        'doc4': {'title': 'Data Science', 'text': 'Data science involves statistics and programming.'},
        'doc5': {'title': 'Web Development', 'text': 'Web development includes frontend and backend.'}
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return {
        'q1': 'What is Python?',
        'q2': 'Explain machine learning',
        'q3': 'How does deep learning work?'
    }


@pytest.fixture
def sample_qrels():
    """Sample qrels (relevance judgments) for testing"""
    return {
        'q1': {'doc1': 1, 'doc4': 1},
        'q2': {'doc2': 1, 'doc3': 1},
        'q3': {'doc3': 1}
    }


@pytest.fixture
def sample_training_data():
    """Sample training data with query-doc pairs"""
    return [
        {
            'query': 'What is Python?',
            'positive': 'Python is a high-level programming language.',
            'negatives': ['Machine learning is a subset of artificial intelligence.'],
            'query_id': 'q1',
            'positive_id': 'doc1',
            'negative_ids': ['doc2']
        },
        {
            'query': 'Explain machine learning',
            'positive': 'Machine learning is a subset of artificial intelligence.',
            'negatives': ['Python is a high-level programming language.'],
            'query_id': 'q2',
            'positive_id': 'doc2',
            'negative_ids': ['doc1']
        }
    ]


@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device('cpu')  # Always use CPU for tests


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds before each test"""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
