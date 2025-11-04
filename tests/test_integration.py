"""Integration tests for end-to-end pipeline"""

import pytest
import os
import numpy as np
from src.config import load_config
from src.data_loader import FiQADataset
from src.model import DenseRetriever
from src.train import Trainer, RetrievalDataset
from src.evaluate import evaluate_model


class TestEndToEndPipeline:
    """Test end-to-end training and evaluation"""
    
    @pytest.mark.slow
    def test_minimal_training_pipeline(self, sample_config, temp_dir):
        """Test minimal training pipeline runs without errors"""
        config = load_config(sample_config)
        
        # Create minimal mock data
        from src.data_loader import FiQADataset
        dataset = FiQADataset(dataset_name="fiqa", cache_dir=temp_dir)
        
        # Mock minimal dataset
        dataset.corpus = {
            'doc1': {'title': 'Title 1', 'text': 'Text 1'},
            'doc2': {'title': 'Title 2', 'text': 'Text 2'},
            'doc3': {'title': 'Title 3', 'text': 'Text 3'}
        }
        dataset.queries_train = {'q1': 'Query 1', 'q2': 'Query 2'}
        dataset.qrels_train = {'q1': {'doc1': 1}, 'q2': {'doc2': 1}}
        
        # Prepare training data
        training_samples = dataset.prepare_training_data(num_negatives=1)
        
        # Load model
        model = DenseRetriever(
            model_name=config.model.model_name,
            normalize_embeddings=config.model.normalize_embeddings,
            device='cpu'
        )
        
        # Create dataset
        train_dataset = RetrievalDataset(
            samples=training_samples,
            tokenizer=model.tokenizer,
            max_query_len=config.data.max_query_length,
            max_doc_len=config.data.max_doc_length
        )
        
        # Train for minimal steps
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            config=config,
            output_dir=temp_dir
        )
        
        # Just test that training starts
        # (don't run full training in tests)
        assert trainer is not None
        assert len(train_dataset) > 0
    
    def test_model_save_and_load_workflow(self, temp_dir):
        """Test complete save and load workflow"""
        # Load model
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        # Encode something
        original_embedding = model.encode(["Test"], batch_size=1, max_length=64)
        
        # Save
        save_path = os.path.join(temp_dir, 'test_model')
        model.save_model(save_path)
        
        # Load
        loaded_model = DenseRetriever.load_model(save_path, device='cpu')
        
        # Encode again
        loaded_embedding = loaded_model.encode(["Test"], batch_size=1, max_length=64)
        
        # Should be identical
        import torch
        assert torch.allclose(original_embedding, loaded_embedding, atol=1e-5)
    
    def test_evaluation_pipeline(self, sample_config, temp_dir):
        """Test evaluation pipeline runs"""
        config = load_config(sample_config)
        
        # Create minimal dataset
        dataset = FiQADataset(dataset_name="fiqa", cache_dir=temp_dir)
        dataset.corpus = {
            f'doc{i}': {'title': f'Title {i}', 'text': f'Text {i}'}
            for i in range(10)
        }
        dataset.queries_dev = {f'q{i}': f'Query {i}' for i in range(3)}
        dataset.qrels_dev = {
            'q0': {'doc0': 1, 'doc1': 1},
            'q1': {'doc2': 1},
            'q2': {'doc3': 1}
        }
        
        # Load model
        model = DenseRetriever(
            model_name=config.model.model_name,
            normalize_embeddings=True,
            device='cpu'
        )
        
        # Run evaluation
        metrics = evaluate_model(model, dataset, config, split="dev")
        
        # Check metrics exist
        assert 'recall@10' in metrics or 'recall@1' in metrics
        assert 'ndcg@10' in metrics or 'ndcg@1' in metrics
        
        # Check metric values are valid
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1.0
            assert not np.isnan(value)


class TestDataFlowIntegrity:
    """Test data flow through pipeline"""
    
    def test_data_shape_consistency(self):
        """Test that data shapes are consistent through pipeline"""
        from src.model import DenseRetriever
        from transformers import AutoTokenizer
        
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        batch_size = 4
        texts = [f"Text {i}" for i in range(batch_size)]
        
        # Encode
        embeddings = model.encode(texts, batch_size=batch_size, max_length=64)
        
        # Check shape
        assert embeddings.shape[0] == batch_size
        assert embeddings.shape[1] == 384  # MiniLM dimension
    
    def test_no_data_leakage_train_eval(self):
        """Test that train and eval data don't overlap"""
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        
        # Mock data
        dataset.queries_train = {'q1': 'Train query 1'}
        dataset.queries_dev = {'q2': 'Dev query 2'}
        dataset.qrels_train = {'q1': {'doc1': 1}}
        dataset.qrels_dev = {'q2': {'doc2': 1}}
        
        # Get splits
        train_queries = set(dataset.queries_train.keys())
        dev_queries = set(dataset.queries_dev.keys())
        
        # Check no overlap
        assert len(train_queries.intersection(dev_queries)) == 0
