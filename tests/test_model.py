"""Tests for model loading and encoding"""

import pytest
import torch
import numpy as np
from src.model import DenseRetriever


class TestDenseRetriever:
    """Test DenseRetriever class"""
    
    def test_model_initialization(self, device):
        """Test model initialization"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.normalize_embeddings is True
        assert model.device == torch.device('cpu')
    
    def test_model_encode_output_shape(self, device):
        """Test that encoding produces correct output shape"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(texts, batch_size=2, max_length=64)
        
        # Check output type and shape
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # MiniLM-L6-v2 dimension
    
    def test_model_encode_normalization(self, device):
        """Test that embeddings are normalized when requested"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = ["Test sentence"]
        embeddings = model.encode(texts, batch_size=1, max_length=64)
        
        # Check L2 norm is approximately 1.0
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_model_encode_without_normalization(self, device):
        """Test encoding without normalization"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=False,
            device='cpu'
        )
        
        texts = ["Test sentence"]
        embeddings = model.encode(texts, batch_size=1, max_length=64)
        
        # Norm should not be 1.0
        norms = torch.norm(embeddings, p=2, dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_model_encode_batch_processing(self, device):
        """Test that batch processing works correctly"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = [f"Test sentence {i}" for i in range(10)]
        
        # Encode with different batch sizes
        embeddings_bs2 = model.encode(texts, batch_size=2, max_length=64)
        embeddings_bs5 = model.encode(texts, batch_size=5, max_length=64)
        
        # Results should be the same regardless of batch size
        assert torch.allclose(embeddings_bs2, embeddings_bs5, atol=1e-5)
    
    def test_model_encode_empty_text(self, device):
        """Test encoding with empty text"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = [""]
        embeddings = model.encode(texts, batch_size=1, max_length=64)
        
        # Should still produce valid embeddings
        assert embeddings.shape[0] == 1
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_mean_pooling(self, device):
        """Test mean pooling function"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        # Create dummy outputs
        batch_size = 2
        seq_length = 10
        hidden_dim = 384
        
        token_embeddings = torch.randn(batch_size, seq_length, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_length)
        
        model_output = (token_embeddings,)
        pooled = model.mean_pooling(model_output, attention_mask)
        
        # Check output shape
        assert pooled.shape == (batch_size, hidden_dim)
        
        # Check no NaN or Inf
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()
    
    def test_save_and_load_model(self, temp_dir):
        """Test model saving and loading"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        # Save model
        save_path = f"{temp_dir}/test_model"
        model.save_model(save_path)
        
        # Check files exist
        import os
        assert os.path.exists(save_path)
        assert os.path.exists(f"{save_path}/config.json")
        
        # Check for either pytorch_model.bin or model.safetensors
        has_pytorch = os.path.exists(f"{save_path}/pytorch_model.bin")
        has_safetensors = os.path.exists(f"{save_path}/model.safetensors")
        assert has_pytorch or has_safetensors, "Model file not found (neither .bin nor .safetensors)"
        
        # Load model
        loaded_model = DenseRetriever.load_model(save_path, device='cpu')
        
        # Test that loaded model produces same embeddings
        texts = ["Test sentence"]
        original_emb = model.encode(texts, batch_size=1, max_length=64)
        loaded_emb = loaded_model.encode(texts, batch_size=1, max_length=64)
        
        assert torch.allclose(original_emb, loaded_emb, atol=1e-5)



class TestModelOutputValidation:
    """Test model output validation"""
    
    def test_embedding_value_ranges(self, device):
        """Test that embeddings have reasonable value ranges"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        texts = ["Test sentence", "Another sentence"]
        embeddings = model.encode(texts, batch_size=2, max_length=64)
        
        # Normalized embeddings should be in [-1, 1]
        assert (embeddings >= -1.0).all()
        assert (embeddings <= 1.0).all()
        
        # No NaN or Inf
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_embedding_determinism(self, device):
        """Test that encoding is deterministic"""
        model = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            normalize_embeddings=True,
            device='cpu'
        )
        
        model.model.eval()
        texts = ["Test sentence"]
        
        # Encode twice
        emb1 = model.encode(texts, batch_size=1, max_length=64)
        emb2 = model.encode(texts, batch_size=1, max_length=64)
        
        # Should be identical
        assert torch.allclose(emb1, emb2, atol=1e-6)
