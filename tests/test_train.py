"""Tests for training components"""

import pytest
import torch
from src.train import ContrastiveLoss, RetrievalDataset, collate_fn
from src.model import DenseRetriever


class TestContrastiveLoss:
    """Test contrastive loss function"""
    
    def test_loss_initialization(self):
        """Test loss function initialization"""
        loss_fn = ContrastiveLoss(temperature=0.05)
        assert loss_fn.temperature == 0.05
    
    def test_loss_forward_shape(self):
        """Test loss computation with correct shapes"""
        loss_fn = ContrastiveLoss(temperature=0.05)
        
        batch_size = 4
        dim = 128
        num_neg = 2
        
        query_emb = torch.randn(batch_size, dim)
        pos_emb = torch.randn(batch_size, dim)
        neg_emb = torch.randn(batch_size, num_neg, dim)
        
        # Normalize
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=2)
        
        loss = loss_fn(query_emb, pos_emb, neg_emb)
        
        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_value_range(self):
        """Test that loss values are in reasonable range"""
        loss_fn = ContrastiveLoss(temperature=0.05)
        
        batch_size = 4
        dim = 128
        num_neg = 2
        
        query_emb = torch.randn(batch_size, dim)
        pos_emb = torch.randn(batch_size, dim)
        neg_emb = torch.randn(batch_size, num_neg, dim)
        
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=2)
        
        loss = loss_fn(query_emb, pos_emb, neg_emb)
        
        # Loss should be reasonable (not too large)
        assert loss.item() < 100
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_loss_decreases_with_similar_positives(self):
        """Test that loss is lower when positive is more similar"""
        loss_fn = ContrastiveLoss(temperature=0.05)
        
        batch_size = 2
        dim = 128
        
        query_emb = torch.randn(batch_size, dim)
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        
        # Case 1: Positive very similar to query
        pos_emb_similar = query_emb + torch.randn(batch_size, dim) * 0.01
        pos_emb_similar = torch.nn.functional.normalize(pos_emb_similar, p=2, dim=1)
        
        # Case 2: Positive less similar to query
        pos_emb_dissimilar = torch.randn(batch_size, dim)
        pos_emb_dissimilar = torch.nn.functional.normalize(pos_emb_dissimilar, p=2, dim=1)
        
        neg_emb = torch.randn(batch_size, 1, dim)
        neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=2)
        
        loss_similar = loss_fn(query_emb, pos_emb_similar, neg_emb)
        loss_dissimilar = loss_fn(query_emb, pos_emb_dissimilar, neg_emb)
        
        # Loss should be lower for similar positives
        assert loss_similar.item() < loss_dissimilar.item()


class TestRetrievalDataset:
    """Test RetrievalDataset class"""
    
    def test_dataset_initialization(self, sample_training_data):
        """Test dataset initialization"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=64,
            max_doc_len=128
        )
        
        assert len(dataset) == len(sample_training_data)
    
    def test_dataset_getitem(self, sample_training_data):
        """Test dataset __getitem__"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=64,
            max_doc_len=128
        )
        
        item = dataset[0]
        
        # Check structure
        assert 'query' in item
        assert 'positive' in item
        assert 'negatives' in item
        
        # Check E5 prefixes are added
        assert item['query'].startswith('query: ')
        assert item['positive'].startswith('passage: ')
        assert all(neg.startswith('passage: ') for neg in item['negatives'])
    
    def test_dataset_length(self, sample_training_data):
        """Test dataset length"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=64,
            max_doc_len=128
        )
        
        assert len(dataset) == len(sample_training_data)


class TestCollateFn:
    """Test collate function"""
    
    def test_collate_fn_output_structure(self, sample_training_data):
        """Test collate function output structure"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=64,
            max_doc_len=128
        )
        
        batch = [dataset[i] for i in range(len(dataset))]
        collated = collate_fn(batch, tokenizer, max_query_len=64, max_doc_len=128)
        
        # Check structure
        assert 'query' in collated
        assert 'positive' in collated
        assert 'negative' in collated
        assert 'num_negatives' in collated
        
        # Check types - BatchEncoding is dict-like
        from transformers.tokenization_utils_base import BatchEncoding
        assert isinstance(collated['query'], (dict, BatchEncoding))
        assert isinstance(collated['positive'], (dict, BatchEncoding))
        assert isinstance(collated['negative'], (dict, BatchEncoding))
        assert isinstance(collated['num_negatives'], int)

    
    def test_collate_fn_tokenization(self, sample_training_data):
        """Test that collate function tokenizes correctly"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=64,
            max_doc_len=128
        )
        
        batch = [dataset[0]]
        collated = collate_fn(batch, tokenizer, max_query_len=64, max_doc_len=128)
        
        # Check tokenization keys (BatchEncoding has dict-like access)
        assert 'input_ids' in collated['query']
        assert 'attention_mask' in collated['query']
        assert 'input_ids' in collated['positive']
        assert 'attention_mask' in collated['positive']
        
        # Check tensor types
        assert isinstance(collated['query']['input_ids'], torch.Tensor)
        assert isinstance(collated['query']['attention_mask'], torch.Tensor)

    
    def test_collate_fn_max_length(self, sample_training_data):
        """Test that max length is respected"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        dataset = RetrievalDataset(
            samples=sample_training_data,
            tokenizer=tokenizer,
            max_query_len=32,
            max_doc_len=64
        )
        
        batch = [dataset[0]]
        collated = collate_fn(batch, tokenizer, max_query_len=32, max_doc_len=64)
        
        # Check max lengths
        assert collated['query']['input_ids'].shape[1] <= 32
        assert collated['positive']['input_ids'].shape[1] <= 64


class TestTrainingPipeline:
    """Test training pipeline components"""
    
    def test_training_data_preprocessing(self, sample_training_data):
        """Test that training data is preprocessed correctly"""
        for sample in sample_training_data:
            # Check all required fields exist
            assert 'query' in sample
            assert 'positive' in sample
            assert 'negatives' in sample
            
            # Check types
            assert isinstance(sample['query'], str)
            assert isinstance(sample['positive'], str)
            assert isinstance(sample['negatives'], list)
            
            # Check non-empty
            assert len(sample['query']) > 0
            assert len(sample['positive']) > 0
            assert len(sample['negatives']) > 0
