"""Tests for data loading and preprocessing"""

import pytest
from src.data_loader import FiQADataset, add_instruction_prefix


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_add_instruction_prefix(self):
        """Test instruction prefix addition"""
        texts = ["text1", "text2", "text3"]
        instruction = "query: "
        
        result = add_instruction_prefix(texts, instruction)
        
        assert len(result) == len(texts)
        assert all(text.startswith(instruction) for text in result)
        assert result[0] == "query: text1"
        assert result[1] == "query: text2"
    
    def test_add_instruction_prefix_types(self):
        """Test instruction prefix with different types"""
        texts = ["text1"]
        
        # Test with different instructions
        for instruction in ["query: ", "passage: ", ""]:
            result = add_instruction_prefix(texts, instruction)
            assert isinstance(result, list)
            assert isinstance(result[0], str)


class TestFiQADataset:
    """Test FiQA dataset class"""
    
    def test_dataset_initialization(self, temp_dir):
        """Test dataset initialization"""
        dataset = FiQADataset(
            dataset_name="fiqa",
            cache_dir=temp_dir
        )
        
        assert dataset.dataset_name == "fiqa"
        assert dataset.cache_dir == temp_dir
        assert dataset.corpus is None
        assert dataset.queries is None
    
    def test_prepare_training_data_structure(self, sample_corpus):
        """Test training data preparation structure"""
        # Create mock dataset
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        dataset.corpus = sample_corpus
        dataset.queries_train = {'q1': 'What is Python?', 'q2': 'Explain ML'}
        dataset.qrels_train = {
            'q1': {'doc1': 1},
            'q2': {'doc2': 1}
        }
        
        training_samples = dataset.prepare_training_data(num_negatives=2)
        
        # Check structure
        assert isinstance(training_samples, list)
        assert len(training_samples) > 0
        
        for sample in training_samples:
            assert 'query' in sample
            assert 'positive' in sample
            assert 'negatives' in sample
            assert 'query_id' in sample
            assert 'positive_id' in sample
            assert 'negative_ids' in sample
            
            # Check types
            assert isinstance(sample['query'], str)
            assert isinstance(sample['positive'], str)
            assert isinstance(sample['negatives'], list)
            assert len(sample['negatives']) <= 2
            
            # Check that positive is not in negatives
            assert sample['positive_id'] not in sample['negative_ids']
    
    def test_prepare_training_data_no_empty_queries(self, sample_corpus):
        """Test that empty queries are filtered out"""
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        dataset.corpus = sample_corpus
        dataset.queries_train = {'q1': 'What is Python?', 'q2': ''}
        dataset.qrels_train = {'q1': {'doc1': 1}}
        
        training_samples = dataset.prepare_training_data(num_negatives=1)
        
        for sample in training_samples:
            assert len(sample['query']) > 0
            assert len(sample['positive']) > 0
    
    def test_get_corpus_list(self, sample_corpus):
        """Test corpus list extraction"""
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        dataset.corpus = sample_corpus
        
        doc_texts, doc_ids = dataset.get_corpus_list()
        
        # Check structure
        assert isinstance(doc_texts, list)
        assert isinstance(doc_ids, list)
        assert len(doc_texts) == len(doc_ids)
        assert len(doc_texts) == len(sample_corpus)
        
        # Check types
        assert all(isinstance(text, str) for text in doc_texts)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        
        # Check content
        assert 'Python Programming' in doc_texts[0]
        assert doc_ids[0] in sample_corpus
    
    def test_get_corpus_list_combines_title_and_text(self, sample_corpus):
        """Test that title and text are combined"""
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        dataset.corpus = sample_corpus
        
        doc_texts, doc_ids = dataset.get_corpus_list()
        
        # Check that both title and text are present
        for doc_id, doc_text in zip(doc_ids, doc_texts):
            corpus_doc = sample_corpus[doc_id]
            assert corpus_doc['title'] in doc_text
            assert corpus_doc['text'] in doc_text
    
    def test_get_validation_data(self, sample_queries, sample_qrels):
        """Test validation data retrieval"""
        dataset = FiQADataset(dataset_name="fiqa", cache_dir="./cache")
        dataset.queries_dev = sample_queries
        dataset.qrels_dev = sample_qrels
        dataset.queries_test = {'q_test': 'test query'}
        dataset.qrels_test = {'q_test': {'doc1': 1}}
        
        # Test dev split
        queries_dev, qrels_dev = dataset.get_validation_data(split="dev")
        assert queries_dev == sample_queries
        assert qrels_dev == sample_qrels
        
        # Test test split
        queries_test, qrels_test = dataset.get_validation_data(split="test")
        assert 'q_test' in queries_test
        
        # Test invalid split
        with pytest.raises(ValueError):
            dataset.get_validation_data(split="invalid")


class TestDataValidation:
    """Test data validation and types"""
    
    def test_training_sample_types(self, sample_training_data):
        """Test that training samples have correct types"""
        for sample in sample_training_data:
            assert isinstance(sample['query'], str)
            assert isinstance(sample['positive'], str)
            assert isinstance(sample['negatives'], list)
            assert isinstance(sample['query_id'], str)
            assert isinstance(sample['positive_id'], str)
            assert isinstance(sample['negative_ids'], list)
            
            # Check nested types
            for neg in sample['negatives']:
                assert isinstance(neg, str)
            for neg_id in sample['negative_ids']:
                assert isinstance(neg_id, str)
    
    def test_training_sample_value_ranges(self, sample_training_data):
        """Test value constraints in training samples"""
        for sample in sample_training_data:
            # Text should not be empty
            assert len(sample['query']) > 0
            assert len(sample['positive']) > 0
            
            # Should have at least one negative
            assert len(sample['negatives']) > 0
            assert len(sample['negative_ids']) > 0
            
            # Negatives and IDs should match in count
            assert len(sample['negatives']) == len(sample['negative_ids'])
    
    def test_corpus_structure(self, sample_corpus):
        """Test corpus has correct structure"""
        for doc_id, doc_data in sample_corpus.items():
            assert isinstance(doc_id, str)
            assert isinstance(doc_data, dict)
            assert 'title' in doc_data
            assert 'text' in doc_data
            assert isinstance(doc_data['title'], str)
            assert isinstance(doc_data['text'], str)
    
    def test_qrels_structure(self, sample_qrels):
        """Test qrels have correct structure"""
        for query_id, doc_scores in sample_qrels.items():
            assert isinstance(query_id, str)
            assert isinstance(doc_scores, dict)
            
            for doc_id, score in doc_scores.items():
                assert isinstance(doc_id, str)
                assert isinstance(score, (int, float))
                assert score >= 0  # Relevance scores should be non-negative
