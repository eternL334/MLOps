"""Tests for configuration management"""

import pytest
import yaml
import os
from src.config import load_config, save_config, Config


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_load_config(self, sample_config):
        """Test loading config from file"""
        config = load_config(sample_config)
        
        assert isinstance(config, Config)
        assert config.experiment.name == 'test-experiment'
        assert config.model.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert config.training.batch_size == 4
    
    def test_config_structure(self, sample_config):
        """Test config has all required sections"""
        config = load_config(sample_config)
        
        assert hasattr(config, 'experiment')
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'index')
    
    def test_config_types(self, sample_config):
        """Test config values have correct types"""
        config = load_config(sample_config)
        
        # Experiment
        assert isinstance(config.experiment.name, str)
        assert isinstance(config.experiment.random_seed, int)
        
        # Data
        assert isinstance(config.data.max_query_length, int)
        assert isinstance(config.data.max_doc_length, int)
        
        # Model
        assert isinstance(config.model.embedding_dim, int)
        assert isinstance(config.model.normalize_embeddings, bool)
        
        # Training
        assert isinstance(config.training.num_epochs, int)
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.training.batch_size, int)
        
        # Evaluation
        assert isinstance(config.evaluation.top_k, list)
        assert isinstance(config.evaluation.metrics, list)
    
    def test_save_config(self, sample_config, temp_dir):
        """Test saving configuration"""
        config = load_config(sample_config)
        
        save_path = os.path.join(temp_dir, 'saved_config.yaml')
        save_config(config, save_path)
        
        assert os.path.exists(save_path)
        
        # Load saved config
        with open(save_path, 'r') as f:
            saved_dict = yaml.safe_load(f)
        
        assert 'experiment' in saved_dict
        assert 'model' in saved_dict
        assert 'training' in saved_dict


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_config_value_ranges(self, sample_config):
        """Test config values are in valid ranges"""
        config = load_config(sample_config)
        
        # Positive values
        assert config.training.num_epochs > 0
        assert config.training.batch_size > 0
        assert config.training.learning_rate > 0
        assert config.data.max_query_length > 0
        assert config.data.max_doc_length > 0
        
        # Rates in [0, 1]
        assert 0 <= config.training.weight_decay <= 1
        
        # Temperature > 0
        assert config.training.temperature > 0
