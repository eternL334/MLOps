"""Configuration management"""

import yaml
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    log_dir: str
    random_seed: int


@dataclass
class DataConfig:
    dataset_name: str
    max_query_length: int
    max_doc_length: int
    cache_dir: str


@dataclass
class ModelConfig:
    model_name: str
    embedding_dim: int
    pooling: str
    normalize_embeddings: bool


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    gradient_accumulation_steps: int
    fp16: bool
    dataloader_num_workers: int
    loss_type: str
    temperature: float
    num_negatives: int


@dataclass
class EvaluationConfig:
    batch_size: int
    top_k: List[int]
    metrics: List[str]
    eval_steps: int
    save_steps: int


@dataclass
class IndexConfig:
    type: str
    metric: str
    use_gpu: bool


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    index: IndexConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Config object with all settings
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(
        experiment=ExperimentConfig(**config_dict['experiment']),
        data=DataConfig(**config_dict['data']),
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        evaluation=EvaluationConfig(**config_dict['evaluation']),
        index=IndexConfig(**config_dict['index']),
    )
    
    logger.info(f"Configuration loaded successfully: {config.experiment.name}")
    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Config object
        save_path: Path to save config
    """
    config_dict = {
        'experiment': config.experiment.__dict__,
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'evaluation': config.evaluation.__dict__,
        'index': config.index.__dict__,
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {save_path}")
