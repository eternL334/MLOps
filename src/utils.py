"""Utility functions for logging and reproducibility"""

import logging
import os
import random
import numpy as np
import torch
from pathlib import Path


def setup_logging(log_dir: str, log_file: str = "training.log") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save logs
        log_file: Name of log file
        
    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_output_dirs(output_dir: str) -> dict:
    """
    Create necessary output directories
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary with paths to created directories
    """
    paths = {
        'base': output_dir,
        'checkpoints': os.path.join(output_dir, 'checkpoints'),
        'final_model': os.path.join(output_dir, 'final_model'),
        'embeddings': os.path.join(output_dir, 'embeddings'),
        'results': os.path.join(output_dir, 'results'),
    }
    
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return paths
