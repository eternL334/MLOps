"""Model loading and management"""

import logging
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import os

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retrieval model wrapper"""
    
    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        device: str = None
    ):
        """
        Initialize dense retriever model
        
        Args:
            model_name: HuggingFace model name or path
            normalize_embeddings: Whether to L2 normalize embeddings
            device: Device to load model on
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            show_progress: Show progress bar
            
        Returns:
            Tensor of embeddings
        """
        self.model.eval()
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(texts), batch_size)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = self.model(**encoded)
                embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_model(self, save_path: str):
        """
        Save model in HuggingFace format
        
        Args:
            save_path: Path to save model
        """
        logger.info(f"Saving model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings,
        }
        
        import json
        with open(os.path.join(save_path, 'retriever_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, model_path: str, device: str = None):
        """
        Load model from path
        
        Args:
            model_path: Path to saved model
            device: Device to load model on
            
        Returns:
            DenseRetriever instance
        """
        import json
        config_path = os.path.join(model_path, 'retriever_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            normalize_embeddings = config.get('normalize_embeddings', True)
        else:
            normalize_embeddings = True
        
        return cls(
            model_name=model_path,
            normalize_embeddings=normalize_embeddings,
            device=device
        )
