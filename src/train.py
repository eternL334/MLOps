"""Training logic for dense retrieval"""

import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict
from tqdm import tqdm
import os
import argparse
import json
from .config import load_config
from .model import DenseRetriever
from .utils import setup_logging, set_seed, create_output_dirs

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss"""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, query_emb, pos_emb, neg_emb):
        """
        Compute contrastive loss
        
        Args:
            query_emb: Query embeddings [batch_size, dim]
            pos_emb: Positive document embeddings [batch_size, dim]
            neg_emb: Negative document embeddings [batch_size, num_neg, dim]
        """
        batch_size = query_emb.size(0)
        
        # Compute similarities
        pos_sim = torch.sum(query_emb * pos_emb, dim=1, keepdim=True)  # [B, 1]
        
        # Reshape for negative similarities
        query_emb_expanded = query_emb.unsqueeze(1)  # [B, 1, dim]
        neg_sim = torch.sum(query_emb_expanded * neg_emb, dim=2)  # [B, num_neg]
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # [B, 1+num_neg]
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = self.criterion(logits, labels)
        return loss


class RetrievalDataset(Dataset):
    """Dataset for dense retrieval training"""
    
    def __init__(self, samples: List[Dict], tokenizer, max_query_len: int, max_doc_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Add E5 instruction prefixes
        query = "query: " + sample['query']
        positive = "passage: " + sample['positive']
        negatives = ["passage: " + neg for neg in sample['negatives']]
        
        return {
            'query': query,
            'positive': positive,
            'negatives': negatives
        }


def collate_fn(batch, tokenizer, max_query_len, max_doc_len):
    """Custom collate function"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    # Handle negatives (can be variable number)
    all_negatives = []
    for item in batch:
        all_negatives.extend(item['negatives'])
    
    # Tokenize
    query_encoded = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_query_len,
        return_tensors='pt'
    )
    
    pos_encoded = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_doc_len,
        return_tensors='pt'
    )
    
    neg_encoded = tokenizer(
        all_negatives,
        padding=True,
        truncation=True,
        max_length=max_doc_len,
        return_tensors='pt'
    )
    
    num_negatives = len(batch[0]['negatives'])
    
    return {
        'query': query_encoded,
        'positive': pos_encoded,
        'negative': neg_encoded,
        'num_negatives': num_negatives
    }


class Trainer:
    """Trainer for dense retrieval model"""
    
    def __init__(
        self,
        model,
        train_dataset,
        config,
        output_dir: str
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.output_dir = output_dir
        
        self.device = model.device
        self.global_step = 0
        
        # Setup loss
        self.loss_fn = ContrastiveLoss(temperature=config.training.temperature)
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.training.weight_decay
            },
            {
                'params': [p for n, p in model.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.training.learning_rate
        )
        
        # Setup dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.dataloader_num_workers,
            collate_fn=lambda batch: collate_fn(
                batch,
                model.tokenizer,
                config.data.max_query_length,
                config.data.max_doc_length
            )
        )
        
        # Setup scheduler
        total_steps = len(self.train_dataloader) * config.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup AMP
        self.scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None
        
        logger.info(f"Trainer initialized with {len(train_dataset)} samples")
        logger.info(f"Total training steps: {total_steps}")
    
    def train(self):
        """Run training"""
        logger.info("Starting training...")
        
        self.model.model.train()
        
        for epoch in range(self.config.training.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                loss = self.training_step(batch)
                epoch_loss += loss
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Save checkpoint
                if (self.global_step + 1) % self.config.evaluation.save_steps == 0:
                    self.save_checkpoint(epoch, step)
                
                self.global_step += 1
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        logger.info("Training completed!")
    
    def training_step(self, batch):
        """Single training step"""
        # Move to device
        query_input = {k: v.to(self.device) for k, v in batch['query'].items()}
        pos_input = {k: v.to(self.device) for k, v in batch['positive'].items()}
        neg_input = {k: v.to(self.device) for k, v in batch['negative'].items()}
        num_negatives = batch['num_negatives']
        
        # Forward pass with AMP
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Encode query
                query_output = self.model.model(**query_input)
                query_emb = self.model.mean_pooling(query_output, query_input['attention_mask'])
                
                # Encode positive
                pos_output = self.model.model(**pos_input)
                pos_emb = self.model.mean_pooling(pos_output, pos_input['attention_mask'])
                
                # Encode negatives
                neg_output = self.model.model(**neg_input)
                neg_emb = self.model.mean_pooling(neg_output, neg_input['attention_mask'])
                
                # Reshape negatives
                batch_size = query_emb.size(0)
                neg_emb = neg_emb.view(batch_size, num_negatives, -1)
                
                # Normalize
                if self.model.normalize_embeddings:
                    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                    pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
                    neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=2)
                
                # Compute loss
                loss = self.loss_fn(query_emb, pos_emb, neg_emb)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular forward pass
            query_output = self.model.model(**query_input)
            query_emb = self.model.mean_pooling(query_output, query_input['attention_mask'])
            
            pos_output = self.model.model(**pos_input)
            pos_emb = self.model.mean_pooling(pos_output, pos_input['attention_mask'])
            
            neg_output = self.model.model(**neg_input)
            neg_emb = self.model.mean_pooling(neg_output, neg_input['attention_mask'])
            
            batch_size = query_emb.size(0)
            neg_emb = neg_emb.view(batch_size, num_negatives, -1)
            
            if self.model.normalize_embeddings:
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
                neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=2)
            
            loss = self.loss_fn(query_emb, pos_emb, neg_emb)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def save_checkpoint(self, epoch, step):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(
            self.output_dir,
            'checkpoints',
            f'checkpoint-epoch{epoch}-step{step}'
        )
        self.model.save_model(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--train_file", type=str, required=True, help="DVC path")
    parser.add_argument("--output_dir", type=str, required=True, help="Save model path")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config.experiment.log_dir, log_file="train_dvc.log")
    set_seed(config.experiment.random_seed)
    
    logger.info(f"Loading training data from {args.train_file}...")
    training_samples = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        for line in f:
            training_samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(training_samples)} samples")

    model = DenseRetriever(
        model_name=config.model.model_name,
        normalize_embeddings=config.model.normalize_embeddings
    )

    train_dataset = RetrievalDataset(
        samples=training_samples,
        tokenizer=model.tokenizer,
        max_query_len=config.data.max_query_length,
        max_doc_len=config.data.max_doc_length
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=config,
        output_dir=args.output_dir
    )

    trainer.train()

    logger.info(f"Saving model to {args.output_dir}")
    model.save_model(args.output_dir)
