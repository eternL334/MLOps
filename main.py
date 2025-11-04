"""Main script for training and evaluation"""

import argparse
import logging
import os
import json
from pathlib import Path

from src.config import load_config, save_config
from src.utils import setup_logging, set_seed, create_output_dirs
from src.data_loader import FiQADataset
from src.model import DenseRetriever
from src.train import Trainer, RetrievalDataset
from src.evaluate import evaluate_model


def main(config_path: str, eval_only: bool = False, model_path: str = None):
    """
    Main training and evaluation pipeline
    
    Args:
        config_path: Path to configuration YAML file
        eval_only: If True, skip training and only run evaluation
        model_path: Path to model checkpoint for evaluation (if eval_only=True)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.experiment.log_dir,
        log_file=f"{config.experiment.name}.log"
    )
    
    logger.info("=" * 80)
    if eval_only:
        logger.info(f"Starting evaluation: {config.experiment.name}")
    else:
        logger.info(f"Starting experiment: {config.experiment.name}")
    logger.info("=" * 80)
    
    # Set random seeds
    set_seed(config.experiment.random_seed)
    logger.info(f"Random seed set to {config.experiment.random_seed}")
    
    # Create output directories
    output_paths = create_output_dirs(config.experiment.output_dir)
    logger.info(f"Output directory: {config.experiment.output_dir}")
    
    # Save configuration
    save_config(config, os.path.join(output_paths['base'], 'config.yaml'))
    
    # ========== 1. Load and preprocess data ==========
    logger.info("Step 1: Loading and preprocessing data")
    dataset = FiQADataset(
        dataset_name=config.data.dataset_name,
        cache_dir=config.data.cache_dir
    )
    dataset.load_data()
    
    # ========== 2. Load model ==========
    logger.info("Step 2: Loading model")
    
    if eval_only and model_path:
        # Load model from specified checkpoint
        logger.info(f"Loading model from checkpoint: {model_path}")
        model = DenseRetriever.load_model(
            model_path=model_path
        )
        logger.info("Checkpoint loaded successfully")
    elif eval_only:
        # Load original pre-trained model for evaluation
        logger.info(f"Loading pre-trained model: {config.model.model_name}")
        model = DenseRetriever(
            model_name=config.model.model_name,
            normalize_embeddings=config.model.normalize_embeddings
        )
        logger.info("Pre-trained model loaded for evaluation")
    else:
        # Normal training mode - load pre-trained model
        model = DenseRetriever(
            model_name=config.model.model_name,
            normalize_embeddings=config.model.normalize_embeddings
        )
    
    # ========== 3-5. Training (skip if eval_only) ==========
    if not eval_only:
        # Prepare training data
        training_samples = dataset.prepare_training_data(
            num_negatives=config.training.num_negatives,
            random_seed=config.experiment.random_seed
        )
        
        logger.info("Step 3: Creating training dataset")
        train_dataset = RetrievalDataset(
            samples=training_samples,
            tokenizer=model.tokenizer,
            max_query_len=config.data.max_query_length,
            max_doc_len=config.data.max_doc_length
        )
        
        logger.info("Step 4: Training model")
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            config=config,
            output_dir=output_paths['base']
        )
        trainer.train()
        
        logger.info("Step 5: Saving final model")
        model.save_model(output_paths['final_model'])
    else:
        logger.info("Skipping training (evaluation-only mode)")
    
    # ========== 6. Evaluate model ==========
    logger.info("Step 6: Evaluating model")
    
    # Evaluate on dev set
    logger.info("Evaluating on dev set...")
    dev_metrics = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        split="dev"
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(
        model=model,
        dataset=dataset,
        config=config,
        split="test"
    )
    
    # Save results
    results = {
        'dev': dev_metrics,
        'test': test_metrics,
        'config': config.experiment.name,
        'eval_only': eval_only,
        'model_path': model_path if model_path else config.model.model_name
    }
    
    # Create unique results filename if eval_only
    if eval_only:
        model_name = os.path.basename(model_path) if model_path else "pretrained"
        results_filename = f'metrics_{model_name}.json'
    else:
        results_filename = 'metrics.json'
    
    results_path = os.path.join(output_paths['results'], results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # ========== Done ==========
    logger.info("=" * 80)
    if eval_only:
        logger.info("Evaluation completed successfully!")
    else:
        logger.info("Training and evaluation completed successfully!")
    logger.info("=" * 80)
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    if model_path:
        print(f"\nModel: {model_path}")
    else:
        print(f"\nModel: {config.model.model_name}")
    print("\nDev Set:")
    for metric, value in sorted(dev_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    print("\nTest Set:")
    for metric, value in sorted(test_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate dense retrieval model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and evaluate
  python main.py --config config.yaml
  
  # Evaluate pre-trained model only
  python main.py --config config.yaml --eval-only
  
  # Evaluate fine-tuned checkpoint
  python main.py --config config.yaml --eval-only --model-path ./outputs/final_model
  
  # Evaluate specific checkpoint
  python main.py --config config.yaml --eval-only --model-path ./outputs/checkpoints/checkpoint-epoch2-step1000
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation (optional, used with --eval-only)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.model_path and not args.eval_only:
        parser.error("--model-path can only be used with --eval-only")
    
    main(
        config_path=args.config,
        eval_only=args.eval_only,
        model_path=args.model_path
    )

