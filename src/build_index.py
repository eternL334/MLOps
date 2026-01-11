import argparse
import os
import sys
import logging

from .config import load_config
from .utils import setup_logging, set_seed
from .data_loader import FiQADataset
from .model import DenseRetriever
from .evaluate import VectorIndex

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--index_out", type=str, required=True, help="Путь, куда сохранить .index файл")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config.experiment.log_dir, log_file="build_index.log")
    set_seed(config.experiment.random_seed)

    logger.info("Loading corpus...")
    dataset = FiQADataset(
        dataset_name=config.data.dataset_name,
        cache_dir=os.path.dirname(args.raw_data_path)
    )
    dataset.load_data()
    doc_texts, doc_ids = dataset.get_corpus_list()
    
    doc_texts = ["passage: " + t for t in doc_texts]

    logger.info(f"Loading model from {args.model_path}")
    model = DenseRetriever.load_model(model_path=args.model_path)

    logger.info(f"Encoding {len(doc_texts)} documents...")
    embeddings = model.encode(
        doc_texts,
        batch_size=config.evaluation.batch_size * 2, 
        max_length=config.data.max_doc_length,
        show_progress=True
    )

    logger.info("Building FAISS index...")
    index = VectorIndex(
        dimension=config.model.embedding_dim, 
        metric=config.index.metric, 
        use_gpu=False 
    )
    index.add_documents(embeddings, doc_ids)

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    index.save(args.index_out)
    logger.info("Index built and saved.")

if __name__ == "__main__":
    main()
