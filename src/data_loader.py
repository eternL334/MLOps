"""Data loading and preprocessing for dense retrieval"""

import logging
from typing import Dict, List, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import random

logger = logging.getLogger(__name__)


class FiQADataset:
    """FiQA dataset loader and preprocessor"""
    
    def __init__(self, dataset_name: str, cache_dir: str):
        """
        Initialize FiQA dataset loader
        
        Args:
            dataset_name: Name of BEIR dataset
            cache_dir: Directory to cache downloaded data
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.corpus = None
        self.queries = None
        self.qrels = None
        
    def load_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        Load FiQA dataset from BEIR
        
        Returns:
            Tuple of (corpus, queries, qrels)
        """
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = util.download_and_unzip(url, self.cache_dir)
        
        # Load train split
        corpus, queries_train, qrels_train = GenericDataLoader(
            data_path
        ).load(split="train")
        
        # Load dev split
        _, queries_dev, qrels_dev = GenericDataLoader(
            data_path
        ).load(split="dev")
        
        # Load test split
        _, queries_test, qrels_test = GenericDataLoader(
            data_path
        ).load(split="test")
        
        self.corpus = corpus
        self.queries_train = queries_train
        self.qrels_train = qrels_train
        self.queries_dev = queries_dev
        self.qrels_dev = qrels_dev
        self.queries_test = queries_test
        self.qrels_test = qrels_test
        
        logger.info(f"Loaded {len(corpus)} documents")
        logger.info(f"Loaded {len(queries_train)} training queries")
        logger.info(f"Loaded {len(queries_dev)} dev queries")
        logger.info(f"Loaded {len(queries_test)} test queries")
        
        return corpus, queries_train, qrels_train
    
    def prepare_training_data(
        self, 
        num_negatives: int = 1,
        random_seed: int = 42
    ) -> List[Dict]:
        """
        Prepare training data with positive and negative samples
        
        Args:
            num_negatives: Number of negative samples per query
            random_seed: Random seed for negative sampling
            
        Returns:
            List of training examples with query, positive, and negatives
        """
        logger.info("Preparing training data with negatives...")
        
        random.seed(random_seed)
        training_samples = []
        
        corpus_ids = list(self.corpus.keys())
        
        for query_id, query_text in self.queries_train.items():
            if query_id not in self.qrels_train:
                continue
            
            # Get positive documents
            positive_docs = list(self.qrels_train[query_id].keys())
            
            if not positive_docs:
                continue
            
            # Sample one positive
            pos_id = random.choice(positive_docs)
            pos_text = self.corpus[pos_id]['title'] + ' ' + self.corpus[pos_id]['text']
            
            # Sample negatives (documents not in positive set)
            negative_candidates = [
                doc_id for doc_id in corpus_ids 
                if doc_id not in positive_docs
            ]
            
            neg_ids = random.sample(
                negative_candidates, 
                min(num_negatives, len(negative_candidates))
            )
            
            neg_texts = [
                self.corpus[neg_id]['title'] + ' ' + self.corpus[neg_id]['text']
                for neg_id in neg_ids
            ]
            
            training_samples.append({
                'query': query_text,
                'positive': pos_text,
                'negatives': neg_texts,
                'query_id': query_id,
                'positive_id': pos_id,
                'negative_ids': neg_ids
            })
        
        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def get_corpus_list(self) -> Tuple[List[str], List[str]]:
        """
        Get corpus as list of texts and IDs
        
        Returns:
            Tuple of (document texts, document IDs)
        """
        doc_ids = []
        doc_texts = []
        
        for doc_id, doc_data in self.corpus.items():
            doc_ids.append(doc_id)
            doc_text = doc_data['title'] + ' ' + doc_data['text']
            doc_texts.append(doc_text)
        
        return doc_texts, doc_ids
    
    def get_validation_data(self, split: str = "dev") -> Tuple[Dict, Dict]:
        """
        Get validation/test queries and qrels
        
        Args:
            split: 'dev' or 'test'
            
        Returns:
            Tuple of (queries, qrels)
        """
        if split == "dev":
            return self.queries_dev, self.qrels_dev
        elif split == "test":
            return self.queries_test, self.qrels_test
        else:
            raise ValueError(f"Unknown split: {split}")


def add_instruction_prefix(texts: List[str], instruction: str) -> List[str]:
    """
    Add instruction prefix to texts (required for E5 models)
    
    Args:
        texts: List of texts
        instruction: Instruction prefix (e.g., "query: " or "passage: ")
        
    Returns:
        Texts with prefix added
    """
    return [instruction + text for text in texts]
