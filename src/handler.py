# src/handler.py
import torch
import os
import pickle
import faiss
import logging
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class RetrievalHandler(BaseHandler):
    def __init__(self):
        super(RetrievalHandler, self).__init__()
        self.model = None
        self.tokenizer = None
        self.index = None
        self.doc_ids = None
        self.corpus_map = None
        self.initialized = False

    def initialize(self, context):
        """
        Загрузка всех тяжелых файлов в память при старте контейнера
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()

        index_path = os.path.join(model_dir, "fiqa.index")
        ids_path = os.path.join(model_dir, "fiqa.index.ids.pkl")
        
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        
        with open(ids_path, 'rb') as f:
            self.doc_ids = pickle.load(f)

        corpus_path = os.path.join(model_dir, "corpus_map.pkl")
        logger.info("Loading Corpus Map...")
        with open(corpus_path, 'rb') as f:
            self.corpus_map = pickle.load(f)

        self.initialized = True
        logger.info("RetrievalHandler Initialized successfully!")

    def preprocess(self, data):
        """
        Получаем JSON от пользователя, вытаскиваем текст запроса
        """
        queries = []
        for row in data:
            q = row.get("data") or row.get("body")
            if isinstance(q, (bytes, bytearray)):
                q = q.decode("utf-8")
            
            if isinstance(q, dict) and "query" in q:
                queries.append(q["query"])
            else:
                queries.append(str(q))
        
        return ["query: " + q for q in queries]

    def inference(self, data, *args, **kwargs):
        """
        Прогон модели + Поиск в индексе
        """
        inputs = self.tokenizer(
            data, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1) 
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        emb_np = embeddings.cpu().numpy()
        k = 3 
        scores, indices = self.index.search(emb_np, k)
        
        return scores, indices

    def postprocess(self, inference_output):
        """
        Формируем красивый JSON ответ
        """
        scores_batch, indices_batch = inference_output
        result = []

        for i, (scores, indices) in enumerate(zip(scores_batch, indices_batch)):
            hits = []
            for score, idx in zip(scores, indices):
                if idx == -1: continue
                
                real_doc_id = self.doc_ids[idx]
                
                text = self.corpus_map.get(real_doc_id, "")
                
                hits.append({
                    "doc_id": real_doc_id,
                    "score": float(score),
                    "text": text[:200] + "..." 
                })
            result.append(hits)
        
        return result
