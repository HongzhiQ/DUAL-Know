
import numpy as np
import os
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import BGE_MODEL_PATH, DEVICE


class EmbeddingEncoder:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or BGE_MODEL_PATH
        self.device = device or DEVICE
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded: return self
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self._loaded = True
        return self

    def encode(self, texts: List[str], batch_size=64, normalize=True) -> np.ndarray:
        if not self._loaded: self.load()
        embs = self.model.encode(texts, batch_size=batch_size,
                                  show_progress_bar=len(texts) > 100,
                                  normalize_embeddings=normalize)
        return np.array(embs, dtype=np.float32)

    def encode_single(self, text: str, normalize=True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]


class FaissIndex:
    def __init__(self):
        self.index = None
        self.id_map = []

    def build(self, embeddings: np.ndarray, ids: List[str]):
        import faiss
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.id_map = list(ids)

    def search(self, query_emb: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
        if query_emb.ndim == 1: query_emb = query_emb.reshape(1, -1)
        scores, indices = self.index.search(query_emb.astype(np.float32), topk)
        return [(self.id_map[i], float(s)) for i, s in zip(indices[0], scores[0]) if 0 <= i < len(self.id_map)]

    def batch_search(self, query_embs: np.ndarray, topk: int = 10) -> List[List[Tuple[str, float]]]:
        scores, indices = self.index.search(query_embs.astype(np.float32), topk)
        results = []
        for row in range(len(indices)):
            results.append([(self.id_map[i], float(s)) for i, s in zip(indices[row], scores[row]) if 0 <= i < len(self.id_map)])
        return results

    def save(self, filepath: str):
        import faiss
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        with open(filepath + ".idmap.pkl", "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, filepath: str):
        import faiss
        self.index = faiss.read_index(filepath)
        with open(filepath + ".idmap.pkl", "rb") as f:
            self.id_map = pickle.load(f)
        return self
