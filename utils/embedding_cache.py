import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional


class NodeEmbeddingCache:
    def __init__(self, cache_dir: str, mmap_mode: Optional[str] = None):

        self.cache_dir = cache_dir
        self.mmap_mode = mmap_mode

        self._embeddings: Optional[np.ndarray] = None   # (N, dim)
        self._id_list: Optional[List[str]] = None        # len = N
        self._id_to_idx: Optional[Dict[str, int]] = None # entity_id -> row index
        self._loaded = False


    def load(self) -> "NodeEmbeddingCache":
        if self._loaded:
            return self

        emb_path = os.path.join(self.cache_dir, "node_embeddings.npy")
        idmap_path = os.path.join(self.cache_dir, "node_id_map.pkl")

        if not os.path.exists(emb_path) or not os.path.exists(idmap_path):
            raise FileNotFoundError(
                f"[NodeEmbeddingCache] 缓存文件不存在: {self.cache_dir}\n"
                f"请先运行离线构建: python embedding_cache.py --build"
            )

        self._embeddings = np.load(emb_path, mmap_mode=self.mmap_mode)
        with open(idmap_path, "rb") as f:
            self._id_list = pickle.load(f)

        self._id_to_idx = {eid: i for i, eid in enumerate(self._id_list)}
        self._loaded = True
        return self


    def __contains__(self, entity_id: str) -> bool:
        return self._id_to_idx is not None and entity_id in self._id_to_idx

    def __getitem__(self, entity_id: str) -> np.ndarray:

        if not self._loaded:
            self.load()
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return np.zeros(self._embeddings.shape[1], dtype=np.float32)
        return np.array(self._embeddings[idx], dtype=np.float32)

    def get_batch(self, entity_ids: List[str]) -> Dict[str, np.ndarray]:

        if not self._loaded:
            self.load()

        result = {}
        dim = self._embeddings.shape[1]

        hit_ids = []
        hit_indices = []
        miss_ids = []

        for eid in entity_ids:
            idx = self._id_to_idx.get(eid)
            if idx is not None:
                hit_ids.append(eid)
                hit_indices.append(idx)
            else:
                miss_ids.append(eid)

        if hit_indices:

            batch_emb = self._embeddings[hit_indices]  # (k, dim)
            for i, eid in enumerate(hit_ids):
                result[eid] = np.array(batch_emb[i], dtype=np.float32)


        zero = np.zeros(dim, dtype=np.float32)
        for eid in miss_ids:
            result[eid] = zero.copy()

        return result

    @property
    def dim(self) -> int:
        if self._embeddings is None:
            raise RuntimeError("Cache not loaded")
        return self._embeddings.shape[1]

    @property
    def size(self) -> int:
        return len(self._id_list) if self._id_list else 0


    @staticmethod
    def build_cache(
        entity_table_path: str,
        encoder,
        cache_dir: str,
        batch_size: int = 256,
        text_template: str = "{name}；{type}；{description}",
    ):

        import time

        entities = {}
        with open(entity_table_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                eid = obj.get("id") or obj.get("entity_id")
                if eid:
                    entities[eid] = obj


        entity_ids = sorted(entities.keys())
        texts = []
        for eid in entity_ids:
            e = entities[eid]
            text = text_template.format(
                name=e.get("name", ""),
                type=e.get("type", ""),
                description=e.get("description", ""),
            )
            texts.append(text)


        t0 = time.time()
        embeddings = encoder.encode(texts, batch_size=batch_size)  # (N, dim)
        elapsed = time.time() - t0


        os.makedirs(cache_dir, exist_ok=True)

        emb_path = os.path.join(cache_dir, "node_embeddings.npy")
        idmap_path = os.path.join(cache_dir, "node_id_map.pkl")
        meta_path = os.path.join(cache_dir, "cache_meta.json")

        np.save(emb_path, embeddings.astype(np.float32))
        with open(idmap_path, "wb") as f:
            pickle.dump(entity_ids, f)

        meta = {
            "num_entities": len(entity_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "text_template": text_template,
            "source": entity_table_path,
            "build_time_sec": round(elapsed, 2),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[BuildCache] 完成: {len(entity_ids)} 实体, dim={int(embeddings.shape[1])}, 保存至 {cache_dir}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DUAL-Know 节点 Embedding 离线缓存构建")
    parser.add_argument("--build", action="store_true", help="执行离线构建")
    parser.add_argument("--entity_table", type=str, default=None,
                        help="entity_table.jsonl 路径")
    parser.add_argument("--bge_model", type=str, default=None,
                        help="BGE 模型路径")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="缓存输出目录")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    if args.build:

        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from configs.config import ENTITY_TABLE_PATH, BGE_MODEL_PATH, OUTPUT_DIR
        from utils.embedding import EmbeddingEncoder

        entity_path = args.entity_table or ENTITY_TABLE_PATH
        bge_path = args.bge_model or BGE_MODEL_PATH
        cache_dir = args.cache_dir or os.path.join(OUTPUT_DIR, "node_emb_cache")

        encoder = EmbeddingEncoder(bge_path).load()
        NodeEmbeddingCache.build_cache(
            entity_table_path=entity_path,
            encoder=encoder,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
        )
    else:
        parser.print_help()
