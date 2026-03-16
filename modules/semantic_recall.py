import numpy as np
from typing import List, Dict, Set, Tuple
import networkx as nx
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EXPLICIT_TOPK, DESC_TOPK, SUBGRAPH_HOP
from utils.embedding import EmbeddingEncoder, FaissIndex


class SemanticRecall:
    def __init__(self, encoder: EmbeddingEncoder, entities: Dict[str, dict], graph: nx.DiGraph):
        self.encoder = encoder
        self.entities = entities
        self.graph = graph
        self.entity_ids = list(entities.keys())
        self.name_index = None
        self.type_embeddings = None
        self.desc_index = None
        self._type_emb_map = {}
        self._built = False

        self.head_role_weight = 1.0
        self.tail_role_weight = 0.4
        self.head_direct_seed_thres = 0.80
        self.tail_direct_seed_thres = 0.90

    def build_indexes(self):
        names = [self.entities[eid]["name"] for eid in self.entity_ids]
        name_embs = self.encoder.encode(names)
        self.name_index = FaissIndex()
        self.name_index.build(name_embs, self.entity_ids)

        types = [self.entities[eid]["type"] or "未知" for eid in self.entity_ids]
        self.type_embeddings = self.encoder.encode(types)
        self._type_emb_map = {eid: self.type_embeddings[i] for i, eid in enumerate(self.entity_ids)}

        descs = [
            f"实体名称：{self.entities[eid]['name']}；实体类型：{self.entities[eid]['type']}；实体描述：{self.entities[eid]['description']}"
            for eid in self.entity_ids
        ]
        desc_embs = self.encoder.encode(descs)
        self.desc_index = FaissIndex()
        self.desc_index.build(desc_embs, self.entity_ids)
        self._built = True

    def save_indexes(self, d):
        os.makedirs(d, exist_ok=True)
        self.name_index.save(os.path.join(d, "name.faiss"))
        self.desc_index.save(os.path.join(d, "desc.faiss"))
        np.save(os.path.join(d, "type_embs.npy"), self.type_embeddings)

    def load_indexes(self, d):
        self.name_index = FaissIndex()
        self.name_index.load(os.path.join(d, "name.faiss"))
        self.desc_index = FaissIndex()
        self.desc_index.load(os.path.join(d, "desc.faiss"))
        self.type_embeddings = np.load(os.path.join(d, "type_embs.npy"))
        self._type_emb_map = {eid: self.type_embeddings[i] for i, eid in enumerate(self.entity_ids)}
        self._built = True

    def explicit_feature_retrieval(self, query_entities, topk=None) -> Set[str]:

        topk = topk or EXPLICIT_TOPK
        recalled = set()
        type_emb_cache = {}

        for qe in query_entities:
            role = qe.get("role", "head")
            is_head = (role == "head")

            role_weight = self.head_role_weight if is_head else self.tail_role_weight
            direct_seed_thres = self.head_direct_seed_thres if is_head else self.tail_direct_seed_thres

            local_topk = topk if is_head else max(1, int(round(topk * 0.4)))


            cids = qe.get("candidate_ids", [])
            cscores = qe.get("candidate_scores", [])
            for cid, s in zip(cids, cscores):
                if s >= direct_seed_thres:
                    recalled.add(cid)


            type_list = qe.get("types", None)
            if not type_list:
                rt = qe.get("resolved_type", None)
                if rt:
                    type_list = [rt]
                else:
                    t = qe.get("type", "未知") or "未知"
                    type_list = [t]

            type_embs = []
            for t in type_list:
                if t not in type_emb_cache:
                    type_emb_cache[t] = self.encoder.encode_single(t or "未知")
                type_embs.append(type_emb_cache[t])


            q_name_emb = self.encoder.encode_single(qe["name"])
            candidates = self.name_index.search(q_name_emb, min(max(local_topk * 3, 3), len(self.entity_ids)))
            scored = []

            for eid, ns in candidates:
                if eid not in self._type_emb_map:
                    continue

                ts = max(
                    max(float(np.dot(te, self._type_emb_map[eid])), 0.0)
                    for te in type_embs
                )


                final_score = ns * ts * role_weight
                scored.append((eid, final_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            for eid, _ in scored[:local_topk]:
                recalled.add(eid)

        return recalled

    def deep_semantic_association(self, query_set: List[str], topk=None) -> Set[str]:

        topk = topk or DESC_TOPK
        recalled = set()
        q_embs = self.encoder.encode(query_set)
        for results in self.desc_index.batch_search(q_embs, topk):
            for eid, _ in results:
                recalled.add(eid)
        return recalled

    def recall_and_build_subgraph(self, query_set, query_entities, hop=None):

        hop = hop or SUBGRAPH_HOP
        N = self.explicit_feature_retrieval(query_entities)
        D = self.deep_semantic_association(query_set)
        E0 = N | D
        sg = self._build_hop_subgraph(E0, hop)
        return E0, sg

    def _build_hop_subgraph(self, seeds: Set[str], hop: int) -> nx.DiGraph:
        valid = {s for s in seeds if s in self.graph}
        if not valid:
            return nx.DiGraph()

        visited, frontier = set(valid), set(valid)
        for _ in range(hop):
            nxt = set()
            for n in frontier:
                for nb in self.graph.successors(n):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
                for nb in self.graph.predecessors(n):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            frontier = nxt
            if not frontier:
                break

        sg = self.graph.subgraph(visited).copy()
        return sg