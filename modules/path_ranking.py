import torch
import torch.nn.functional as F
import networkx as nx
from typing import Dict, List, Set, Tuple
from itertools import islice
import sys, os
import numpy as np  # 新增，给 _compute_query_path_relevance 用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import PATH_TOPK, UNCERTAINTY_LAMBDA, MAX_PATH_LENGTH, NUM_CANDIDATE_PATHS


class PathRanker:

    def __init__(self, topk=None, lam=None, max_len=None, n_cand=None, encoder=None):
        self.topk = topk or PATH_TOPK
        self.lam = lam or UNCERTAINTY_LAMBDA
        self.max_len = max_len or MAX_PATH_LENGTH
        self.n_cand = n_cand or NUM_CANDIDATE_PATHS
        self.encoder = encoder

        self.anchor_bonus_scale = 0.2

        self.semantic_relevance_scale = 0.2


    def _get_name(self, nid, subgraph, entities):
        if nid in entities:
            return entities[nid].get("name", nid)
        if nid in subgraph.nodes:
            return subgraph.nodes[nid].get("name", nid)
        return nid

    def _resolve_edge_direction(self, s, t, subgraph):

        if subgraph.has_edge(s, t):
            rel = subgraph[s][t].get("relation", "相关")
            return s, rel, t
        elif subgraph.has_edge(t, s):
            rel = subgraph[t][s].get("relation", "相关")
            return t, rel, s
        else:
            return s, "相关", t

    def path_to_readable(self, path, subgraph, entities):

        if not path:
            return ""

        if len(path) == 1:
            return self._get_name(path[0], subgraph, entities)

        pieces = [self._get_name(path[0], subgraph, entities)]

        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            rel, is_forward = self._resolve_edge_for_path_display(s, t, subgraph)

            if is_forward:
                pieces.append(f"--{rel}-->")
            else:
                pieces.append(f"<--{rel}--")

            pieces.append(self._get_name(t, subgraph, entities))

        return " ".join(pieces)

    def paths_to_readable(self, paths, subgraph, entities):
        return [self.path_to_readable(path, subgraph, entities) for path in paths]

    def path_to_readable_struct(self, path, subgraph, entities):

        items = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]

            if subgraph.has_edge(s, t):
                rel = subgraph[s][t].get("relation", "相关")
                direction = "forward"
            elif subgraph.has_edge(t, s):
                rel = subgraph[t][s].get("relation", "相关")
                direction = "backward"
            else:
                rel = "相关"
                direction = "unknown"

            items.append({
                "source_id": s,
                "source": self._get_name(s, subgraph, entities),
                "relation": rel,
                "target_id": t,
                "target": self._get_name(t, subgraph, entities),
                "direction": direction,
            })
        return items

    def paths_to_readable_struct(self, paths, subgraph, entities):
        return [self.path_to_readable_struct(path, subgraph, entities) for path in paths]

    def _resolve_edge_for_path_display(self, s, t, subgraph):

        if subgraph.has_edge(s, t):
            rel = subgraph[s][t].get("relation", "相关")
            return rel, True
        elif subgraph.has_edge(t, s):
            rel = subgraph[t][s].get("relation", "相关")
            return rel, False
        else:
            return "相关", True


    def enumerate_paths(self, subgraph: nx.DiGraph, seed_ids: Set[str]) -> List[List[str]]:

        all_paths = []
        seeds = [s for s in seed_ids if s in subgraph]
        ug = subgraph.to_undirected()
        per_seed = max(self.n_cand // max(len(seeds), 1), 2)

        for i, src in enumerate(seeds):

            for tgt in seeds[i + 1:]:
                try:
                    for p in islice(nx.all_simple_paths(ug, src, tgt, cutoff=self.max_len), per_seed):
                        all_paths.append(p)
                except nx.NetworkXError:
                    pass


            if len(all_paths) < self.n_cand:
                for p in islice(self._dfs(ug, src), per_seed):
                    if len(p) >= 2:
                        all_paths.append(p)


        seen = set()
        unique = []
        for p in all_paths:
            k = tuple(p)
            if k not in seen:
                seen.add(k)
                unique.append(p)

        return unique[:self.n_cand]

    def _dfs(self, g, src):
        stack = [(src, [src])]
        while stack:
            n, path = stack.pop()
            if len(path) >= 2:
                yield path
            if len(path) < self.max_len:
                for nb in g.neighbors(n):
                    if nb not in path:
                        stack.append((nb, path + [nb]))


    def _compute_anchor_bonus(self, path, seed_ids: Set[str]):

        if not path or not seed_ids:
            return 1.0

        hit = sum(1 for n in path if n in seed_ids)
        ratio = hit / max(len(path), 1)


        bonus = 1.0 + self.anchor_bonus_scale * ratio
        return bonus

    def _compute_query_path_relevance(self, paths, question, subgraph, entities):

        if self.encoder is None or not question:
            return None

        q_emb = np.asarray(self.encoder.encode_single(question), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)


        if q_norm < 1e-12:
            return [0.0 for _ in paths]

        relevances = []

        for path in paths:
            parts = []
            for i, nid in enumerate(path):
                name = self._get_name(nid, subgraph, entities)
                parts.append(name)
                if i < len(path) - 1:
                    next_nid = path[i + 1]
                    _, rel, _ = self._resolve_edge_direction(nid, next_nid, subgraph)
                    parts.append(rel)

            path_text = "，".join(parts)
            p_emb = np.asarray(self.encoder.encode_single(path_text), dtype=np.float32)
            p_norm = np.linalg.norm(p_emb)


            if p_norm < 1e-12:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(q_emb, p_emb) / (q_norm * p_norm))


            relevances.append(max(cos_sim, 0.0))

        return relevances

    def compute_path_scores(self, paths, node_features, query_vector,
                            seed_ids: Set[str] = None,
                            question: str = None, subgraph=None, entities=None):

        if not paths:
            return []

        reprs = []
        for path in paths:
            nodes = [n for n in path if n in node_features]
            if not nodes:
                reprs.append(torch.zeros_like(query_vector))
                continue

            embs = torch.stack([node_features[n] for n in nodes])
            w = F.softmax(torch.matmul(embs, query_vector), dim=0)
            reprs.append((w.unsqueeze(-1) * embs).sum(0))

        R = torch.stack(reprs)
        h_bar = R.mean(0)
        sigma = torch.sqrt(torch.sum((R - h_bar) ** 2, dim=-1) + 1e-8)
        sim = F.cosine_similarity(R, query_vector.unsqueeze(0), dim=-1)


        base_scores = sim * torch.exp(-self.lam * sigma)


        if seed_ids is not None and len(seed_ids) > 0:
            bonus = torch.tensor(
                [self._compute_anchor_bonus(p, seed_ids) for p in paths],
                dtype=base_scores.dtype,
                device=base_scores.device
            )
            scores = base_scores * bonus
        else:
            scores = base_scores

        relevances = self._compute_query_path_relevance(paths, question, subgraph, entities)
        if relevances is not None:

            rel_tensor = torch.tensor(relevances, dtype=scores.dtype, device=scores.device)

            rel_min, rel_max = rel_tensor.min(), rel_tensor.max()
            if rel_max - rel_min > 1e-8:
                rel_norm = (rel_tensor - rel_min) / (rel_max - rel_min)
            else:
                rel_norm = torch.ones_like(rel_tensor) * 0.5

            rel_factor = 1.0 + self.semantic_relevance_scale * (2 * rel_norm - 1)
            scores = scores * rel_factor

        scored = sorted(
            zip(paths, scores.detach().cpu().tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        return scored


    def select_topk_paths(self, paths, node_features, query_vector, subgraph, entities,
                          seed_ids: Set[str] = None, question: str = None):

        scored = self.compute_path_scores(
            paths, node_features, query_vector,
            seed_ids=seed_ids, question=question,
            subgraph=subgraph, entities=entities,
        )
        tops = [p for p, _ in scored[:self.topk]]
        si = self._build_input(tops, subgraph, entities)
        return tops, si


    def _build_input(self, paths, subgraph, entities):

        triples = set()
        for path in paths:
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                head, rel, tail = self._resolve_edge_direction(s, t, subgraph)
                triples.add((head, rel, tail))

        result = []
        for hid, rel, tid in triples:
            hi = entities.get(hid, {}) or dict(subgraph.nodes.get(hid, {}))
            ti = entities.get(tid, {}) or dict(subgraph.nodes.get(tid, {}))
            result.append({
                "head": hi.get("name", hid),
                "head_description": hi.get("description", ""),
                "relation": rel,
                "tail": ti.get("name", tid),
                "tail_description": ti.get("description", ""),
            })

        return result


def format_structured_input_for_llm(si: List[dict], question: str = "") -> str:

    MAX_TRIPLES = 6
    MAX_TOTAL_CHARS = 2600
    RELATED_DESC_MAX = 1000
    UNRELATED_DESC_MAX = 20

    question_norm = (question or "").replace(" ", "").strip()


    MIN_OVERLAP = 3

    def _is_query_related(entity_name: str) -> bool:

        name = (entity_name or "").replace(" ", "").strip()
        if len(name) < 2:
            return False

        if name in question_norm or question_norm in name:
            return True

        check_len = min(len(name), len(question_norm))
        for n in range(check_len, MIN_OVERLAP - 1, -1):
            for start in range(len(name) - n + 1):
                if name[start:start + n] in question_norm:
                    return True
        return False

    def _shorten(text: str, max_len: int) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text[:max_len] + "..." if len(text) > max_len else text

    si = si[:MAX_TRIPLES]
    lines = ["以下是从麻醉学知识图谱中检索到的相关知识路径：\n"]
    total_chars = 0

    for i, item in enumerate(si, 1):
        path_line = f"路径{i}: ({item['head']}) --[{item['relation']}]--> ({item['tail']})"
        lines.append(path_line)
        total_chars += len(path_line)

        head_desc = (item.get("head_description") or "").strip()
        if head_desc:
            head_related = _is_query_related(item["head"])
            limit = RELATED_DESC_MAX if head_related else UNRELATED_DESC_MAX
            if head_related or total_chars < MAX_TOTAL_CHARS:
                desc_text = f"  · {item['head']}: {_shorten(head_desc, limit)}"
                lines.append(desc_text)
                total_chars += len(desc_text)


        tail_desc = (item.get("tail_description") or "").strip()
        if tail_desc:
            tail_related = _is_query_related(item["tail"])
            limit = RELATED_DESC_MAX if tail_related else UNRELATED_DESC_MAX
            if tail_related or total_chars < MAX_TOTAL_CHARS:
                desc_text = f"  · {item['tail']}: {_shorten(tail_desc, limit)}"
                lines.append(desc_text)
                total_chars += len(desc_text)

    return "\n".join(lines)
# def format_structured_input_for_llm(si: List[dict]) -> str:
#
#
#     def _shorten(text: str, max_len: int = 120) -> str:
#         text = (text or "").strip().replace("\n", " ")
#         return text[:max_len] + "..." if len(text) > max_len else text
#
#
#     si = si[:6]
#
#     lines = ["以下是从麻醉学知识图谱中检索到的相关知识路径：\n"]
#     for i, item in enumerate(si, 1):
#         lines.append(f"路径{i}: ({item['head']}) --[{item['relation']}]--> ({item['tail']})")
#
#
#         if item.get("tail_description"):
#             lines.append(f"  · {item['tail']}: {_shorten(item['tail_description'], 120)}")
#         elif item.get("head_description"):
#             lines.append(f"  · {item['head']}: {_shorten(item['head_description'], 120)}")
#
#     return "\n".join(lines)