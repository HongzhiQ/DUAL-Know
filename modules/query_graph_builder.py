import os
import json
import re
import networkx as nx
from typing import Dict, List


class QueryGraphBuilder:


    def __init__(
        self,
        encoder,
        entities: Dict[str, dict],
        name_index=None,
        score_thres: float = 0.62,
        strong_score_thres: float = 0.82,
        min_lexical_overlap: float = 0.35,
        link_topk: int = 5,
        direct_seed_topk: int = 2,
        alignment_dir: str = None,
    ):

        self.encoder = encoder
        self.entities = entities
        self.name_index = name_index
        self.score_thres = score_thres
        self.strong_score_thres = strong_score_thres
        self.min_lexical_overlap = min_lexical_overlap
        self.link_topk = link_topk
        self.direct_seed_topk = direct_seed_topk
        self.alignment_dir = alignment_dir

        self._name_to_eids: Dict[str, List[str]] = {}
        for eid, einfo in entities.items():
            raw_name = (einfo.get("name", "") or "").strip()
            if not raw_name:
                continue

            for alias in self._expand_alias_forms(raw_name):
                self._name_to_eids.setdefault(alias, []).append(eid)


        self.alignment_map = self._load_alignment_map(alignment_dir)


    @staticmethod
    def _norm(x: str) -> str:
        return (x or "").strip().lower()

    def _expand_alias_forms(self, name: str) -> List[str]:

        name = (name or "").strip()
        if not name:
            return []

        forms = []
        raw = name
        norm = self._norm(name)

        for x in [raw, norm]:
            if x and x not in forms:
                forms.append(x)

        return forms

    def _load_alignment_map(self, alignment_dir: str) -> Dict[str, str]:

        amap: Dict[str, str] = {}

        if not alignment_dir or not os.path.isdir(alignment_dir):
            return amap

        for fn in os.listdir(alignment_dir):
            if not fn.endswith(".json"):
                continue

            fp = os.path.join(alignment_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                if not isinstance(obj, dict):
                    continue

                for k, v in obj.items():
                    alias_key = (str(k).strip() if k is not None else "")
                    canonical = (str(v).strip() if v is not None else "")

                    if not alias_key or not canonical:
                        continue

                    for form in self._expand_alias_forms(alias_key):
                        amap[form] = canonical

                    if "^" in alias_key:
                        for part in alias_key.split("^"):
                            part = part.strip()
                            if not part:
                                continue
                            for form in self._expand_alias_forms(part):
                                amap[form] = canonical

            except Exception:
                continue

        return amap

    def _canonicalize_name(self, name: str) -> str:

        name = (name or "").strip()
        if not name:
            return name

        if name in self.alignment_map:
            return self.alignment_map[name]

        norm = self._norm(name)
        if norm in self.alignment_map:
            return self.alignment_map[norm]

        return name

    def _dedup_triples(self, triples: List[Dict]) -> List[Dict]:

        out, seen = [], set()
        for t in triples:
            h = (t.get("head", "") or "").strip()
            r = (t.get("relation", "") or "").strip()
            ta = (t.get("tail", "") or "").strip()
            if not h or not r:
                continue
            key = (h, r, ta)
            if key not in seen:
                seen.add(key)
                out.append({"head": h, "relation": r, "tail": ta})
        return out


    def _clean_query_entity_name(self, name: str) -> str:

        x = (name or "").strip()
        if not x:
            return x

        # 去掉常见句式前缀
        prefixes = [
            "对患者进行",
            "对病人进行",
            "对患者做",
            "对病人做",
            "进行",
            "用于",
            "为了",
            "主要任务是",
            "主要目的是",
            "其目的是",
            "目的在于",
        ]
        for p in prefixes:
            if x.startswith(p):
                x = x[len(p):].strip()

        x = x.replace("识别和减少", "")
        x = x.replace("识别并减少", "")
        x = x.replace("识别和降低", "")
        x = x.replace("识别并降低", "")
        x = x.replace("的风险", "风险")
        x = x.replace("围术期并发症风险", "围术期并发症风险")


        x = x.strip("，。；;、 ")

        return x

    def _generate_link_queries(self, original_name: str, canonical_name: str) -> List[str]:

        variants = []

        def add(x: str):
            x = (x or "").strip()
            if x and x not in variants:
                variants.append(x)

        add(original_name)
        add(canonical_name)

        cleaned_original = self._clean_query_entity_name(original_name)
        cleaned_canonical = self._clean_query_entity_name(canonical_name)
        add(cleaned_original)
        add(cleaned_canonical)


        for src in [cleaned_original, cleaned_canonical]:
            if "、" in src or "，" in src or "," in src:
                parts = re.split(r"[、，,]", src)
                for p in parts[:3]:
                    p = p.strip(" 等及和与或;； ")
                    if len(p) >= 2:
                        add(p)

        return variants

    def _char_overlap_score(self, a: str, b: str) -> float:

        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0

        if a == b:
            return 1.0
        if a in b or b in a:
            return min(len(a), len(b)) / max(len(a), len(b))

        sa = set(a)
        sb = set(b)
        inter = len(sa & sb)
        if inter == 0:
            return 0.0
        return inter / max(min(len(sa), len(sb)), 1)

    def _link_entity(self, name: str) -> dict:

        original_name = (name or "").strip()
        canonical_name = self._canonicalize_name(original_name)
        link_queries = self._generate_link_queries(original_name, canonical_name)

        candidate_ids: List[str] = []
        candidate_scores: List[float] = []
        candidate_sources: List[str] = []


        exact_hits = []
        for q in link_queries:
            for form in self._expand_alias_forms(q):
                for eid in self._name_to_eids.get(form, []):
                    exact_hits.append((eid, 1.0, "exact"))

        best_exact = {}
        for eid, score, src in exact_hits:
            if eid not in best_exact or score > best_exact[eid][0]:
                best_exact[eid] = (score, src)

        if best_exact:
            for eid, (score, src) in sorted(best_exact.items(), key=lambda x: x[1][0], reverse=True):
                candidate_ids.append(eid)
                candidate_scores.append(score)
                candidate_sources.append(src)


        if not candidate_ids:
            best_fuzzy = {}

            def _vector_search(query_text: str):
                local = []
                if self.name_index is None or not query_text.strip():
                    return local
                q_emb = self.encoder.encode_single(query_text)
                results = self.name_index.search(q_emb, self.link_topk)
                for eid, score in results:
                    score = float(score)
                    if score < self.score_thres:
                        continue

                    cand_name = (self.entities[eid].get("name", "") or "").strip()

                    lex = max(self._char_overlap_score(v, cand_name) for v in link_queries)


                    if not (lex >= self.min_lexical_overlap or score >= self.strong_score_thres):
                        continue


                    final_score = 0.8 * score + 0.2 * lex
                    local.append((eid, final_score, score, lex, query_text))

                return local

            fuzzy_results = []
            for q in link_queries:
                fuzzy_results.extend(_vector_search(q))

            for eid, final_score, raw_score, lex, src_query in fuzzy_results:
                if eid not in best_fuzzy or final_score > best_fuzzy[eid][0]:
                    best_fuzzy[eid] = (final_score, raw_score, lex, src_query)

            for eid, (final_score, raw_score, lex, src_query) in sorted(
                best_fuzzy.items(), key=lambda x: x[1][0], reverse=True
            )[:self.link_topk]:
                candidate_ids.append(eid)
                candidate_scores.append(final_score)
                candidate_sources.append(f"fuzzy:{src_query}|raw={raw_score:.3f}|lex={lex:.3f}")


        pairs = sorted(zip(candidate_ids, candidate_scores, candidate_sources), key=lambda x: x[1], reverse=True)
        pairs = pairs[:self.link_topk]
        candidate_ids = [x[0] for x in pairs]
        candidate_scores = [x[1] for x in pairs]
        candidate_sources = [x[2] for x in pairs]


        types = []
        resolved_type = "未知"

        if candidate_ids:
            for eid in candidate_ids:
                t = (self.entities[eid].get("type", "") or "未知").strip()
                if t and t not in types:
                    types.append(t)

            if not types:
                types = ["未知"]

            best_eid = candidate_ids[0]
            resolved_type = (self.entities[best_eid].get("type", "") or "未知").strip() or "未知"
        else:
            types = ["未知"]

        return {
            "original_name": original_name,
            "canonical_name": canonical_name,
            "link_queries": link_queries,
            "candidate_ids": candidate_ids,
            "candidate_scores": candidate_scores,
            "candidate_sources": candidate_sources,
            "types": types,
            "resolved_type": resolved_type,
            "linked": len(candidate_ids) > 0,
        }


    def build(self, query_triples: List[Dict], question: str = None, query_set: List[str] = None):

        triples = self._dedup_triples(query_triples)

        Gq = nx.DiGraph()
        Gq.graph["question"] = question
        Gq.graph["query_set"] = query_set

        node_cache: Dict[str, dict] = {}


        node_roles: Dict[str, str] = {}

        for t in triples:
            h, r, ta = t["head"], t["relation"], t["tail"]

            if h not in node_cache:
                info = self._link_entity(h)
                node_cache[h] = {"name": h, **info}
                Gq.add_node(h, **node_cache[h])

            node_roles[h] = "head"

            if ta:
                if ta not in node_cache:
                    info = self._link_entity(ta)
                    node_cache[ta] = {"name": ta, **info}
                    Gq.add_node(ta, **node_cache[ta])


                if node_roles.get(ta) != "head":
                    node_roles[ta] = "tail"

                Gq.add_edge(h, ta, relation=r)
            else:

                Gq.graph.setdefault("query_slots", []).append({
                    "head": h,
                    "relation": r,
                    "tail": ""
                })


        qe = []
        for n, d in Gq.nodes(data=True):
            qe.append({
                "name": d["canonical_name"] if d["canonical_name"] else d["name"],
                "original_name": d["original_name"],
                "canonical_name": d["canonical_name"],
                "types": d["types"],
                "resolved_type": d["resolved_type"],
                "candidate_ids": d["candidate_ids"][:self.direct_seed_topk],
                "candidate_scores": d.get("candidate_scores", [])[:self.direct_seed_topk],
                "linked": d["linked"],

                "role": node_roles.get(n, "head"),
            })

        return qe, Gq