
import re
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import PATH_TOPK, UNCERTAINTY_LAMBDA, LLM_MAX_GEN_LENGTH


LLM_PATH_RANKING_PROMPT = """你是一位经验丰富的临床麻醉医师。请根据用户提出的临床问题，对以下从麻醉学知识图谱中检索到的候选推理路径进行**临床相关性评分**。

评分标准（1-10分）：
- 10分：路径直接回答了问题的核心要点，涉及关键药物交互、并发症机制等
- 7-9分：路径高度相关，涉及问题中的关键实体和临床逻辑
- 4-6分：路径部分相关，但未触及核心临床推理
- 1-3分：路径与问题关联薄弱，可能引入噪声

用户问题：{question}

候选路径：
{paths_text}

请严格按以下JSON格式输出每条路径的评分，不要输出其他内容：
{{"scores": [{example_scores}]}}
"""


class LLMPathRanker:


    def __init__(self, llm, topk=None, encoder=None):

        self.llm = llm
        self.topk = topk or PATH_TOPK
        self.encoder = encoder

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

    def _path_to_text(self, path, subgraph, entities):

        if not path:
            return ""
        if len(path) == 1:
            return self._get_name(path[0], subgraph, entities)

        pieces = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            head_id, rel, tail_id = self._resolve_edge_direction(s, t, subgraph)
            head_name = self._get_name(head_id, subgraph, entities)
            tail_name = self._get_name(tail_id, subgraph, entities)
            pieces.append(f"({head_name}) --[{rel}]--> ({tail_name})")
        return " → ".join(pieces)

    def _path_to_text_with_desc(self, path, subgraph, entities):

        if not path:
            return ""

        lines = []

        triples = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            head_id, rel, tail_id = self._resolve_edge_direction(s, t, subgraph)
            head_name = self._get_name(head_id, subgraph, entities)
            tail_name = self._get_name(tail_id, subgraph, entities)
            triples.append(f"({head_name}) --[{rel}]--> ({tail_name})")
        lines.append(" → ".join(triples))


        seen = set()
        for nid in path:
            if nid in seen:
                continue
            seen.add(nid)
            info = entities.get(nid, {})
            desc = info.get("description", "")
            name = info.get("name", self._get_name(nid, subgraph, entities))
            if desc and len(desc.strip()) > 5:
                short_desc = desc.strip()[:100]
                lines.append(f"  [{name}]: {short_desc}")

        return "\n".join(lines)

    def _build_ranking_prompt(self, question, paths, subgraph, entities):

        path_texts = []
        for i, path in enumerate(paths):
            text = self._path_to_text_with_desc(path, subgraph, entities)
            path_texts.append(f"路径{i+1}:\n{text}")

        paths_text = "\n\n".join(path_texts)
        example_scores = ", ".join(["0"] * len(paths))

        prompt = LLM_PATH_RANKING_PROMPT.format(
            question=question,
            paths_text=paths_text,
            example_scores=example_scores,
        )
        return prompt

    def _parse_llm_scores(self, response_text, num_paths):

        scores = []


        try:

            json_match = re.search(r'\{[^{}]*"scores"\s*:\s*\[([^\]]*)\][^{}]*\}', response_text)
            if json_match:
                parsed = json.loads(json_match.group(0))
                scores = [float(s) for s in parsed["scores"]]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        if not scores:
            numbers = re.findall(r'(?:路径\d+[：:]\s*)?(\d+(?:\.\d+)?)\s*分?', response_text)
            scores = [float(n) for n in numbers]


        if not scores:
            numbers = re.findall(r'(\d+(?:\.\d+)?)', response_text)
            scores = [float(n) for n in numbers if 1.0 <= float(n) <= 10.0]


        if len(scores) < num_paths:

            scores.extend([5.0] * (num_paths - len(scores)))
        scores = scores[:num_paths]


        scores = [max(0.0, min(s / 10.0, 1.0)) for s in scores]

        return scores

    def score_paths_by_llm(self, paths, question, subgraph, entities):

        if not paths:
            return []

        BATCH_SIZE = 8
        all_scored = []
        llm_fail_count = 0

        for batch_start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[batch_start: batch_start + BATCH_SIZE]

            prompt = self._build_ranking_prompt(question, batch_paths, subgraph, entities)

            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt="你是一位临床麻醉学专家，请严格按JSON格式输出评分。",
                    temperature=0.1,  # 低温度保证一致性
                    top_p=0.7,
                    max_new_tokens=256,
                )
                scores = self._parse_llm_scores(response, len(batch_paths))
            except Exception as e:
                llm_fail_count += 1
                if llm_fail_count <= 2:
                    print(f"  [LLMPathRanker] LLM 打分失败 (batch {batch_start}): {type(e).__name__}: {e}")
                scores = [0.5] * len(batch_paths)

            for path, score in zip(batch_paths, scores):
                all_scored.append((path, score))

        if llm_fail_count > 0:
            print(f"  [LLMPathRanker] 共 {llm_fail_count} 批次 LLM 调用失败，已用默认分数 0.5 填充")


        all_scored.sort(key=lambda x: x[1], reverse=True)
        return all_scored

    def select_topk_paths(self, paths, question, subgraph, entities, topk=None):

        k = topk or self.topk
        scored = self.score_paths_by_llm(paths, question, subgraph, entities)
        tops = [p for p, _ in scored[:k]]
        si = self._build_input(tops, subgraph, entities)
        return tops, si

    def _build_input(self, paths, subgraph, entities):
        """构建结构化输入（复用 PathRanker 的逻辑）"""
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


class CascadePathRanker:

    def __init__(self, gnn_ranker, llm_ranker, topk=None, cascade_factor=2):

        self.gnn_ranker = gnn_ranker
        self.llm_ranker = llm_ranker
        self.topk = topk or PATH_TOPK
        self.cascade_factor = cascade_factor

    def select_topk_gnn_then_llm(self, paths, node_features, query_vector,
                                  subgraph, entities, question,
                                  seed_ids=None):

        if not paths:
            return [], []


        pre_topk = self.topk * self.cascade_factor
        scored_by_gnn = self.gnn_ranker.compute_path_scores(
            paths, node_features, query_vector,
            seed_ids=seed_ids, question=question,
            subgraph=subgraph, entities=entities,
        )
        gnn_top_paths = [p for p, _ in scored_by_gnn[:pre_topk]]


        scored_by_llm = self.llm_ranker.score_paths_by_llm(
            gnn_top_paths, question, subgraph, entities
        )
        tops = [p for p, _ in scored_by_llm[:self.topk]]

        si = self.llm_ranker._build_input(tops, subgraph, entities)
        return tops, si

    def select_topk_llm_then_gnn(self, paths, node_features, query_vector,
                                  subgraph, entities, question,
                                  seed_ids=None):

        if not paths:
            return [], []


        pre_topk = self.topk * self.cascade_factor
        scored_by_llm = self.llm_ranker.score_paths_by_llm(
            paths, question, subgraph, entities
        )
        llm_top_paths = [p for p, _ in scored_by_llm[:pre_topk]]


        scored_by_gnn = self.gnn_ranker.compute_path_scores(
            llm_top_paths, node_features, query_vector,
            seed_ids=seed_ids, question=question,
            subgraph=subgraph, entities=entities,
        )
        tops = [p for p, _ in scored_by_gnn[:self.topk]]

        si = self.gnn_ranker._build_input(tops, subgraph, entities)
        return tops, si
