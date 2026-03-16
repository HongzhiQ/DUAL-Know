import math
import re
import sys
import os
from typing import List, Dict, Tuple, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import FUSION_WEIGHT_CONF, FUSION_WEIGHT_OVERLAP, FUSION_WEIGHT_SIM
from bert_score import score as bertscore_score


class MultiMetricFusion:
    def __init__(
        self,
        encoder=None,
        w1: Optional[float] = None,
        w2: Optional[float] = None,
        w3: Optional[float] = None,
        bert_lang: str = "zh",
        bert_model_type: Optional[str] = None,
        use_idf: bool = False,
        conf_center: float = -0.25,
        conf_scale: float = 10.0,
        enable_brevity_guard: bool = False,
        brevity_ratio_threshold: float = 0.60,
        brevity_penalty: float = 0.10,
    ):

        self.encoder = encoder
        self.w1 = float(w1 if w1 is not None else FUSION_WEIGHT_CONF)
        self.w2 = float(w2 if w2 is not None else FUSION_WEIGHT_OVERLAP)
        self.w3 = float(w3 if w3 is not None else FUSION_WEIGHT_SIM)

        self.bert_lang = bert_lang
        self.bert_model_type = bert_model_type
        self.use_idf = use_idf

        self.conf_center = float(conf_center)
        self.conf_scale = float(conf_scale)

        self.enable_brevity_guard = bool(enable_brevity_guard)
        self.brevity_ratio_threshold = float(brevity_ratio_threshold)
        self.brevity_penalty = float(brevity_penalty)


    def _safe_text(self, text: Optional[str]) -> str:
        return (text or "").strip()

    def _normalize_text(self, text: Optional[str]) -> str:
        text = self._safe_text(text).lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，。；：、！？,.!?;:()\[\]{}“”\"'‘’《》<>【】\-—_]", "", text)
        return text


    def _confidence(self, avg_logprob: float) -> float:
        x = float(avg_logprob)
        z = self.conf_scale * (x - self.conf_center)
        z = max(min(z, 60.0), -60.0)
        return float(1.0 / (1.0 + math.exp(-z)))


    def _overlap(self, answer: str, entities: List[str]) -> float:
        if not entities:
            return 0.0

        answer_raw = self._safe_text(answer)
        answer_norm = self._normalize_text(answer_raw)

        hits = 0
        valid_entities = 0

        for e in entities:
            e_raw = self._safe_text(e)
            if not e_raw:
                continue
            valid_entities += 1

            e_norm = self._normalize_text(e_raw)

            if (e_raw in answer_raw) or (e_norm and e_norm in answer_norm):
                hits += 1

        if valid_entities == 0:
            return 0.0
        return float(hits / valid_entities)


    def _sim(self, answer: str, query: str) -> float:
        answer = self._safe_text(answer)
        query = self._safe_text(query)

        if not answer or not query:
            return 0.0

        if self.encoder is None:
            return 0.5

        try:
            a = np.array(self.encoder.encode_single(answer), dtype=np.float32)
            q = np.array(self.encoder.encode_single(query), dtype=np.float32)

            na = np.linalg.norm(a)
            nq = np.linalg.norm(q)
            if na < 1e-8 or nq < 1e-8:
                return 0.0

            sim = float(np.dot(a, q) / (na * nq))
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.5


    # def _sim(self, answer: str, query: str) -> float:
    #     """
    #
    #     Sim(i) = BERTScore(A(i), q)
    #     """
    #     answer = self._safe_text(answer)
    #     query = self._safe_text(query)
    #
    #     if not answer or not query:
    #         return 0.0
    #
    #     try:
    #         P, R, F1 = bertscore_score(
    #             [answer],
    #             [query],
    #             lang=self.bert_lang if self.bert_model_type is None else None,
    #             model_type=self.bert_model_type,
    #             verbose=False,
    #             idf=self.use_idf,
    #             batch_size=1,
    #             device=None,
    #         )
    #         val = float(F1[0].item())
    #         return float(max(0.0, min(1.0, val)))
    #     except Exception as e:
    #         print(f"[Fusion] BERTScore 计算失败，回退为 0.5: {e}")
    #         return 0.5


    def _brevity_penalty_for_rag(self, answer_llm: str, answer_rag: str) -> float:
        if not self.enable_brevity_guard:
            return 0.0

        llm_len = len(self._safe_text(answer_llm))
        rag_len = len(self._safe_text(answer_rag))

        if llm_len <= 0:
            return 0.0

        ratio = rag_len / max(llm_len, 1)
        if ratio < self.brevity_ratio_threshold:
            return self.brevity_penalty
        return 0.0


    def select_answer(
        self,
        query: str,
        answer_llm: str,
        answer_rag: str,
        lp_llm: float,
        lp_rag: float,
        entities: List[str]
    ) -> Tuple[str, Dict]:

        conf_raw = [float(lp_llm), float(lp_rag)]
        conf = [
            self._confidence(conf_raw[0]),
            self._confidence(conf_raw[1]),
        ]


        overlap_raw = [
            self._overlap(answer_llm, entities),
            self._overlap(answer_rag, entities),
        ]
        overlap = overlap_raw[:]


        sim_raw = [
            self._sim(answer_llm, query),
            self._sim(answer_rag, query),
        ]
        sim = sim_raw[:]


        score_llm = self.w1 * conf[0] + self.w2 * overlap[0] + self.w3 * sim[0]
        score_rag = self.w1 * conf[1] + self.w2 * overlap[1] + self.w3 * sim[1]


        rag_penalty = self._brevity_penalty_for_rag(answer_llm, answer_rag)
        if rag_penalty > 0:
            score_rag -= rag_penalty

        selected = "RAG" if score_rag >= score_llm else "LLM"
        final_answer = answer_rag if selected == "RAG" else answer_llm

        detail = {
            "selected": selected,
            "score_llm": round(float(score_llm), 4),
            "score_rag": round(float(score_rag), 4),
            "confidence": {
                "llm_raw": round(conf_raw[0], 4),
                "rag_raw": round(conf_raw[1], 4),
                "llm_used": round(conf[0], 4),
                "rag_used": round(conf[1], 4),
                "mapping": "sigmoid(avg_logprob)",
                "center": self.conf_center,
                "scale": self.conf_scale,
            },
            "retrieval_consistency": {
                "llm_raw": round(overlap_raw[0], 4),
                "rag_raw": round(overlap_raw[1], 4),
                "llm_used": round(overlap[0], 4),
                "rag_used": round(overlap[1], 4),
                "matching": "substring_or_normalized_substring",
            },
            "semantic_coherence": {
                "llm_raw": round(sim_raw[0], 4),
                "rag_raw": round(sim_raw[1], 4),
                "llm_used": round(sim[0], 4),
                "rag_used": round(sim[1], 4),
                "metric": "BGE-cosine",
                # 若你切到上面的 BERTScore 版本，这里改成 "BERTScore-F1"
            },
            "weights": {
                "confidence": self.w1,
                "retrieval_consistency": self.w2,
                "semantic_coherence": self.w3,
            },
            "engineering_enhancement": {
                "brevity_guard_enabled": self.enable_brevity_guard,
                "rag_brevity_penalty": round(float(rag_penalty), 4),
            }
        }

        return final_answer, detail