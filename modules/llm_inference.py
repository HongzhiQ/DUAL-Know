import os
import sys
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_wrapper import AnesGLM
from modules.query_augmentation import QueryAugmentor, TripleExtractor, QueryAugmentorAndExtractor
from modules.path_ranking import format_structured_input_for_llm
from configs.config import (
    ANESGLM_MODEL_PATH,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_GEN_LENGTH,
)


DEFAULT_SYSTEM_PROMPT = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。"

DIRECT_QA_PROMPT = """请尽量全面详细的回答用户问题，先下定义,再列举治疗手段,分点写,适当展开。：

{question}
"""

RAG_QA_PROMPT = """你是一位经验丰富的临床麻醉医师。请根据以下从知识图谱中检索到的证据链来回答问题。
要求：
1. 若知识图谱三元组不符合原问题则不要使用；
2. 请尽量参考相关的实体的描述进行回答，与实体描述类似，尽量全面介绍；
3. 若证据不足，可结合医学常识进行回答，但不要编造，请全面详细。
4. 先下定义,再列举治疗手段,分点写,适当展开。

{knowledge_paths}

问题：{question}
请回答：
"""


class LLMInference:

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or ANESGLM_MODEL_PATH
        self.device = device
        self.llm = AnesGLM(self.model_path, device=self.device)
        self.query_augmentor = None
        self.triple_extractor = None
        self.combined_augmentor = None
        self._initialized = False

    def load(self):
        if self._initialized:
            return self

        self.llm.load()
        self.query_augmentor = QueryAugmentor(self.llm)
        self.triple_extractor = TripleExtractor(self.llm)
        self.combined_augmentor = QueryAugmentorAndExtractor(self.llm)
        self._initialized = True
        return self


    def generate(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self.load()
        return self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or LLM_TEMPERATURE,
            top_p=top_p or LLM_TOP_P,
            max_new_tokens=max_new_tokens or LLM_MAX_GEN_LENGTH,
        )

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, float]:
        self.load()
        return self.llm.generate_with_logprobs(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or LLM_TEMPERATURE,
            top_p=top_p or LLM_TOP_P,
            max_new_tokens=max_new_tokens or LLM_MAX_GEN_LENGTH,
        )


    def rewrite_and_extract(
        self, question: str, num_rewrites: Optional[int] = None
    ) -> Tuple[List[str], List[Dict[str, str]]]:

        self.load()
        return self.combined_augmentor.rewrite_and_extract(question, num_rewrites)


    def rewrite_query(self, question: str, num_rewrites: Optional[int] = None) -> List[str]:
        self.load()
        return self.query_augmentor.rewrite(question, num_rewrites=num_rewrites)


    def extract_triples_from_query(self, query: str) -> List[Dict[str, str]]:
        self.load()
        return self.triple_extractor.extract_from_query(query)

    def extract_triples_from_query_set(self, query_set: List[str]) -> List[Dict[str, str]]:
        self.load()
        return self.triple_extractor.extract_from_query_set(query_set)


    def generate_direct_answer(
        self,
        question: str,
        return_logprobs: bool = True,
    ):
        prompt = DIRECT_QA_PROMPT.format(question=question).strip()
        if return_logprobs:
            return self.generate_with_logprobs(prompt)
        return self.generate(prompt)

    def generate_rag_answer(
        self,
        question: str,
        structured_input: List[Dict],
        return_logprobs: bool = True,
    ):
        if structured_input:
            knowledge_text = format_structured_input_for_llm(structured_input, question=question)
        else:
            knowledge_text = "未检索到明确知识路径。"

        prompt = RAG_QA_PROMPT.format(
            knowledge_paths=knowledge_text,
            question=question,
        ).strip()

        if return_logprobs:
            return self.generate_with_logprobs(prompt)
        return self.generate(prompt)


    def generate_dual_answers_parallel(
        self,
        question: str,
        structured_input: Optional[List[Dict]] = None,
    ) -> Dict:

        self.load()

        if not structured_input:
            answer_llm, logprob_llm = self.generate_direct_answer(question, return_logprobs=True)
            return {
                "answer_llm": answer_llm,
                "logprob_llm": float(logprob_llm),
                "answer_rag": answer_llm,
                "logprob_rag": float(logprob_llm),
            }


        with ThreadPoolExecutor(max_workers=2) as executor:
            future_llm = executor.submit(
                self.generate_direct_answer, question, True
            )
            future_rag = executor.submit(
                self.generate_rag_answer, question, structured_input, True
            )


            answer_llm, logprob_llm = future_llm.result()
            answer_rag, logprob_rag = future_rag.result()

        return {
            "answer_llm": answer_llm,
            "logprob_llm": float(logprob_llm),
            "answer_rag": answer_rag,
            "logprob_rag": float(logprob_rag),
        }


    def generate_dual_answers(
        self,
        question: str,
        structured_input: Optional[List[Dict]] = None,
    ) -> Dict:
        self.load()

        answer_llm, logprob_llm = self.generate_direct_answer(
            question=question,
            return_logprobs=True,
        )

        if structured_input:
            answer_rag, logprob_rag = self.generate_rag_answer(
                question=question,
                structured_input=structured_input,
                return_logprobs=True,
            )
        else:
            answer_rag, logprob_rag = answer_llm, logprob_llm

        return {
            "answer_llm": answer_llm,
            "logprob_llm": float(logprob_llm),
            "answer_rag": answer_rag,
            "logprob_rag": float(logprob_rag),
        }


