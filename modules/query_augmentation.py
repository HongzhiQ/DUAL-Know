import json
import re
from typing import List, Dict, Tuple
import sys, os

from configs.config import (
    NUM_QUERY_REWRITES,
    REWRITE_TEMPERATURE,
    REWRITE_TOP_P,
    TRIPLE_TEMPERATURE,
    TRIPLE_TOP_P,
)


COMBINED_REWRITE_AND_EXTRACT_PROMPT = """你是麻醉学知识检索助手，同时也是知识图谱构建助手。
请你对下面的【原始问题】完成两个任务，并以 JSON 格式输出结果。

## 任务一：查询改写
围绕原始问题的核心意图，生成 {num_rewrites} 个不同表达的改写问题。
要求：
1. 严格保持原问题的问法意图不变，不能把"作用"改写成"用途/适应证"，不能把"机制"改写成"并发症"。
2. 允许换说法、换表述，但不能引入新的医学关注点。
3. 若原问题是"属性型问题"（如作用、机制、特点、目的、任务、适应证、副作用等），改写后仍必须保持为同一属性槽位的问题。

## 任务二：三元组抽取
从原始问题及你生成的改写问题中，综合抽取最能表达查询意图的知识三元组。
要求：
1. 只抽取与问题核心意图直接相关的三元组，至少抽取 2 个；
2. head 必须是具体的医学实体或概念，不能为空；
3. relation 必须是明确的医学关系（如药物分类、适应证、副作用、作用机制、治疗、定义、包括、用于等）；
4. tail 可以为空字符串（属性槽位型问题），也可以填写你认为最合理的答案实体。

## 输出格式（严格 JSON，不要输出任何多余文字）：
{{
  "rewrites": ["改写问题1", "改写问题2", "改写问题3"],
  "triples": [
    {{"head": "...", "relation": "...", "tail": "..."}},
    {{"head": "...", "relation": "...", "tail": ""}}
  ]
}}

原始问题：
{question}"""



class QueryAugmentorAndExtractor:

    INTENT_MAP = {
        "作用": "主要作用", "功效": "主要作用", "效果": "主要作用",
        "机制": "作用机制", "原理": "作用机制",
        "定义": "定义", "是什么": "定义", "是啥": "定义",
        "适应": "适应证", "用于": "适应证", "应用": "适应证",
        "禁忌": "禁忌证",
        "副作用": "副作用", "不良反应": "副作用",
        "并发症": "并发症", "风险": "并发症",
        "剂量": "用法用量", "用量": "用法用量", "怎么用": "用法用量",
        "注意": "注意事项", "注意事项": "注意事项",
        "区别": "比较", "对比": "比较", "比较": "比较",
        "包括": "包括", "包含": "包括", "哪些": "包括",
        "步骤": "操作流程", "流程": "操作流程", "怎么做": "操作流程",
        "特点": "特点", "优势": "特点", "优点": "特点",
        "任务": "主要任务", "职责": "主要任务", "目的": "目的",
        "治疗": "治疗方法", "处理": "治疗方法",
    }

    def __init__(self, llm):
        self.llm = llm

    def rewrite_and_extract(
        self, question: str, num_rewrites: int = None
    ) -> Tuple[List[str], List[Dict[str, str]]]:

        num_rewrites = num_rewrites or NUM_QUERY_REWRITES
        prompt = COMBINED_REWRITE_AND_EXTRACT_PROMPT.format(
            num_rewrites=num_rewrites, question=question
        )


        temperature = (REWRITE_TEMPERATURE + TRIPLE_TEMPERATURE) / 2  # 0.4
        top_p = (REWRITE_TOP_P + TRIPLE_TOP_P) / 2                    # 0.8

        response = self.llm.generate(
            prompt,
            system_prompt=(
                "你是麻醉学知识检索与知识图谱构建助手。"
                "请严格按要求的 JSON 格式输出，不要输出任何多余文字。"
            ),
            temperature=temperature,
            top_p=top_p,
        )

        rewrites, triples = self._parse_combined(response, num_rewrites, question)


        query_set = [question] + rewrites


        if len(triples) < 2:
            fallback = self._fallback_extract([question] + rewrites)
            seen = set((t["head"], t["relation"], t["tail"]) for t in triples)
            for t in fallback:
                key = (t["head"], t["relation"], t["tail"])
                if key not in seen:
                    seen.add(key)
                    triples.append(t)

        return query_set, triples


    def _parse_combined(
        self, response: str, expected_rewrites: int, original_question: str
    ) -> Tuple[List[str], List[Dict[str, str]]]:

        response = response.strip()
        data = None


        try:
            data = json.loads(response)
        except Exception:
            pass


        if data is None:
            try:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(response[start:end + 1])
            except Exception:
                pass


        if data is None:
            rewrites = self._fallback_parse_rewrites(response, expected_rewrites, original_question)
            triples = self._fallback_parse_triples(response)
            return rewrites, triples


        raw_rewrites = data.get("rewrites", [])
        raw_triples = data.get("triples", [])


        rewrites = []
        seen = set()
        for r in raw_rewrites:
            r = str(r).strip()
            r = re.sub(r"^[\d]+[.、)）]\s*", "", r).strip()
            if r and len(r) > 3 and r != original_question and r not in seen:
                seen.add(r)
                rewrites.append(r)
        rewrites = rewrites[:expected_rewrites]


        triples = self._normalize_triples(raw_triples)

        return rewrites, triples

    def _normalize_triples(self, data: list) -> List[Dict[str, str]]:
        triples, seen = [], set()
        for item in data:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "") or "").strip()
            rel = str(item.get("relation", "") or "").strip()
            tail = str(item.get("tail", "") or "").strip()
            if not head or not rel:
                continue
            key = (head, rel, tail)
            if key not in seen:
                seen.add(key)
                triples.append({"head": head, "relation": rel, "tail": tail})
        return triples


    def _fallback_parse_rewrites(
        self, response: str, expected: int, original_question: str
    ) -> List[str]:
        lines = []
        seen = set()
        for line in response.strip().split("\n"):
            line = re.sub(r"^[\d]+[.、)）]\s*", "", line.strip())
            line = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", line)
            line = re.sub(r"^[-•·]\s*", "", line).strip()
            if (
                line
                and len(line) > 3
                and line != original_question
                and line not in seen
                and not line.startswith("{")
                and not line.startswith("[")
            ):
                seen.add(line)
                lines.append(line)
        return lines[:expected]

    def _fallback_parse_triples(self, response: str) -> List[Dict[str, str]]:
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(response[start:end + 1])
                if isinstance(data, list):
                    return self._normalize_triples(data)
        except Exception:
            pass
        return []


    def _fallback_extract(self, queries: List[str]) -> List[Dict[str, str]]:
        fallback_triples = []
        seen = set()
        for q in queries:
            entities = self._extract_entities_from_question(q)
            relations = []
            for keyword, rel in self.INTENT_MAP.items():
                if keyword in q:
                    relations.append(rel)
            if not relations:
                relations = ["相关"]
            for ent in entities:
                for rel in relations:
                    key = (ent, rel, "")
                    if key not in seen:
                        seen.add(key)
                        fallback_triples.append({"head": ent, "relation": rel, "tail": ""})
        return fallback_triples

    @staticmethod
    def _extract_entities_from_question(question: str) -> List[str]:
        entities = []
        q = question.strip().rstrip("？?。.！!")
        if "的" in q:
            parts = q.split("的")
            head = parts[0].strip()
            if 2 <= len(head) <= 20:
                entities.append(head)
        for sep in ["和", "与", "还是", "以及", "，", ","]:
            if sep in q:
                for part in q.split(sep):
                    part = part.strip()
                    part = re.sub(r'(的主要|的具体|的常见|是什么|是啥|有哪些|怎么样|如何|怎么|吗|呢).*$', '', part).strip()
                    if 2 <= len(part) <= 15:
                        entities.append(part)
        if not entities:
            cleaned = re.sub(r'(是什么|是啥|有哪些|怎么样|如何|怎么|吗|呢|的主要|的具体|的常见).*$', '', q).strip()
            if 2 <= len(cleaned) <= 20:
                entities.append(cleaned)
        unique = []
        seen = set()
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique



QUERY_REWRITE_PROMPT = """你是麻醉学知识检索助手。
请围绕【原始问题的核心意图】生成 {num_rewrites} 个不同表达的改写问题。

非常重要：
1. 必须严格保持原问题的问法意图不变，不能把"作用"改写成"用途/适应证"，不能把"机制"改写成"并发症"，不能把"任务"改写成"角色"以外无关方向。
2. 允许换说法、换表述，但不能引入新的医学关注点。
3. 若原问题是"属性型问题"（如：作用、机制、特点、目的、任务、适应证、副作用等），改写后仍必须保持为同一属性槽位的问题。
4. 输出 {num_rewrites} 个编号句子（如 1. 2. 3.），不要解释，不要多余文字。

原始问题：
{question}
"""

TRIPLE_EXTRACTION_PROMPT = """你是麻醉学知识图谱构建助手。
请从下面的【问题集合】中，综合抽取最能表达查询意图的知识三元组。

要求：
1. 只抽取与这些问题核心意图直接相关的三元组，每个问题尽量抽取 2 个或以上三元组；
2. head 必须是具体的医学实体或概念，不能为空；
3. relation 必须是明确的医学关系，如：药物分类、适应证、副作用、作用机制、治疗、定义、包括、用于、导致、比较、特点等；
4. tail 可以为空：
   - 如果问题属于"属性槽位型问题"（如"X 的主要作用是什么""X 的治疗方法有哪些""X 的定义是什么"），
     且无法稳定确定具体答案实体，可以令 tail 为空字符串；
   - 如果你能够根据医学知识补出一个合理且高相关的 tail，也可以填写；
5. 优先保证三元组"贴近原问题意图"，不要为了凑 tail 而随意猜测不稳定答案；
6. 仅输出一个 JSON 数组，每项格式严格为：
   {{"head": "...", "relation": "...", "tail": "..."}}
7. 务必保证 JSON 合法，不要输出任何解释文字。

下面给出一些抽取风格示例（仅作风格参考，不要照抄）：

- 问题：丙泊酚的主要作用是什么？
  输出：
  [
    {{"head":"丙泊酚","relation":"主要作用","tail":"镇静"}},
    {{"head":"丙泊酚","relation":"作用机制","tail":"GABA受体激动"}}
  ]
  或者：
  [
    {{"head":"丙泊酚","relation":"主要作用","tail":""}},
    {{"head":"丙泊酚","relation":"作用机制","tail":""}}
  ]

- 问题：麻醉前评估门诊的主要任务是什么？
  输出：
  [
    {{"head":"麻醉前评估门诊","relation":"主要任务","tail":"术前风险评估"}},
    {{"head":"麻醉前评估门诊","relation":"主要任务","tail":"麻醉相关准备"}}
  ]
  或者：
  [
    {{"head":"麻醉前评估门诊","relation":"主要任务","tail":""}},
    {{"head":"麻醉前评估门诊","relation":"包括","tail":"术前评估"}}
  ]

- 问题：术前评估包括哪些内容？
  输出：
  [
    {{"head":"术前评估","relation":"包括","tail":"病史采集"}},
    {{"head":"术前评估","relation":"包括","tail":"体格检查"}},
    {{"head":"术前评估","relation":"包括","tail":"实验室检查"}}
  ]

- 问题：硬膜外麻醉和腰麻有什么区别？
  输出：
  [
    {{"head":"硬膜外麻醉","relation":"比较","tail":"腰麻"}},
    {{"head":"硬膜外麻醉","relation":"操作部位","tail":"硬膜外腔"}},
    {{"head":"腰麻","relation":"操作部位","tail":"蛛网膜下腔"}}
  ]

问题集合：
{queries}"""


class QueryAugmentor:


    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str, num_rewrites: int = None) -> List[str]:
        num_rewrites = num_rewrites or NUM_QUERY_REWRITES
        prompt = QUERY_REWRITE_PROMPT.format(num_rewrites=num_rewrites, question=question)

        response = self.llm.generate(
            prompt,
            system_prompt="你是麻醉学知识检索助手，擅长在不改变问题意图的前提下改写临床问题。",
            temperature=REWRITE_TEMPERATURE,
            top_p=REWRITE_TOP_P,
            max_new_tokens=128,
        )

        rewrites = self._parse(response, num_rewrites, question)

        return [question] + rewrites

    def _parse(self, response: str, expected: int, original_question: str) -> List[str]:
        raw_lines = []
        for line in response.strip().split("\n"):
            line = re.sub(r"^[\d]+[.、)）]\s*", "", line.strip())
            line = re.sub(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", line)
            line = re.sub(r"^[-•·]\s*", "", line).strip()
            if line and len(line) > 3:
                raw_lines.append(line)

        lines = []
        seen = set()
        for line in raw_lines:
            if line not in seen and line != original_question:
                seen.add(line)
                lines.append(line)

        return lines[:expected]


class TripleExtractor:


    INTENT_MAP = {
        "作用": "主要作用", "功效": "主要作用", "效果": "主要作用",
        "机制": "作用机制", "原理": "作用机制",
        "定义": "定义", "是什么": "定义", "是啥": "定义",
        "适应": "适应证", "用于": "适应证", "应用": "适应证",
        "禁忌": "禁忌证",
        "副作用": "副作用", "不良反应": "副作用",
        "并发症": "并发症", "风险": "并发症",
        "剂量": "用法用量", "用量": "用法用量", "怎么用": "用法用量",
        "注意": "注意事项", "注意事项": "注意事项",
        "区别": "比较", "对比": "比较", "比较": "比较",
        "包括": "包括", "包含": "包括", "哪些": "包括",
        "步骤": "操作流程", "流程": "操作流程", "怎么做": "操作流程",
        "特点": "特点", "优势": "特点", "优点": "特点",
        "任务": "主要任务", "职责": "主要任务", "目的": "目的",
        "治疗": "治疗方法", "处理": "治疗方法",
    }

    def __init__(self, llm):
        self.llm = llm

    def extract_from_query(self, query: str) -> List[Dict[str, str]]:
        prompt = TRIPLE_EXTRACTION_PROMPT.format(queries=query)
        response = self.llm.generate(
            prompt,
            system_prompt=(
                "你是麻醉学知识图谱构建助手，请严格按 JSON 数组输出。"
                "head 和 relation 必须有；tail 可以为空，也可以填写你认为最合理的答案实体。"
            ),
            temperature=TRIPLE_TEMPERATURE,
            top_p=TRIPLE_TOP_P,
            max_new_tokens=256,
        )
        return self._parse(response)

    def extract_from_query_set(self, query_set: List[str]) -> List[Dict[str, str]]:
        unique_queries = list(dict.fromkeys(query_set))
        queries_text = "\n".join([f"- {q}" for q in unique_queries])
        all_triples_raw = self.extract_from_query(queries_text)

        all_triples, seen = [], set()
        for t in all_triples_raw:
            head = (t.get("head", "") or "").strip()
            rel = (t.get("relation", "") or "").strip()
            tail = (t.get("tail", "") or "").strip()
            if not head or not rel:
                continue
            key = (head, rel, tail)
            if key not in seen:
                seen.add(key)
                all_triples.append({"head": head, "relation": rel, "tail": tail})

        if len(all_triples) < 2:
            fallback = self._fallback_extract(unique_queries)
            for t in fallback:
                key = (t["head"], t["relation"], t["tail"])
                if key not in seen:
                    seen.add(key)
                    all_triples.append(t)

        return all_triples

    def _parse(self, response: str) -> List[Dict[str, str]]:
        response = response.strip()
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return self._normalize_triples(data)
        except Exception:
            pass
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                sub = response[start:end + 1]
                data = json.loads(sub)
                if isinstance(data, list):
                    return self._normalize_triples(data)
        except Exception:
            pass
        return []

    def _normalize_triples(self, data: List[Dict]) -> List[Dict[str, str]]:
        triples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head", "") or "").strip()
            rel = str(item.get("relation", "") or "").strip()
            tail = str(item.get("tail", "") or "").strip()
            if not head or not rel:
                continue
            triples.append({"head": head, "relation": rel, "tail": tail})
        return triples

    def _fallback_extract(self, queries: List[str]) -> List[Dict[str, str]]:
        fallback_triples = []
        seen = set()
        for q in queries:
            entities = self._extract_entities_from_question(q)
            relations = []
            for keyword, rel in self.INTENT_MAP.items():
                if keyword in q:
                    relations.append(rel)
            if not relations:
                relations = ["相关"]
            for ent in entities:
                for rel in relations:
                    key = (ent, rel, "")
                    if key not in seen:
                        seen.add(key)
                        fallback_triples.append({"head": ent, "relation": rel, "tail": ""})
        return fallback_triples

    @staticmethod
    def _extract_entities_from_question(question: str) -> List[str]:
        entities = []
        q = question.strip().rstrip("？?。.！!")
        if "的" in q:
            parts = q.split("的")
            head = parts[0].strip()
            if 2 <= len(head) <= 20:
                entities.append(head)
        for sep in ["和", "与", "还是", "以及", "，", ","]:
            if sep in q:
                for part in q.split(sep):
                    part = part.strip()
                    part = re.sub(r'(的主要|的具体|的常见|是什么|是啥|有哪些|怎么样|如何|怎么|吗|呢).*$', '', part).strip()
                    if 2 <= len(part) <= 15:
                        entities.append(part)
        if not entities:
            cleaned = re.sub(r'(是什么|是啥|有哪些|怎么样|如何|怎么|吗|呢|的主要|的具体|的常见).*$', '', q).strip()
            if 2 <= len(cleaned) <= 20:
                entities.append(cleaned)
        unique = []
        seen = set()
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique