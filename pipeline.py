import json, os, time, torch
import numpy as np
from typing import Dict, List
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import *
from utils.data_loader import (
    load_entity_table, load_kg_triples, load_kg_graph,
    build_graph_from_triples, load_test_qa
)
from utils.embedding import EmbeddingEncoder
from modules.query_augmentation import QueryAugmentor, TripleExtractor, QueryAugmentorAndExtractor
from modules.semantic_recall import SemanticRecall
from modules.dghma import DGHMAModule
from modules.path_ranking import PathRanker, format_structured_input_for_llm
from modules.answer_fusion import MultiMetricFusion
from modules.query_graph_builder import QueryGraphBuilder

from utils.embedding_cache import NodeEmbeddingCache
from utils.vllm_wrapper import create_llm


LLM_BACKEND = getattr(sys.modules.get("configs.config"), "LLM_BACKEND", "transformers")
VLLM_SERVER_URL = getattr(sys.modules.get("configs.config"), "VLLM_SERVER_URL", "http://localhost:8000")
VLLM_TENSOR_PARALLEL = getattr(sys.modules.get("configs.config"), "VLLM_TENSOR_PARALLEL", 1)
VLLM_GPU_MEM_UTIL = getattr(sys.modules.get("configs.config"), "VLLM_GPU_MEM_UTIL", 0.90)
NODE_EMB_CACHE_DIR = getattr(
    sys.modules.get("configs.config"), "NODE_EMB_CACHE_DIR",
    os.path.join(OUTPUT_DIR, "node_emb_cache")
)


class DualKnowPipeline:
    def __init__(self, model_path=None, encoder_path=None, data_dir=None):
        self.model_path = model_path or ANESGLM_MODEL_PATH
        self.encoder_path = encoder_path or BGE_MODEL_PATH
        self.data_dir = data_dir or DATA_DIR
        self._initialized = False

    def initialize(self):

        self.entities = load_entity_table(os.path.join(self.data_dir, "entity_table.jsonl"))
        self.triples = load_kg_triples(os.path.join(self.data_dir, "kg_triples.jsonl"))
        self.graph = load_kg_graph(os.path.join(self.data_dir, "kg_graph.pkl"))
        if self.graph is None:
            self.graph = build_graph_from_triples(self.triples, self.entities)


        self.encoder = EmbeddingEncoder(self.encoder_path).load()


        cache_dir = NODE_EMB_CACHE_DIR
        if os.path.exists(os.path.join(cache_dir, "node_embeddings.npy")):
            self.node_emb_cache = NodeEmbeddingCache(cache_dir).load()
        else:
            NodeEmbeddingCache.build_cache(
                entity_table_path=os.path.join(self.data_dir, "entity_table.jsonl"),
                encoder=self.encoder,
                cache_dir=cache_dir,
            )
            self.node_emb_cache = NodeEmbeddingCache(cache_dir).load()


        self.llm = create_llm(
            backend=LLM_BACKEND,
            model_path=self.model_path,
            device=DEVICE,
            server_url=VLLM_SERVER_URL,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL,
            gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        )
        self.llm.load()


        self.combined_augmentor = QueryAugmentorAndExtractor(self.llm)
        self.qa = QueryAugmentor(self.llm)
        self.te = TripleExtractor(self.llm)


        self.sr = SemanticRecall(self.encoder, self.entities, self.graph)
        idx_dir = os.path.join(OUTPUT_DIR, "index")
        if os.path.exists(os.path.join(idx_dir, "name.faiss")):
            self.sr.load_indexes(idx_dir)
        else:
            self.sr.build_indexes()
            self.sr.save_indexes(idx_dir)


        self.qgb = QueryGraphBuilder(
            self.encoder,
            self.entities,
            self.sr.name_index,
            alignment_dir=os.path.join(self.data_dir, "alignment")
        )


        tl = sorted(set(e.get("type", "未知") or "未知" for e in self.entities.values()))
        rl = sorted(set(t["relation"] for t in self.triples))
        dghma_ckpt = os.path.join(OUTPUT_DIR, "dghma_best.pt")
        try:
            if os.path.exists(dghma_ckpt):
                ckpt = torch.load(dghma_ckpt, map_location=DEVICE)
                tl = ckpt.get("type_list", tl)
                rl = ckpt.get("relation_list", rl)
                self.dghma = DGHMAModule(type_list=tl, relation_list=rl).to(DEVICE)
                self.dghma.load_state_dict(ckpt["model_state_dict"])
            else:
                self.dghma = DGHMAModule(type_list=tl, relation_list=rl).to(DEVICE)
        except Exception:
            self.dghma = DGHMAModule(type_list=tl, relation_list=rl).to(DEVICE)
        self.dghma.eval()


        self.pr = PathRanker(encoder=self.encoder)
        self.af = MultiMetricFusion(encoder=self.encoder, enable_brevity_guard=True)

        self._initialized = True

    def run(self, question: str, verbose=True) -> Dict:
        assert self._initialized, "请先 initialize()"
        r = {"question": question}
        T = {}


        t0 = time.perf_counter()
        qs, qt = self.combined_augmentor.rewrite_and_extract(question)
        r["query_set"] = qs
        r["query_triples"] = qt
        T["rewrite_and_triple"] = time.perf_counter() - t0

        if verbose:
            print(f"\n[Q] {question}")
            for i, q in enumerate(qs):
                print(f"  {'原始' if i == 0 else f'改写{i}'}: {q}")


        t0 = time.perf_counter()
        qe, Gq = self.qgb.build(qt, question=question, query_set=qs)
        r["query_subgraph"] = {
            "nodes": [{"node_id": n, **d} for n, d in Gq.nodes(data=True)],
            "edges": [
                {"head": u, "tail": v, "relation": d.get("relation", "")}
                for u, v, d in Gq.edges(data=True)
            ],
        }
        T["entity_link"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        seeds, sg = self.sr.recall_and_build_subgraph(qs, qe)
        T["recall"] = time.perf_counter() - t0


        t_stage = time.perf_counter()

        nids = list(sg.nodes())
        if nids:

            t_cache = time.perf_counter()
            nem = self.node_emb_cache.get_batch(nids)


            miss_nids = [nid for nid in nids if nid not in self.node_emb_cache]
            if miss_nids:
                miss_txts = []
                for nid in miss_nids:
                    e = self.entities.get(nid, sg.nodes[nid])
                    miss_txts.append(
                        f"{e.get('name', '')}；{e.get('type', '')}；{e.get('description', '')}"
                    )
                miss_embs = self.encoder.encode(miss_txts)
                for i, nid in enumerate(miss_nids):
                    nem[nid] = miss_embs[i]

            T["node_embed_from_cache"] = time.perf_counter() - t_cache

            t_query_enc = time.perf_counter()
            qemb = self.encoder.encode_single(question)
            T["query_encode"] = time.perf_counter() - t_query_enc

            t_forward = time.perf_counter()
            with torch.no_grad():
                nf, fq = self.dghma(sg, nem, qemb, self.entities)
            T["dghma_forward"] = time.perf_counter() - t_forward
        else:
            nf, fq = {}, torch.zeros(DGHMA_HIDDEN_DIM, device=DEVICE)
            T["node_embed_from_cache"] = 0.0
            T["query_encode"] = 0.0
            T["dghma_forward"] = 0.0

        T["dghma"] = time.perf_counter() - t_stage


        t0 = time.perf_counter()
        paths = self.pr.enumerate_paths(sg, seeds)
        if paths and nf:
            tp, si = self.pr.select_topk_paths(
                paths, nf, fq, sg, self.entities,
                seed_ids=seeds, question=question
            )
        else:
            tp, si = [], []

        r["top_paths"] = [[str(n) for n in p] for p in tp]
        if tp:
            try:
                r["top_paths_readable"] = self.pr.paths_to_readable(tp, sg, self.entities)
            except Exception as e:
                r["top_paths_readable"] = [f"[error] {e}"]
            if hasattr(self.pr, "paths_to_readable_struct"):
                try:
                    r["top_paths_struct"] = self.pr.paths_to_readable_struct(tp, sg, self.entities)
                except Exception as e:
                    r["top_paths_struct"] = [f"[error] {e}"]
        else:
            r["top_paths_readable"] = []
            r["top_paths_struct"] = []

        r["structured_input"] = si
        T["path"] = time.perf_counter() - t0


        t0 = time.perf_counter()

        a_llm, lp_llm, a_rag, lp_rag = self._generate_dual_answers(
            question, si if si else None
        )

        r["answer_llm"] = a_llm
        r["answer_rag"] = a_rag
        T["gen"] = time.perf_counter() - t0


        t0 = time.perf_counter()
        ents = list(set([x["head"] for x in si] + [x["tail"] for x in si]))
        fa, fd = self.af.select_answer(question, a_llm, a_rag, lp_llm, lp_rag, ents)
        r["final_answer"] = fa
        r["fusion_detail"] = fd
        T["fusion"] = time.perf_counter() - t0

        r["timing"] = T

        if verbose:
            print(f"选择: {fd['selected']} | LLM={fd['score_llm']:.4f} RAG={fd['score_rag']:.4f}")
            if r.get("top_paths_readable"):
                print("Top paths:")
                for i, p in enumerate(r["top_paths_readable"], 1):
                    print(f"  {i}. {p}")
            print(f"答案: {fa[:200]}")
            total = sum(T.values())
            print(f"耗时: {total:.1f}s")

            for k, v in T.items():
                print(f"  {k}: {v:.2f}s")

        return r

    def _generate_dual_answers(self, question, structured_input):

        from modules.llm_inference import DIRECT_QA_PROMPT, RAG_QA_PROMPT
        from modules.path_ranking import format_structured_input_for_llm

        direct_prompt = DIRECT_QA_PROMPT.format(question=question).strip()

        if structured_input:
            knowledge_text = format_structured_input_for_llm(structured_input, question=question)
        else:
            knowledge_text = "未检索到明确知识路径。"

        rag_prompt = RAG_QA_PROMPT.format(
            knowledge_paths=knowledge_text,
            question=question,
        ).strip()

        if hasattr(self.llm, "generate_batch"):

            results = self.llm.generate_batch(
                prompts=[direct_prompt, rag_prompt],
                temperature=ANSWER_TEMPERATURE,
                top_p=ANSWER_TOP_P,
                logprobs=True,
            )
            a_llm, lp_llm = results[0]
            a_rag, lp_rag = results[1]
        else:

            a_llm, lp_llm = self.llm.generate_with_logprobs(
                direct_prompt,
                temperature=ANSWER_TEMPERATURE,
                top_p=ANSWER_TOP_P,
            )
            if structured_input:
                a_rag, lp_rag = self.llm.generate_with_logprobs(
                    rag_prompt,
                    temperature=ANSWER_TEMPERATURE,
                    top_p=ANSWER_TOP_P,
                )
            else:
                a_rag, lp_rag = a_llm, lp_llm

        return a_llm, float(lp_llm or 0), a_rag, float(lp_rag or 0)

    def batch_run(self, questions, save_path=None, verbose=False):
        results = []
        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}]")
            try:
                results.append(self.run(q, verbose=verbose))
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"question": q, "error": str(e)})

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    for rr in results:
                        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        return results
