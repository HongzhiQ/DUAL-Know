# -*- coding: utf-8 -*-
"""
DGHMA 路径级训练脚本（支持 hard / soft 两种监督）

训练目标：
让“与参考答案更相关的路径”获得更高评分。

【hard 模式】
正样本定义：路径上的实体名在参考答案中的命中率 >= POS_HIT_THRESHOLD
负样本定义：否则为负样本
Loss：MarginRankingLoss，要求正路径得分 > 负路径得分 + margin

【soft 模式】
路径目标分数：路径文本与参考答案文本的 BGE 余弦相似度，映射到 [0, 1]
Loss：
  - mse   : MSELoss(sigmoid(score), soft_target)
  - kldiv : KLDivLoss(log_softmax(score / T), soft_target_distribution)

两阶段：
1) preprocess:
   不调用 LLM，仅使用 BGE encoder 做轻量离线预处理：
   问题 embedding -> name/description 双通道召回 -> h-hop 子图 -> 枚举路径
   -> 生成 hard 标签 + soft 标签 -> 缓存
2) train:
   加载缓存，训练 DGHMA，并保存 best checkpoint

用法：
  python train_dghma.py --phase preprocess
  python train_dghma.py --phase train --epochs 30 --lr 1e-4
  python train_dghma.py --phase all --epochs 30

  # hard-label（与你原来的训练方式一致）
  python train_dghma.py --phase all --label_mode hard

  # soft-label + MSE
  python train_dghma.py --phase all --label_mode soft --soft_loss mse

  # soft-label + KLDiv
  python train_dghma.py --phase all --label_mode soft --soft_loss kldiv --soft_temperature 0.5

可选：
  python train_dghma.py --phase all --qa_path path/to/trainQA.jsonl
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    DATA_DIR,
    OUTPUT_DIR,
    BGE_MODEL_PATH,
    DEVICE,
    EMBEDDING_DIM,
    DGHMA_HIDDEN_DIM,
    DGHMA_NUM_LAYERS,
    DGHMA_NUM_HEADS,
    DGHMA_DROPOUT,
    SUBGRAPH_HOP,
    DESC_TOPK,
    EXPLICIT_TOPK,
    UNCERTAINTY_LAMBDA,
)

from utils.data_loader import (
    load_entity_table,
    load_kg_triples,
    load_kg_graph,
    build_graph_from_triples,
    load_test_qa,
)
from utils.embedding import EmbeddingEncoder
from modules.semantic_recall import SemanticRecall
from modules.dghma import DGHMAModule
from modules.path_ranking import PathRanker



CACHE_DIR = os.path.join(OUTPUT_DIR, "dghma_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "train_cache.pkl")
META_PATH = os.path.join(CACHE_DIR, "meta.json")
CKPT_PATH = os.path.join(OUTPUT_DIR, "dghma_best.pt")

POS_HIT_THRESHOLD = 0.3
DEFAULT_MARGIN = 0.3
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
GRAD_CLIP = 1.0

DEFAULT_LABEL_MODE = "hard"      # ["hard", "soft"]
DEFAULT_SOFT_LOSS = "kldiv"      # ["mse", "kldiv"]
DEFAULT_SOFT_TEMPERATURE = 0.5
SOFT_EPS = 1e-8



def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace(" ", "").strip()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_qa_file(qa_path: str):
    """
    格式 {"question": "...", "answer": "..."} 的 JSONL。
    """
    return load_test_qa(qa_path)


def choose_default_qa_path(data_dir: str) -> str:
    train_path = os.path.join(data_dir, "TrainQAFinal.jsonl")
    test_path = os.path.join(data_dir, "testQAFinal.jsonl")

    if os.path.exists(train_path):
        print(f"[QA] 默认使用训练集: {train_path}")
        return train_path

    if os.path.exists(test_path):
        print(f"[警告] 未找到 trainQAFinal.jsonl，当前退回使用: {test_path}")
        return test_path

    raise FileNotFoundError(
        f"未找到默认 QA 文件。请检查 {train_path} 或 {test_path}，或手动传入 --qa_path"
    )


def cosine_similarity_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def label_path(path: List[str], entities: Dict[str, dict], ref_answer: str) -> int:

    ref_answer = normalize_text(ref_answer)
    names = []

    for nid in path:
        name = normalize_text(entities.get(nid, {}).get("name", ""))
        if len(name) >= 2:
            names.append(name)

    if not names:
        return 0

    hits = sum(1 for n in names if n in ref_answer)
    hit_ratio = hits / len(names)
    return 1 if hit_ratio >= POS_HIT_THRESHOLD else 0



def build_path_text(path: List[str], sg: nx.DiGraph, entities: Dict[str, dict]) -> str:

    parts = []
    path_len = len(path)

    for i, nid in enumerate(path):
        e = entities.get(nid, {})
        name = str(e.get("name", "") or "")
        etype = str(e.get("type", "") or "未知")
        desc = str(e.get("description", "") or "")

        node_text = f"{name}（类型：{etype}"
        if desc:
            node_text += f"；描述：{desc}"
        node_text += "）"
        parts.append(node_text)

        if i < path_len - 1:
            s, t = path[i], path[i + 1]
            relation = "相关"
            if sg.has_edge(s, t):
                relation = sg[s][t].get("relation", "相关")
            elif sg.has_edge(t, s):
                relation = sg[t][s].get("relation", "相关")
            parts.append(f" --{relation}--> ")

    return "".join(parts)


def compute_path_soft_scores(
    path_texts: List[str],
    ref_answer: str,
    encoder: EmbeddingEncoder,
) -> List[float]:

    if not path_texts:
        return []

    answer_emb = to_numpy(encoder.encode_single(ref_answer))
    path_embs = to_numpy(encoder.encode(path_texts))

    if path_embs.ndim == 1:
        path_embs = path_embs[None, :]

    scores = []
    for pe in path_embs:
        cos = cosine_similarity_np(pe, answer_emb)
        score = max(0.0, min(1.0, (cos + 1.0) / 2.0))
        scores.append(float(score))

    return scores


def build_path_index_tensors(
    paths: List[List[str]],
    node_ids: List[str],
):

    nid2idx = {nid: i for i, nid in enumerate(node_ids)}
    valid_paths = []
    max_len = 0

    for path in paths:
        idxs = [nid2idx[nid] for nid in path if nid in nid2idx]
        if len(idxs) == 0:
            idxs = [0]
            mask_len = 0
        else:
            mask_len = len(idxs)
        valid_paths.append((idxs, mask_len))
        max_len = max(max_len, len(idxs))

    if max_len == 0:
        max_len = 1

    path_node_indices = np.zeros((len(paths), max_len), dtype=np.int64)
    path_mask = np.zeros((len(paths), max_len), dtype=np.float32)

    for i, (idxs, mask_len) in enumerate(valid_paths):
        path_node_indices[i, :len(idxs)] = np.asarray(idxs, dtype=np.int64)
        if mask_len > 0:
            path_mask[i, :mask_len] = 1.0

    return path_node_indices, path_mask


def score_paths_differentiable(
    path_node_indices: torch.Tensor,
    path_mask: torch.Tensor,
    node_feature_matrix: torch.Tensor,
    query_vector: torch.Tensor,
    lam: float = UNCERTAINTY_LAMBDA,
) -> torch.Tensor:

    path_embs = node_feature_matrix[path_node_indices]

    attn_logits = torch.matmul(path_embs, query_vector)


    attn_logits = attn_logits.masked_fill(path_mask <= 0, -1e30)
    weights = F.softmax(attn_logits, dim=1)
    weights = weights * path_mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

    R = (weights.unsqueeze(-1) * path_embs).sum(dim=1)

    h_bar = R.mean(dim=0)


    sigma = torch.sqrt(torch.sum((R - h_bar) ** 2, dim=-1) + 1e-8)


    sim = F.cosine_similarity(R, query_vector.unsqueeze(0), dim=-1)


    scores = sim * torch.exp(-lam * sigma)
    return scores



def preprocess(data_dir: str, encoder_path: str, qa_path: str, max_samples=None):
    print("=" * 70)
    print("Phase 1: 预处理 DGHMA 训练缓存")
    print("=" * 70)

    ensure_dir(CACHE_DIR)


    entities = load_entity_table(os.path.join(data_dir, "entity_table.jsonl"))
    triples = load_kg_triples(os.path.join(data_dir, "kg_triples.jsonl"))
    graph = load_kg_graph(os.path.join(data_dir, "kg_graph.pkl"))
    if graph is None:
        graph = build_graph_from_triples(triples, entities)

    qa_data = load_qa_file(qa_path)
    if max_samples is not None:
        qa_data = qa_data[:max_samples]

    print(f"[数据] 实体: {len(entities)}")
    print(f"[数据] 三元组: {len(triples)}")
    print(f"[数据] QA: {len(qa_data)}")


    encoder = EmbeddingEncoder(encoder_path).load()
    sr = SemanticRecall(encoder, entities, graph)

    idx_dir = os.path.join(OUTPUT_DIR, "index")
    name_idx_file = os.path.join(idx_dir, "name.faiss")
    if os.path.exists(name_idx_file):
        print(f"[索引] 读取已有索引: {idx_dir}")
        sr.load_indexes(idx_dir)
    else:
        print("[索引] 未找到索引，开始构建...")
        sr.build_indexes()
        sr.save_indexes(idx_dir)
        print(f"[索引] 已保存到: {idx_dir}")

    pr = PathRanker()


    type_list = sorted(set((e.get("type", "未知") or "未知") for e in entities.values()))
    relation_list = sorted(set(t["relation"] for t in triples))


    cache = []
    stats = {
        "total": 0,
        "valid": 0,
        "skipped": 0,
        "pos_paths": 0,
        "neg_paths": 0,
        "soft_score_sum": 0.0,
        "soft_score_count": 0,
    }

    for idx, qa in enumerate(qa_data):
        # 兼容 {"messages": [...]} 格式
        if "messages" in qa:
            question, answer = "", ""
            for msg in qa["messages"]:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    answer = msg.get("content", "")
        # 兼容原有的 {"question": "...", "answer": "..."} 格式
        else:
            question = qa.get("question", "")
            answer = qa.get("answer", "")


        if not question or not answer:
            stats["skipped"] += 1
            continue

        stats["total"] += 1

        if (idx + 1) % 50 == 0:
            soft_avg = (
                stats["soft_score_sum"] / max(stats["soft_score_count"], 1)
            )
            print(
                f"[{idx + 1}/{len(qa_data)}] "
                f"valid={stats['valid']} skipped={stats['skipped']} "
                f"pos={stats['pos_paths']} neg={stats['neg_paths']} "
                f"soft_avg={soft_avg:.4f}"
            )

        try:

            q_emb = to_numpy(encoder.encode_single(question))


            desc_results = sr.desc_index.search(q_emb, DESC_TOPK)
            desc_seeds = set(eid for eid, _ in desc_results)

            name_results = sr.name_index.search(q_emb, EXPLICIT_TOPK)
            name_seeds = set(eid for eid, _ in name_results)

            seeds = desc_seeds | name_seeds
            if len(seeds) < 2:
                stats["skipped"] += 1
                continue

            sg = sr._build_hop_subgraph(seeds, SUBGRAPH_HOP)
            if sg.number_of_nodes() < 3 or sg.number_of_edges() < 2:
                stats["skipped"] += 1
                continue

            node_ids = list(sg.nodes())
            node_texts = []
            node_types = {}

            for nid in node_ids:
                e = entities.get(nid, sg.nodes[nid])
                node_types[nid] = (e.get("type", "") or "未知")
                node_texts.append(
                    f"{e.get('name', '')}；{e.get('type', '')}；{e.get('description', '')}"
                )

            node_embs = to_numpy(encoder.encode(node_texts))
            node_embeddings = {nid: node_embs[i] for i, nid in enumerate(node_ids)}


            paths = pr.enumerate_paths(sg, seeds)
            if len(paths) < 2:
                stats["skipped"] += 1
                continue


            path_labels = [label_path(path, entities, answer) for path in paths]
            n_pos = sum(path_labels)
            n_neg = len(path_labels) - n_pos


            path_texts = [build_path_text(path, sg, entities) for path in paths]
            path_soft_scores = compute_path_soft_scores(path_texts, answer, encoder)

            if len(path_soft_scores) != len(paths):
                stats["skipped"] += 1
                continue


            if n_pos == 0 or n_neg == 0:
                stats["skipped"] += 1
                continue


            path_node_indices, path_mask = build_path_index_tensors(paths, node_ids)

            cache.append({
                "question": question,
                "answer": answer,
                "question_embedding": q_emb,
                "node_ids": node_ids,
                "node_embeddings": node_embeddings,
                "node_types": node_types,
                "subgraph": sg,
                "paths": paths,
                "path_labels": path_labels,
                "path_texts": path_texts,
                "path_soft_scores": path_soft_scores,
                "path_node_indices": path_node_indices,
                "path_mask": path_mask,
            })

            stats["valid"] += 1
            stats["pos_paths"] += n_pos
            stats["neg_paths"] += n_neg
            stats["soft_score_sum"] += float(sum(path_soft_scores))
            stats["soft_score_count"] += len(path_soft_scores)

        except Exception as e:
            print(f"[预处理失败] sample={idx}, err={e}")
            stats["skipped"] += 1
            continue


    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    meta = {
        "type_list": type_list,
        "relation_list": relation_list,
        "stats": stats,
        "qa_path": qa_path,
        "pos_hit_threshold": POS_HIT_THRESHOLD,
        "soft_score_source": "bge_cosine_[0,1]",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[预处理完成]")
    print(f"总样本: {stats['total']}")
    print(f"有效样本: {stats['valid']}")
    print(f"跳过样本: {stats['skipped']}")
    print(f"正路径数: {stats['pos_paths']}")
    print(f"负路径数: {stats['neg_paths']}")
    if stats["soft_score_count"] > 0:
        print(f"平均 soft score: {stats['soft_score_sum'] / stats['soft_score_count']:.4f}")
    print(f"缓存文件: {CACHE_PATH}")
    print(f"元数据文件: {META_PATH}")




def split_cache(cache: list, val_ratio: float = DEFAULT_VAL_RATIO, seed: int = DEFAULT_SEED):
    assert 0.0 < val_ratio < 1.0, "val_ratio 必须在 (0, 1) 之间"
    indices = np.arange(len(cache))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = max(1, int(len(indices) * (1 - val_ratio)))
    train_ids = indices[:split].tolist()
    val_ids = indices[split:].tolist()

    if len(val_ids) == 0:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]

    return train_ids, val_ids



def compute_soft_distribution(
    soft_scores: torch.Tensor,
    temperature: float = DEFAULT_SOFT_TEMPERATURE,
    eps: float = SOFT_EPS,
) -> torch.Tensor:

    temperature = max(float(temperature), eps)
    logits = soft_scores / temperature
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=eps)
    probs = probs / probs.sum()
    return probs


def dict_node_features_to_matrix(
    node_features: Dict[str, torch.Tensor],
    node_ids: List[str],
) -> torch.Tensor:

    return torch.stack([node_features[nid] for nid in node_ids], dim=0)


def compute_sample_loss(
    model: DGHMAModule,
    sample: dict,
    entities: Dict[str, dict],
    criterion,
    margin: float,
    label_mode: str = DEFAULT_LABEL_MODE,
    soft_loss: str = DEFAULT_SOFT_LOSS,
    soft_temperature: float = DEFAULT_SOFT_TEMPERATURE,
):
    sg = sample["subgraph"]

    node_features, query_vector = model(
        sg,
        sample["node_embeddings"],
        sample["question_embedding"],
        entities,
    )

    node_feature_matrix = dict_node_features_to_matrix(node_features, sample["node_ids"])

    path_node_indices = torch.tensor(
        sample["path_node_indices"], dtype=torch.long, device=DEVICE
    )
    path_mask = torch.tensor(
        sample["path_mask"], dtype=torch.float32, device=DEVICE
    )

    scores = score_paths_differentiable(
        path_node_indices=path_node_indices,
        path_mask=path_mask,
        node_feature_matrix=node_feature_matrix,
        query_vector=query_vector,
        lam=UNCERTAINTY_LAMBDA,
    )


    if label_mode == "hard":
        labels = torch.tensor(sample["path_labels"], dtype=torch.float32, device=DEVICE)
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None

        pos_scores = scores[pos_mask]  # (P,)
        neg_scores = scores[neg_mask]  # (N,)

        pos_expand = pos_scores.unsqueeze(1).expand(-1, neg_scores.size(0)).reshape(-1)
        neg_expand = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1).reshape(-1)
        target = torch.ones_like(pos_expand)

        loss = criterion(pos_expand, neg_expand, target)

        with torch.no_grad():
            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()

        return {
            "loss": loss,
            "main_mean": pos_mean,
            "aux_mean": neg_mean,
            "main_name": "pos",
            "aux_name": "neg",
        }


    elif label_mode == "soft":
        if "path_soft_scores" not in sample:
            raise KeyError("当前缓存中缺少 path_soft_scores，请重新运行 preprocess。")

        soft_targets = torch.tensor(
            sample["path_soft_scores"], dtype=torch.float32, device=DEVICE
        )

        if soft_targets.numel() != scores.numel():
            raise ValueError(
                f"path_soft_scores 数量({soft_targets.numel()})与路径数({scores.numel()})不一致。"
            )

        if soft_loss == "mse":
            pred = torch.sigmoid(scores)
            loss = F.mse_loss(pred, soft_targets)

            with torch.no_grad():
                pred_mean = pred.mean().item()
                target_mean = soft_targets.mean().item()

            return {
                "loss": loss,
                "main_mean": pred_mean,
                "aux_mean": target_mean,
                "main_name": "pred",
                "aux_name": "target",
            }

        elif soft_loss == "kldiv":
            temperature = max(float(soft_temperature), SOFT_EPS)
            pred_log_probs = F.log_softmax(scores / temperature, dim=-1)
            target_probs = compute_soft_distribution(
                soft_targets,
                temperature=temperature,
                eps=SOFT_EPS,
            )

            loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

            with torch.no_grad():
                pred_probs = torch.softmax(scores / temperature, dim=-1)
                pred_mean = pred_probs.mean().item()
                target_mean = target_probs.mean().item()

            return {
                "loss": loss,
                "main_mean": pred_mean,
                "aux_mean": target_mean,
                "main_name": "pred_prob",
                "aux_name": "target_prob",
            }

        else:
            raise ValueError(f"不支持的 soft_loss: {soft_loss}")

    else:
        raise ValueError(f"不支持的 label_mode: {label_mode}")


def run_epoch(
    model: DGHMAModule,
    cache: list,
    sample_ids: List[int],
    entities: Dict[str, dict],
    criterion,
    optimizer=None,
    grad_clip: float = GRAD_CLIP,
    label_mode: str = DEFAULT_LABEL_MODE,
    soft_loss: str = DEFAULT_SOFT_LOSS,
    soft_temperature: float = DEFAULT_SOFT_TEMPERATURE,
    margin: float = DEFAULT_MARGIN,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    valid_count = 0
    main_name = None
    aux_name = None

    if is_train:
        random.shuffle(sample_ids)

    for idx in sample_ids:
        sample = cache[idx]

        try:
            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                out = compute_sample_loss(
                    model=model,
                    sample=sample,
                    entities=entities,
                    criterion=criterion,
                    margin=margin,
                    label_mode=label_mode,
                    soft_loss=soft_loss,
                    soft_temperature=soft_temperature,
                )
                if out is None:
                    continue

                loss = out["loss"]
                main_mean = out["main_mean"]
                aux_mean = out["aux_mean"]
                main_name = out["main_name"]
                aux_name = out["aux_name"]

                if is_train:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += float(loss.item())
            total_main += float(main_mean)
            total_aux += float(aux_mean)
            valid_count += 1

        except Exception as e:
            print(f"[样本失败] idx={idx}, err={e}")
            continue

    if valid_count == 0:
        return {
            "loss": float("inf"),
            "main_mean": 0.0,
            "aux_mean": 0.0,
            "main_name": "main",
            "aux_name": "aux",
            "valid": 0,
        }

    return {
        "loss": total_loss / valid_count,
        "main_mean": total_main / valid_count,
        "aux_mean": total_aux / valid_count,
        "main_name": main_name or "main",
        "aux_name": aux_name or "aux",
        "valid": valid_count,
    }


def train(
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    margin: float = DEFAULT_MARGIN,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    label_mode: str = DEFAULT_LABEL_MODE,
    soft_loss: str = DEFAULT_SOFT_LOSS,
    soft_temperature: float = DEFAULT_SOFT_TEMPERATURE,
):
    print("=" * 70)
    print("Phase 2: 训练 DGHMA")
    print("=" * 70)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"未找到缓存文件: {CACHE_PATH}，请先运行 preprocess。")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"未找到元数据文件: {META_PATH}，请先运行 preprocess。")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(cache) < 2:
        raise ValueError("缓存样本过少，无法训练。")

    type_list = meta["type_list"]
    relation_list = meta["relation_list"]

    print(f"[缓存] 样本数: {len(cache)}")
    print(f"[缓存] 实体类型数: {len(type_list)}")
    print(f"[缓存] 关系类型数: {len(relation_list)}")
    print(f"[训练模式] label_mode={label_mode}")
    if label_mode == "soft":
        print(f"[训练模式] soft_loss={soft_loss}, temperature={soft_temperature}")

    train_ids, val_ids = split_cache(cache, val_ratio=val_ratio, seed=seed)
    print(f"[划分] train={len(train_ids)} | val={len(val_ids)}")

    entities = load_entity_table(os.path.join(DATA_DIR, "entity_table.jsonl"))

    model = DGHMAModule(
        input_dim=EMBEDDING_DIM,
        hidden_dim=DGHMA_HIDDEN_DIM,
        num_layers=DGHMA_NUM_LAYERS,
        num_heads=DGHMA_NUM_HEADS,
        type_list=type_list,
        relation_list=relation_list,
        dropout=DGHMA_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    if label_mode == "hard":
        criterion = torch.nn.MarginRankingLoss(margin=margin)
    elif label_mode == "soft":
        criterion = None
    else:
        raise ValueError(f"不支持的 label_mode: {label_mode}")

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_stats = run_epoch(
            model=model,
            cache=cache,
            sample_ids=train_ids.copy(),
            entities=entities,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=GRAD_CLIP,
            label_mode=label_mode,
            soft_loss=soft_loss,
            soft_temperature=soft_temperature,
            margin=margin,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                cache=cache,
                sample_ids=val_ids.copy(),
                entities=entities,
                criterion=criterion,
                optimizer=None,
                grad_clip=0.0,
                label_mode=label_mode,
                soft_loss=soft_loss,
                soft_temperature=soft_temperature,
                margin=margin,
            )

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"({train_stats['main_name']}={train_stats['main_mean']:.4f}, "
            f"{train_stats['aux_name']}={train_stats['aux_mean']:.4f}, "
            f"valid={train_stats['valid']}) | "
            f"val_loss={val_stats['loss']:.4f} "
            f"({val_stats['main_name']}={val_stats['main_mean']:.4f}, "
            f"{val_stats['aux_name']}={val_stats['aux_mean']:.4f}, "
            f"valid={val_stats['valid']}) | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            ensure_dir(os.path.dirname(CKPT_PATH))

            torch.save({
                "model_state_dict": model.state_dict(),
                "type_list": type_list,
                "relation_list": relation_list,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_size": len(train_ids),
                "val_size": len(val_ids),
                "seed": seed,
                "margin": margin,
                "lr": lr,
                "label_mode": label_mode,
                "soft_loss": soft_loss if label_mode == "soft" else None,
                "soft_temperature": soft_temperature if label_mode == "soft" else None,
            }, CKPT_PATH)

            print(f"  -> 保存 best checkpoint: epoch={epoch}, val_loss={best_val_loss:.4f}")

    print("\n[训练完成]")
    print(f"best epoch: {best_epoch}")
    print(f"best val loss: {best_val_loss:.4f}")
    print(f"checkpoint: {CKPT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="DGHMA 路径级训练脚本（支持 hard / soft 两种监督）")
    parser.add_argument("--phase", default="all", choices=["preprocess", "train", "all"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--qa_path", default=None)
    parser.add_argument("--encoder_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)


    parser.add_argument("--label_mode", type=str, default=DEFAULT_LABEL_MODE, choices=["hard", "soft"])
    parser.add_argument("--soft_loss", type=str, default=DEFAULT_SOFT_LOSS, choices=["mse", "kldiv"])
    parser.add_argument("--soft_temperature", type=float, default=DEFAULT_SOFT_TEMPERATURE)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = args.data_dir or DATA_DIR
    qa_path = args.qa_path or choose_default_qa_path(data_dir)
    encoder_path = args.encoder_path or BGE_MODEL_PATH

    if args.phase in ("preprocess", "all"):
        preprocess(
            data_dir=data_dir,
            encoder_path=encoder_path,
            qa_path=qa_path,
            max_samples=args.max_samples,
        )

    if args.phase in ("train", "all"):
        train(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            margin=args.margin,
            val_ratio=args.val_ratio,
            seed=args.seed,
            label_mode=args.label_mode,
            soft_loss=args.soft_loss,
            soft_temperature=args.soft_temperature,
        )


if __name__ == "__main__":
    main()