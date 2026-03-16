"""
快速验证 DUAL-Know 工程的数据和环境

用途：
1. 检查关键依赖是否安装
2. 检查数据文件是否存在
3. 检查模型路径是否存在
4. 检查数据格式是否基本正确
5. 检查图是否可读取
6. 可选检查 embedding / LLM 是否能成功加载

运行示例：
python verify.py
python verify.py --check_encoder
python verify.py --check_llm
"""

import os
import sys
import json
import traceback
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    PROJECT_ROOT,
    DATA_DIR,
    ANESGLM_MODEL_PATH,
    BGE_MODEL_PATH,
    ENTITY_TABLE_PATH,
    KG_GRAPH_PATH,
    TEST_QA_PATH,
)

from utils.data_loader import (
    load_entity_table,
    load_kg_triples,
    load_kg_graph,
    load_test_qa,
    build_graph_from_triples,
)

KG_TRIPLES_PATH = os.path.join(DATA_DIR, "kg_triples.jsonl")


def print_ok(msg: str):
    print(f"[OK] {msg}")


def print_warn(msg: str):
    print(f"[WARN] {msg}")


def print_fail(msg: str):
    print(f"[FAIL] {msg}")


def try_import(pkg_name: str) -> Tuple[bool, str]:
    try:
        __import__(pkg_name)
        return True, ""
    except Exception as e:
        return False, str(e)


def check_dependencies() -> bool:
    print("=" * 60)
    print("1. 检查依赖")
    print("=" * 60)

    deps = [
        "torch",
        "transformers",
        "sentence_transformers",
        "networkx",
        "numpy",
    ]

    all_ok = True
    for dep in deps:
        ok, err = try_import(dep)
        if ok:
            print_ok(f"{dep} 已安装")
        else:
            print_fail(f"{dep} 未安装: {err}")
            all_ok = False

    faiss_ok = False
    for name in ["faiss", "faiss_cpu"]:
        ok, _ = try_import(name)
        if ok:
            faiss_ok = True
            break
    if faiss_ok:
        print_ok("faiss 已安装")
    else:
        print_warn("faiss 未检测到，若只做基础检查可先忽略；若要建索引需安装 faiss-cpu")

    return all_ok


def resolve_kg_triples_path():
    return KG_TRIPLES_PATH if os.path.exists(KG_TRIPLES_PATH) else None

def check_paths() -> bool:
    print("\n" + "=" * 60)
    print("2. 检查路径")
    print("=" * 60)

    all_ok = True

    if os.path.exists(PROJECT_ROOT):
        print_ok(f"PROJECT_ROOT: {PROJECT_ROOT}")
    else:
        print_fail(f"PROJECT_ROOT 不存在: {PROJECT_ROOT}")
        all_ok = False

    if os.path.exists(DATA_DIR):
        print_ok(f"DATA_DIR: {DATA_DIR}")
    else:
        print_fail(f"DATA_DIR 不存在: {DATA_DIR}")
        all_ok = False

    if os.path.exists(ENTITY_TABLE_PATH):
        print_ok(f"entity_table.jsonl 存在")
    else:
        print_fail(f"entity_table.jsonl 不存在: {ENTITY_TABLE_PATH}")
        all_ok = False

    kg_triples_path = resolve_kg_triples_path()
    if kg_triples_path:
        print_ok(f"KG triples 文件存在: {os.path.basename(kg_triples_path)}")
    else:
        print_fail("kg_triples.jsonl / kg_triples.jsonl 都不存在")
        all_ok = False

    if os.path.exists(KG_GRAPH_PATH):
        print_ok("kg_graph.pkl 存在")
    else:
        print_warn(f"kg_graph.pkl 不存在，将来可由 triples 重建: {KG_GRAPH_PATH}")

    if os.path.exists(TEST_QA_PATH):
        print_ok("testQAFinal.jsonl 存在")
    else:
        print_fail(f"testQAFinal.jsonl 不存在: {TEST_QA_PATH}")
        all_ok = False

    if os.path.exists(ANESGLM_MODEL_PATH):
        print_ok(f"AnesGLM 模型目录存在: {ANESGLM_MODEL_PATH}")
    else:
        print_warn(f"AnesGLM 模型目录不存在: {ANESGLM_MODEL_PATH}")

    print_ok(f"BGE 模型配置为: {BGE_MODEL_PATH}")

    return all_ok


def check_data_format() -> bool:
    print("\n" + "=" * 60)
    print("3. 检查数据格式")
    print("=" * 60)

    all_ok = True
    kg_triples_path = resolve_kg_triples_path()
    if kg_triples_path is None:
        print_fail("无法检查 triples，因为未找到 triples 文件")
        return False

    # entity
    try:
        entities = load_entity_table(ENTITY_TABLE_PATH)
        if not entities:
            print_fail("实体表为空")
            all_ok = False
        else:
            first_id = next(iter(entities))
            first_item = entities[first_id]
            if all(k in first_item for k in ["name", "type", "description"]):
                print_ok(f"实体表格式正常，示例 id={first_id}")
            else:
                print_fail("实体表字段不完整，应包含 name/type/description")
                all_ok = False
    except Exception as e:
        print_fail(f"读取实体表失败: {e}")
        traceback.print_exc()
        return False

    # triples
    try:
        triples = load_kg_triples(kg_triples_path)
        if not triples:
            print_fail("三元组表为空")
            all_ok = False
        else:
            sample = triples[0]
            required = ["head_id", "head", "head_type", "relation", "tail_id", "tail", "tail_type"]
            missing = [k for k in required if k not in sample]
            if missing:
                print_fail(f"三元组字段缺失: {missing}")
                all_ok = False
            else:
                print_ok("三元组格式正常")
    except Exception as e:
        print_fail(f"读取三元组失败: {e}")
        traceback.print_exc()
        return False

    # qa
    try:
        qa = load_test_qa(TEST_QA_PATH)
        if not qa:
            print_fail("测试 QA 为空")
            all_ok = False
        else:
            sample = qa[0]
            if "question" in sample and "answer" in sample:
                print_ok("测试 QA 格式正常")
            else:
                print_fail("测试 QA 应包含 question / answer")
                all_ok = False
    except Exception as e:
        print_fail(f"读取测试 QA 失败: {e}")
        traceback.print_exc()
        return False

    return all_ok


def check_graph() -> bool:
    print("\n" + "=" * 60)
    print("4. 检查图结构")
    print("=" * 60)

    kg_triples_path = resolve_kg_triples_path()
    if kg_triples_path is None:
        print_fail("无法检查图，因为未找到 triples 文件")
        return False

    try:
        entities = load_entity_table(ENTITY_TABLE_PATH)
        triples = load_kg_triples(kg_triples_path)

        if os.path.exists(KG_GRAPH_PATH):
            g = load_kg_graph(KG_GRAPH_PATH)
            if g is not None:
                print_ok(f"已成功读取 kg_graph.pkl: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")
                return True
            print_warn("kg_graph.pkl 存在但读取失败，将尝试从 triples 重建")

        g = build_graph_from_triples(triples, entities)
        print_ok(f"从 triples 成功重建图: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")
        return True

    except Exception as e:
        print_fail(f"图检查失败: {e}")
        traceback.print_exc()
        return False


def check_encoder_load() -> bool:
    print("\n" + "=" * 60)
    print("5. 检查 EmbeddingEncoder 加载")
    print("=" * 60)

    try:
        from utils.embedding import EmbeddingEncoder

        enc = EmbeddingEncoder(BGE_MODEL_PATH).load()
        vec = enc.encode_single("麻醉期间心律失常的主要原因是什么？")
        print_ok(f"Encoder 加载成功，向量维度 = {len(vec)}")
        return True
    except Exception as e:
        print_fail(f"Encoder 加载失败: {e}")
        traceback.print_exc()
        return False


def check_llm_load() -> bool:
    print("\n" + "=" * 60)
    print("6. 检查 AnesGLM 加载")
    print("=" * 60)

    try:
        from utils.llm_wrapper import AnesGLM

        llm = AnesGLM(ANESGLM_MODEL_PATH).load()
        out = llm.generate(
            "请简短回答：麻醉前评估门诊的主要任务是什么？",
            max_new_tokens=64,
        )
        print_ok("AnesGLM 加载成功")
        print("模型测试输出：", out[:200])
        return True
    except Exception as e:
        print_fail(f"AnesGLM 加载失败: {e}")
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证 DUAL-Know 数据与环境")
    parser.add_argument("--check_encoder", action="store_true", help="额外检查 embedding 模型能否成功加载")
    parser.add_argument("--check_llm", action="store_true", help="额外检查 AnesGLM 能否成功加载")
    args = parser.parse_args()

    results = []

    results.append(("dependencies", check_dependencies()))
    results.append(("paths", check_paths()))
    results.append(("data_format", check_data_format()))
    results.append(("graph", check_graph()))

    if args.check_encoder:
        results.append(("encoder", check_encoder_load()))

    if args.check_llm:
        results.append(("llm", check_llm_load()))

    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    for name, ok in results:
        if ok:
            print_ok(name)
        else:
            print_fail(name)

    if all(ok for _, ok in results):
        print("\n整体看起来没问题，可以继续跑 DUAL-Know。")
    else:
        print("\n存在失败项，建议先修复后再继续。")


if __name__ == "__main__":
    main()