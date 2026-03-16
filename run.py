"""
DUAL-Know 运行入口
  python run.py --mode single --question "麻醉前评估门诊的主要任务是什么？"
  python run.py --mode batch --dataset_name testQAFinal
  python run.py --mode batch --dataset_name mydataset --max_samples 100
  python run.py --mode build_index
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import *
from pipeline import DualKnowPipeline
from utils.data_loader import load_test_qa


def parse_qa_sample(sample):
    # 格式1：{"question": "...", "answer": "..."}
    if "question" in sample and "answer" in sample:
        return sample["question"], sample["answer"]

    # 格式2：{"messages": [{"role": "...", "content": "..."}]}
    if "messages" in sample and isinstance(sample["messages"], list):
        system_prompt = None
        question = None
        answer = None

        for msg in sample["messages"]:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system" and system_prompt is None:
                system_prompt = content
            elif role == "user" and question is None:
                question = content
            elif role == "assistant" and answer is None:
                answer = content

        if question is not None and answer is not None:
            if system_prompt:
                merged_question = f"{system_prompt}\n\n{question}"
            else:
                merged_question = question
            return merged_question, answer

    raise ValueError(f"无法识别的数据格式: {sample}")



def main():
    parser = argparse.ArgumentParser(description="DUAL-Know")
    parser.add_argument("--mode", default="single", choices=["single", "batch", "build_index"])
    parser.add_argument("--question", default="麻醉前评估门诊的主要任务是什么？")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--encoder_path", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="batch模式下的数据集名称，不带扩展名，例如 testQAFinal"
    )

    args = parser.parse_args()

    pipe = DualKnowPipeline(args.model_path, args.encoder_path, args.data_dir)

    if args.mode == "build_index":
        from utils.embedding import EmbeddingEncoder
        from modules.semantic_recall import SemanticRecall
        from utils.data_loader import (
            load_entity_table,
            load_kg_triples,
            load_kg_graph,
            build_graph_from_triples,
        )

        dd = args.data_dir or DATA_DIR
        ent = load_entity_table(os.path.join(dd, "entity_table.jsonl"))
        tri = load_kg_triples(os.path.join(dd, "kg_triples.jsonl"))
        g = load_kg_graph(os.path.join(dd, "kg_graph.pkl"))
        if g is None:
            g = build_graph_from_triples(tri, ent)

        enc = EmbeddingEncoder(args.encoder_path or BGE_MODEL_PATH).load()
        sr = SemanticRecall(enc, ent, g)
        sr.build_indexes()
        sr.save_indexes(os.path.join(OUTPUT_DIR, "index"))
        print("索引构建完成")

    else:
        pipe.initialize()

        if args.mode == "single":
            r = pipe.run(args.question, verbose=True)
            os.makedirs(RESULT_DIR, exist_ok=True)
            p = os.path.join(RESULT_DIR, "single_result.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"保存: {p}")

        elif args.mode == "batch":
            if not args.dataset_name:
                raise ValueError("batch 模式下必须传入 --dataset_name，例如：--dataset_name testQAFinal")

            data_dir = args.data_dir or DATA_DIR
            dataset_file = os.path.join(data_dir, f"{args.dataset_name}.jsonl")

            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"未找到数据集文件: {dataset_file}")

            qa = load_test_qa(dataset_file)

            if args.max_samples:
                qa = qa[:args.max_samples]

            parsed = [parse_qa_sample(x) for x in qa]
            qs = [x[0] for x in parsed]
            refs = [x[1] for x in parsed]

            os.makedirs(RESULT_DIR, exist_ok=True)
            save_path = os.path.join(RESULT_DIR, f"{args.dataset_name}_batch_results.jsonl")

            results = pipe.batch_run(qs, save_path=save_path, verbose=False)

            for i, r in enumerate(results):
                if i < len(refs):
                    r["reference_answer"] = refs[i]

            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"保存: {save_path}")


if __name__ == "__main__":
    main()