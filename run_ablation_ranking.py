

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import *
from pipeline_ablation import DualKnowAblationPipeline, VALID_STRATEGIES
from utils.data_loader import load_test_qa



def parse_qa_sample(sample):
    if "question" in sample and "answer" in sample:
        return sample["question"], sample["answer"]

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



def summarize_timing(results, strategy_name):
    """汇总某个策略的平均耗时"""
    valid = [r for r in results if "timing" in r]
    if not valid:
        return {}

    keys = valid[0]["timing"].keys()
    summary = {}
    for k in keys:
        vals = [r["timing"].get(k, 0) for r in valid]
        summary[k] = {
            "mean": sum(vals) / len(vals),
            "total": sum(vals),
        }

    total_per_query = [sum(r["timing"].values()) for r in valid]
    summary["total_per_query"] = {
        "mean": sum(total_per_query) / len(total_per_query),
        "min": min(total_per_query),
        "max": max(total_per_query),
    }

    return summary


def print_summary_table(all_summaries):

    print("\n" + "=" * 80)
    print("  路径排序策略消融实验 — 耗时对比")
    print("=" * 80)

    strategies = list(all_summaries.keys())
    if not strategies:
        print("  无有效结果")
        return

    header = f"{'策略':<20} {'平均总耗时(s)':<15} {'路径排序(s)':<15} {'DGHMA(s)':<12} {'生成(s)':<12}"
    print(header)
    print("-" * 80)

    for strategy in strategies:
        s = all_summaries[strategy]
        total = s.get("total_per_query", {}).get("mean", 0)
        path = s.get("path_ranking", {}).get("mean", s.get("path", {}).get("mean", 0))
        dghma = s.get("dghma", {}).get("mean", 0)
        gen = s.get("gen", {}).get("mean", 0)

        print(f"  {strategy:<18} {total:<15.3f} {path:<15.3f} {dghma:<12.3f} {gen:<12.3f}")

    print("=" * 80)



def main():
    parser = argparse.ArgumentParser(description="DUAL-Know 路径排序策略消融实验")
    parser.add_argument("--mode", default="batch", choices=["single", "batch"])
    parser.add_argument("--question", default="麻醉前评估门诊的主要任务是什么？")
    parser.add_argument("--dataset_name", type=str, default="testQAFinal")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--encoder_path", default=None)
    parser.add_argument("--data_dir", default=None)

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="指定单个策略运行: llm_rank / gnn_then_llm / llm_then_gnn。不指定则运行全部。"
    )
    parser.add_argument(
        "--include_v1",
        action="store_true",
        help="是否也重新运行 V1 (gnn_rank)。默认不重跑。"
    )
    parser.add_argument(
        "--cascade_factor",
        type=int,
        default=2,
        help="级联方案 (V3/V4) 的初筛倍率，默认 2 (即初筛 top-2K)"
    )

    args = parser.parse_args()

    if args.strategy:
        assert args.strategy in VALID_STRATEGIES, \
            f"--strategy 必须是 {VALID_STRATEGIES} 之一"
        strategies = [args.strategy]
    else:

        strategies = ["llm_rank", "gnn_then_llm", "llm_then_gnn"]
        if args.include_v1:
            strategies.insert(0, "gnn_rank")

    ablation_dir = os.path.join(RESULT_DIR, "ablation_ranking")
    os.makedirs(ablation_dir, exist_ok=True)

    all_summaries = {}

    if args.mode == "single":

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"  策略: {strategy}")
            print(f"{'='*60}")

            pipe = DualKnowAblationPipeline(
                ranking_strategy=strategy,
                model_path=args.model_path,
                encoder_path=args.encoder_path,
                data_dir=args.data_dir,
                cascade_factor=args.cascade_factor,
            )
            pipe.initialize()

            result = pipe.run(args.question, verbose=True)

            save_path = os.path.join(ablation_dir, f"single_{strategy}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  保存: {save_path}")

    elif args.mode == "batch":

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

        print(f"\n数据集: {args.dataset_name} | 样本数: {len(qs)}")
        print(f"要运行的策略: {strategies}\n")

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"  开始运行策略: {strategy}")
            print(f"  数据集: {args.dataset_name} | 样本数: {len(qs)}")
            print(f"{'='*60}\n")

            save_path = os.path.join(
                ablation_dir,
                f"{args.dataset_name}_{strategy}_results.jsonl"
            )

            existing_count = 0
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    existing_count = sum(1 for _ in f)
                if existing_count >= len(qs):
                    print(f"  ✓ 已有完整结果 ({existing_count} 条)，跳过")

                    with open(save_path, "r", encoding="utf-8") as f:
                        results = [json.loads(line) for line in f]
                    all_summaries[strategy] = summarize_timing(results, strategy)
                    continue
                else:
                    print(f"  发现已有部分结果 ({existing_count} 条)，从第 {existing_count+1} 条继续")

            t_start = time.perf_counter()

            pipe = DualKnowAblationPipeline(
                ranking_strategy=strategy,
                model_path=args.model_path,
                encoder_path=args.encoder_path,
                data_dir=args.data_dir,
                cascade_factor=args.cascade_factor,
            )
            pipe.initialize()

            results = []
            if existing_count > 0:
                with open(save_path, "r", encoding="utf-8") as f:
                    results = [json.loads(line) for line in f]

            remaining_qs = qs[existing_count:]
            remaining_refs = refs[existing_count:]

            new_results = pipe.batch_run(remaining_qs, save_path=None, verbose=False)

            for i, r in enumerate(new_results):
                idx = existing_count + i
                if idx < len(refs):
                    r["reference_answer"] = refs[idx]
            results.extend(new_results)

            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            t_elapsed = time.perf_counter() - t_start

            print(f"\n  ✓ {strategy} 完成 | 耗时: {t_elapsed:.1f}s | 保存: {save_path}")

            all_summaries[strategy] = summarize_timing(results, strategy)

        print_summary_table(all_summaries)

        summary_path = os.path.join(ablation_dir, f"{args.dataset_name}_timing_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n耗时汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
