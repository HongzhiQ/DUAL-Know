import json
import argparse
import math
import os
from collections import Counter

import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge


def bleu4(ref: str, hyp: str) -> float:
    score = sentence_bleu(
        [list(ref)],
        list(hyp),
        smoothing_function=SmoothingFunction().method3,
    )
    return round(score * 100, 4)


def rouge_scores(ref: str, hyp: str) -> dict:
    hypothesis = list(jieba.cut(hyp))
    reference = list(jieba.cut(ref))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    return {
        "rouge-1": round(result["rouge-1"]["f"] * 100, 4),
        "rouge-2": round(result["rouge-2"]["f"] * 100, 4),
        "rouge-l": round(result["rouge-l"]["f"] * 100, 4),
    }



def gleu(ref: str, hyp: str) -> float:

    rt, ht = list(ref), list(hyp)
    if len(ht) < 1 or len(rt) < 1:
        return 0.0

    max_order = min(4, len(rt), len(ht))
    if max_order < 1:
        return 0.0

    log_score = 0.0
    for n in range(1, max_order + 1):
        rn = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
        hn = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
        clip = sum(min(hn[g], rn[g]) for g in hn)

        hyp_total = sum(hn.values())
        ref_total = sum(rn.values())

        if hyp_total == 0 or ref_total == 0:
            return 0.0

        precision = clip / hyp_total
        recall = clip / ref_total
        gleu_n = min(precision, recall)

        if gleu_n == 0:
            return 0.0
        log_score += math.log(gleu_n) / max_order

    return math.exp(log_score) * 100


def distinct_n(texts: list, n: int) -> float:

    all_ng = []
    for t in texts:
        ts = list(jieba.cut(t))
        all_ng.extend(tuple(ts[i:i + n]) for i in range(len(ts) - n + 1))
    return len(set(all_ng)) / len(all_ng) * 100 if all_ng else 0.0



def evaluate(path: str):
    data = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    valid = [d for d in data if d.get("final_answer") and d.get("reference_answer")]
    print(f"有效样本: {len(valid)}/{len(data)}")
    if not valid:
        return


    b4_list = []
    r1_list, r2_list, rl_list = [], [], []
    gl_list = []

    for d in valid:
        ref = d["reference_answer"]
        hyp = d["final_answer"]


        b4_list.append(bleu4(ref, hyp))
        rs = rouge_scores(ref, hyp)
        r1_list.append(rs["rouge-1"])
        r2_list.append(rs["rouge-2"])
        rl_list.append(rs["rouge-l"])


        gl_list.append(gleu(ref, hyp))

    hyps = [d["final_answer"] for d in valid]
    rag_n = sum(1 for d in valid if d.get("fusion_detail", {}).get("selected") == "RAG")


    metrics = {
        "BLEU-4": sum(b4_list) / len(b4_list),
        "ROUGE-1": sum(r1_list) / len(r1_list),
        "ROUGE-2": sum(r2_list) / len(r2_list),
        "ROUGE-L": sum(rl_list) / len(rl_list),
        "GLEU": sum(gl_list) / len(gl_list),
        "Distinct-1": distinct_n(hyps, 1),
        "Distinct-2": distinct_n(hyps, 2),
        "RAG_ratio%": rag_n / len(valid) * 100,
        "N": len(valid),
    }

    print("\n" + "=" * 50)
    print("DUAL-Know 评测结果")
    print("=" * 50)
    print("  ---- LlamaFactory 对齐指标 ----")
    for k in ["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        print(f"  {k:<10}: {metrics[k]:.2f}")
    print("  ---- 论文补充指标 ----")
    for k in ["GLEU", "Distinct-1", "Distinct-2"]:
        print(f"  {k:<10}: {metrics[k]:.2f}")
    print("  ---- 统计 ----")
    print(f"  RAG_ratio%: {metrics['RAG_ratio%']:.2f}")
    print(f"  N         : {metrics['N']}")


    mp = os.path.join(os.path.dirname(path), "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {mp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DUAL-Know 评测")
    p.add_argument("--result_path", default="outputs/results/HyponsQAtest_batch_results.jsonl")
    evaluate(p.parse_args().result_path)