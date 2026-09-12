"""M2 main experiment: targeted Unicode obfuscation on the BIPIA code split.

Held-out test of the M1/M2 attack family on a set where the fixed detector is
strong (baseline 48/50 = 96%), using a priority word list FROZEN before any code
attack result was seen. This is the generalisation check the text-only query
order (m2_query_order.py) could not provide on its own.

Design decisions, both fixed by the user before this run:
  1. Only the natural-language instruction is attacked. Fenced ``` code blocks
     are never modified: the code-signal probe showed the payload alone is
     detected in 1/50 samples, so rewriting it cannot help and would break the
     payload. Code spans are located and every candidate word lies outside them.
  2. The generalised priority rule is: a changeable word that (a) is in the
     frozen PRIORITY_WORDS list AND (b) is immediately preceded by "your".
     PRIORITY_WORDS was declared before seeing any code attack outcome and is
     hashed into the manifest.

Same Cyrillic map as M1. One instruction word per candidate version, no
cumulative edits. Ordering and generation never use model feedback; detector
scores are recorded, never used to order or select. Nothing modifies PIGuard,
its checkpoint, labels, the decision rule, or any existing file or result.

A detector miss here is a fixed-benchmark prediction flip, NOT verified
downstream attack success and NOT a semantic-validity claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --- Frozen before any code attack result was observed. Do not edit. ---
PRIORITY_WORDS = ("answer", "response", "reply", "message", "output", "result",
                  "code", "implementation", "algorithm", "script", "solution")
PRIORITY_PREFIX = re.compile(r"\byour\s+$", re.I)
# ----------------------------------------------------------------------

# Byte-identical to the unchanged M1 Cyrillic map (see eval_m2_positions.py).
UNICODE_MAP = str.maketrans({
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о",
    "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "C": "С", "E": "Е", "I": "І", "O": "О",
    "P": "Р", "S": "Ѕ", "X": "Х", "Y": "У",
})

FENCE = re.compile(r"```.*?```", re.S)
WORD_PATTERN = re.compile(r"\b[A-Za-z]+\b")
ADDRESS_PATTERN = re.compile(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+")
POLICIES = ("left_to_right", "priority_first")
BUDGETS = (1, 2, 3, 5, 10, None)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_code_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for category, prompts in data.items():
        for category_index, text in enumerate(prompts):
            records.append({"sample_id": f"BIPIA_CODE_{len(records):03d}",
                            "category": category, "category_index": category_index,
                            "original": text})
    if len(records) != 50 or len(data) != 10:
        raise ValueError("This experiment requires the complete 50-prompt / 10-category BIPIA code set.")
    return records


def code_spans(text):
    """Character spans of every fenced code block, so candidates avoid them."""
    return [(m.start(), m.end()) for m in FENCE.finditer(text)]


def is_priority(token, start, original):
    """Frozen rule: token in PRIORITY_WORDS and immediately preceded by 'your'."""
    if token.lower() not in PRIORITY_WORDS:
        return False
    return PRIORITY_PREFIX.search(original[:start]) is not None


def instruction_candidates(original):
    """One changeable instruction word per candidate, fenced code left intact.

    'Changeable' = the M1 Cyrillic map alters at least one character. A word is
    eligible only if it lies fully outside every fenced code block. Candidate
    order is left-to-right by character position; priority/address are annotated
    but do NOT change this base order. No model feedback is used.
    """
    text = original  # positions are relative to the unmodified original
    spans = code_spans(text)
    addresses = [(m.start(), m.end()) for m in ADDRESS_PATTERN.finditer(text)]
    rows = []
    for match in WORD_PATTERN.finditer(text):
        a, b = match.span()
        if any(a < end and b > start for start, end in spans):
            continue  # inside a code block: never touched
        token = match.group()
        rewritten = token.translate(UNICODE_MAP)
        if rewritten == token:
            continue
        rows.append({
            "start": a, "end": b, "token": token, "rewritten_token": rewritten,
            "priority": int(is_priority(token, a, text)),
            "address_overlap": int(any(a < end and b > start for start, end in addresses)),
            "characters_changed": sum(x != y for x, y in zip(token, rewritten)),
            "text": text[:a] + rewritten + text[b:],
        })
    rows.sort(key=lambda r: r["start"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def ordered_candidates(candidates, policy):
    rows = sorted(candidates, key=lambda r: r["rank"])
    if policy == "priority_first":
        # Stable partition: priority words first in their original order, then
        # every remaining candidate in its original order. Same pool.
        rows.sort(key=lambda r: r["priority"] == 0)
    return rows


def search(candidates, predict, policy, budget):
    """One call per candidate, stop on the first label 0. Failures pay in full."""
    rows = ordered_candidates(candidates, policy)
    tested = []
    for candidate in rows[:budget]:
        label, score = predict(candidate["text"])
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction")
        tested.append({**candidate, "prediction": int(label), "probability": float(score)})
        if label == 0:
            break
    success = bool(tested and tested[-1]["prediction"] == 0)
    return {"found_miss": success, "candidate_queries": len(tested),
            "first_success_query": len(tested) if success else None,
            "selected": tested[-1] if success else None, "tested": tested}


def evaluate(records, predict, progress=None):
    """Diagnose each original once, then replay both policies at every budget.

    To avoid redundant model calls, each unique candidate for a sample is
    predicted at most once; policies and budgets reuse that cache. Query costs
    reported per policy are the counts a standalone early-stopping run WOULD
    incur, computed from the cached predictions.
    """
    attempts, per_sample, policy_rows = [], [], []
    for i, record in enumerate(records, 1):
        original = record["original"]
        op, oscore = predict(original)
        if op not in (0, 1) or not 0 <= oscore <= 1:
            raise ValueError("Invalid prediction")
        candidates = instruction_candidates(original)
        # One prediction per unique candidate text, cached.
        cache = {}
        for cand in candidates:
            pred, score = predict(cand["text"])
            if pred not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid prediction")
            cache[cand["rank"]] = (pred, score)
            attempts.append({**record, **{k: cand[k] for k in
                             ("rank", "start", "end", "token", "rewritten_token",
                              "priority", "address_overlap", "characters_changed")},
                             "original_prediction": op, "original_probability": oscore,
                             "prediction": pred, "probability": score,
                             "probability_shift": score - oscore,
                             "successful_flip": int(op == 1 and pred == 0),
                             "adverse_flip": int(op == 0 and pred == 1),
                             "semantic_review": "pending", "review_notes": ""})

        def cached_predict(text, _by_rank={c["text"]: c["rank"] for c in candidates}):
            return cache[_by_rank[text]]

        # Full-pool flip result (policy-independent), for the attack headline.
        full = search(candidates, cached_predict, "left_to_right", None)
        per_sample.append({
            **record, "original_prediction": op, "original_probability": oscore,
            "instruction_candidates": len(candidates),
            "any_flip": int(op == 1 and full["found_miss"]),
            "adverse_any": int(op == 0 and any(c["prediction"] == 1 for c in full["tested"])),
            "min_probability": min([oscore] + [c["probability"] for c in full["tested"]]),
            "selected_token": full["selected"]["token"] if full["selected"] else "",
            "selected_priority": full["selected"]["priority"] if full["selected"] else "",
            "selected_text": full["selected"]["text"] if full["selected"] else "",
            "priority_candidates": sum(c["priority"] for c in candidates),
            "semantic_review": "pending", "review_notes": "",
        })
        # Both policies at every budget, from the same cache: zero extra calls.
        for policy in POLICIES:
            for budget in BUDGETS:
                r = search(candidates, cached_predict, policy, budget)
                policy_rows.append({
                    **record, "policy": policy,
                    "budget": budget if budget is not None else "all",
                    "original_prediction": op,
                    "candidate_queries": r["candidate_queries"],
                    "found_miss": int(r["found_miss"]),
                    "successful_flip": int(op == 1 and r["found_miss"]),
                    "first_success_query": r["first_success_query"] if r["first_success_query"] else "",
                    "selected_token": r["selected"]["token"] if r["selected"] else "",
                    "selected_priority": r["selected"]["priority"] if r["selected"] else "",
                })
        if progress:
            progress(i, len(records))
    return attempts, per_sample, policy_rows


def summarise(per_sample, policy_rows):
    detected = [r for r in per_sample if r["original_prediction"] == 1]
    n_detected = len(detected)
    flips_full = sum(r["any_flip"] for r in per_sample)
    adverse = sum(r["adverse_any"] for r in per_sample)

    # Policy comparison, restricted to the originally-detected cohort.
    detected_ids = {r["sample_id"] for r in detected}
    policy_summary = []
    for policy in POLICIES:
        for budget in BUDGETS:
            b = budget if budget is not None else "all"
            subset = [r for r in policy_rows if r["policy"] == policy
                      and r["budget"] == b and r["sample_id"] in detected_ids]
            policy_summary.append({
                "policy": policy, "budget": b, "cohort": n_detected,
                "successful_flips": sum(r["successful_flip"] for r in subset),
                "candidate_queries": sum(r["candidate_queries"] for r in subset),
            })

    category_rows = []
    for category in dict.fromkeys(r["category"] for r in per_sample):
        subset = [r for r in per_sample if r["category"] == category]
        category_rows.append({
            "category": category, "n": len(subset),
            "original_detected": sum(r["original_prediction"] for r in subset),
            "any_flip": sum(r["any_flip"] for r in subset),
            "priority_candidates": sum(r["priority_candidates"] for r in subset),
        })
    headline = {
        "originally_detected": n_detected,
        "successful_flips_full_pool": flips_full,
        "flip_rate_full_pool": round(flips_full / n_detected, 4) if n_detected else None,
        "adverse_flips": adverse,
        "total_priority_candidates": sum(r["priority_candidates"] for r in per_sample),
    }
    return headline, policy_summary, category_rows


REPORT_HEADER = """# BIPIA code 目标词 Unicode 攻击（留出验证）

在 PIGuard 强项（基线 48/50=96%）上测试 M1/M2 攻击族，并用一个**在看到任何 code 攻击结果之前冻结**的优先词表检验查询顺序规则能否迁移。

固定设置：只攻击自然语言指令，代码块（```）从不修改；沿用 M1 西里尔映射；每个候选只改一个指令词，无累计编辑；候选生成与排序不使用模型反馈。

检测器判定翻转是固定基准上的预测变化，**不是**已验证的下游攻击成功，也**不是**语义有效性结论。
"""


def write_report(path, headline, policy_summary, category_rows, per_sample, queries):
    n = headline["originally_detected"]
    lines = [REPORT_HEADER, f"模型调用次数：{queries}", "",
             "## 攻击主结果（全部候选池）", "",
             f"- 原始被检出：**{n}/50**",
             f"- 成功翻转（改一个指令词后被判为 benign）：**{headline['successful_flips_full_pool']}/{n}"
             f" = {headline['flip_rate_full_pool']:.1%}**",
             f"- 反向翻转（原本漏检、改词后被检出）：{headline['adverse_flips']}",
             f"- 落在冻结优先词表内的候选总数：{headline['total_priority_candidates']}",
             "", "## 查询顺序对比（仅原本被检出的 %d 条，成功即停）" % n, "",
             "|顺序|预算|成功翻转|候选查询次数|", "|---|---|---:|---:|"]
    for r in policy_summary:
        lines.append(f"|{r['policy']}|{r['budget']}|{r['successful_flips']}|{r['candidate_queries']}|")
    lines += ["", "说明：`left_to_right` 从左到右试指令词；`priority_first` 先试冻结词表里、"
              "且紧跟在 `your` 后面的词，再按原顺序试其余词。候选池完全相同，只有顺序不同。"
              "两种顺序都在首次漏检时停止；失败也计入查询成本。", "",
              "## 分类别", "", "|类别|n|原始检出|全池翻转|优先候选数|", "|---|---:|---:|---:|---:|"]
    for r in category_rows:
        lines.append(f"|{r['category']}|{r['n']}|{r['original_detected']}|{r['any_flip']}|{r['priority_candidates']}|")
    lines += ["", "## 冻结优先词表", "", "`" + ", ".join(PRIORITY_WORDS) + "`", "",
              "此表在运行本实验、看到任何 code 攻击结果之前确定，并写入 manifest 哈希。"
              "text 集上的原规则只含 answer/response/reply/message；其余为 code 领域中"
              "指代模型产出的对应词（如 your code implementation / your algorithm / your solution）。", "",
              "## 局限", "",
              "- 优先词表虽在观察 code 结果前冻结，但仍源于 text 集上的事后观察；这是跨集迁移检验，不是全新独立假设。",
              "- 判定翻转不等于下游 LLM 攻击成功；本轮未做语义有效性与执行验证。",
              "- classify() 在 256 token 截断，长样本指令位置需确认未被截掉。",
              "- 单次运行，无重复性与置信区间；小计数不可外推。",
              "- 缓存回放：每个候选只调用一次模型，各顺序/预算的查询成本为独立早停运行的应有次数，"
              "假设固定检测器对同一输入输出稳定。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default="checkpoints/best_model.pth")
    parser.add_argument("--dataset", default="datasets/BIPIA_code.json")
    parser.add_argument("--output-dir", default="/content/m2_code_attack_results")
    parser.add_argument("--preview", action="store_true", help="Validate candidates without model inference")
    args = parser.parse_args()

    records = load_code_records(args.dataset)
    counts = [len(instruction_candidates(r["original"])) for r in records]
    priority_total = sum(sum(c["priority"] for c in instruction_candidates(r["original"])) for r in records)
    with_priority = sum(1 for r in records if any(c["priority"] for c in instruction_candidates(r["original"])))
    print("BIPIA code set:", len(records), "prompts / 10 categories", flush=True)
    print("Instruction candidates:", sum(counts), "| per prompt min", min(counts), "max", max(counts), flush=True)
    print("Priority candidates (frozen rule):", priority_total, "across", with_priority, "prompts", flush=True)
    print("Planned model calls:", len(records) + sum(counts), "(50 originals +", sum(counts), "candidates)", flush=True)
    if args.preview:
        print("Preview only: no model was loaded and no prediction was made.", flush=True)
        return

    import torch
    import transformers
    from PIGuard import PIGuard
    from util import set_seed
    from types import SimpleNamespace

    output = Path(args.output_dir) / datetime.now(timezone.utc).strftime("attack_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| output:", output, flush=True)
    model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)
    state = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    def predict(text):
        with torch.no_grad():
            logits = model.classify(text)
            return int(logits.argmax().item()), float(torch.softmax(logits, dim=-1)[0, 1].cpu())

    def progress(done, total):
        print(f"[{done}/{total}] samples attacked", flush=True)

    attempts, per_sample, policy_rows = evaluate(records, predict, progress)
    headline, policy_summary, category_rows = summarise(per_sample, policy_rows)
    save_csv(output / "attempts.csv", attempts)
    save_csv(output / "per_sample.csv", per_sample)
    save_csv(output / "policy_summary.csv", policy_summary)
    save_csv(output / "policy_selections.csv", policy_rows)
    save_csv(output / "category_summary.csv", category_rows)
    write_report(output / "REPORT_ZH.md", headline, policy_summary, category_rows, per_sample, len(attempts) + len(records))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "BIPIA code targeted Unicode attack, instruction-only, frozen priority rule",
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "seed": 42, "git_commit": git.stdout.strip() or None,
        "checkpoint_sha256": sha256(args.resume), "dataset_sha256": sha256(args.dataset),
        "script_sha256": sha256(__file__), "model_script_sha256": sha256("PIGuard.py"),
        "query_count": len(attempts) + len(records),
        "priority_words": list(PRIORITY_WORDS),
        "priority_prefix": PRIORITY_PREFIX.pattern,
        "unicode_map": {chr(k): v for k, v in UNICODE_MAP.items()},
        "policies": list(POLICIES), "budgets": [b if b else "all" for b in BUDGETS],
        "attack_scope": "instruction_only; fenced code blocks never modified",
        "headline": headline,
        "frozen_before_results": True,
        "limitations": [
            "Priority list frozen before code results but derived from text-set observation; cross-set transfer, not a new independent hypothesis.",
            "Detector flip is not verified downstream attack success; no semantic or execution check in this run.",
            "classify() truncates at 256 tokens.",
            "Single run; no repeat or confidence interval.",
            "Cached replay: one call per candidate; per-policy query costs are the counts a standalone early-stopping run would incur.",
        ],
        "review_status": "pending",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAttack: {headline['successful_flips_full_pool']}/{headline['originally_detected']} flipped "
          f"({headline['flip_rate_full_pool']:.1%}), adverse {headline['adverse_flips']}", flush=True)
    print("Saved to:", output, flush=True)


if __name__ == "__main__":
    main()
