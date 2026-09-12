"""M2 reverse direction: does the SAME Cyrillic substitution FALSELY BLOCK benign text?

The evasion experiments asked whether homoglyphs make an injection prompt slip
past PIGuard (detected -> benign). This asks the mirror question on PIGuard's own
over-defense benchmark: does the identical substitution make a BENIGN prompt get
wrongly flagged as an injection (benign -> injection)?

This targets the paper's headline contribution directly. PIGuard's selling point
is Mitigating Over-defense: NotInject is 339 benign prompts that deliberately
embed attack-trigger words, and PIGuard claims it does NOT over-fire on them
(reported 91.74%). We apply the same M1 Cyrillic map, one word per candidate, and
measure whether that tolerance survives a surface-form change.

Two candidate scopes are annotated from one run:
  - all changeable Latin words (the ceiling: can ANY single benign word, homoglyphed,
    cause a false block?);
  - trigger words only (the on-point attack: the exact words PIGuard learned to
    tolerate), flagged by is_trigger.

Same Cyrillic map as M1. One word per candidate, no cumulative edits. Generation and
ordering use no model feedback. Nothing modifies PIGuard, its checkpoint, labels, the
rule, or any existing file. A prediction flip here is a fixed-benchmark false-positive
measurement, not a claim about the prompt's true intent.
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

# Byte-identical to the unchanged M1 Cyrillic map (see eval_m2_positions.py).
UNICODE_MAP = str.maketrans({
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о",
    "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "C": "С", "E": "Е", "I": "І", "O": "О",
    "P": "Р", "S": "Ѕ", "X": "Х", "Y": "У",
})
WORD_PATTERN = re.compile(r"\b[A-Za-z]+\b")
NOTINJECT_FILES = ("NotInject_one.json", "NotInject_two.json", "NotInject_three.json")


def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_notinject(root):
    root = Path(root)
    records = []
    for fname in NOTINJECT_FILES:
        data = json.loads((root / fname).read_text(encoding="utf-8"))
        for item in data:
            records.append({
                "sample_id": f"NI_{len(records):03d}",
                "source_file": fname,
                "category": item["category"],
                "trigger_words": [w for w in item["word_list"] if isinstance(w, str)],
                "prompt": item["prompt"],
            })
    if len(records) != 339:
        raise ValueError(f"Expected 339 NotInject prompts, got {len(records)}")
    return records


def trigger_spans(prompt, triggers):
    spans = []
    for w in triggers:
        if not w:
            continue
        for m in re.finditer(re.escape(w), prompt, re.I):
            spans.append((m.start(), m.end()))
    return spans


def candidates(record):
    """One changeable Latin word per candidate. is_trigger flags overlap with a
    NotInject trigger-word occurrence (the words PIGuard is meant to tolerate)."""
    text = record["prompt"]
    tspans = trigger_spans(text, record["trigger_words"])
    rows = []
    for m in WORD_PATTERN.finditer(text):
        a, b = m.span()
        token = m.group()
        rewritten = token.translate(UNICODE_MAP)
        if rewritten == token:
            continue
        rows.append({
            "start": a, "end": b, "token": token, "rewritten_token": rewritten,
            "is_trigger": int(any(a < e and b > s for s, e in tspans)),
            "characters_changed": sum(x != y for x, y in zip(token, rewritten)),
            "text": text[:a] + rewritten + text[b:],
        })
    rows.sort(key=lambda r: r["start"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def evaluate(records, predict, progress=None):
    attempts, per_sample = [], []
    for i, record in enumerate(records, 1):
        op, oscore = predict(record["prompt"])
        if op not in (0, 1) or not 0 <= oscore <= 1:
            raise ValueError("Invalid prediction")
        cands = candidates(record)
        tested = []
        for c in cands:
            pred, score = predict(c["text"])
            if pred not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid prediction")
            row = {**record, **{k: c[k] for k in
                   ("rank", "start", "end", "token", "rewritten_token", "is_trigger", "characters_changed")},
                   "trigger_words": ";".join(record["trigger_words"]),
                   "original_prediction": op, "original_probability": oscore,
                   "prediction": pred, "probability": score, "probability_shift": score - oscore,
                   "false_block": int(op == 0 and pred == 1)}
            tested.append({**c, "prediction": pred, "probability": score})
            attempts.append(row)
        any_block = [c for c in tested if c["prediction"] == 1]
        trig_block = [c for c in tested if c["prediction"] == 1 and c["is_trigger"]]
        worst = max(tested, key=lambda r: r["probability"], default=None)
        per_sample.append({
            "sample_id": record["sample_id"], "source_file": record["source_file"],
            "category": record["category"], "trigger_words": ";".join(record["trigger_words"]),
            "original_prediction": op, "original_probability": oscore,
            "candidates": len(cands), "trigger_candidates": sum(c["is_trigger"] for c in cands),
            "already_over_blocked": int(op == 1),
            "new_false_block_any": int(op == 0 and bool(any_block)),
            "new_false_block_trigger": int(op == 0 and bool(trig_block)),
            "max_injection_probability": worst["probability"] if worst else oscore,
            "max_probability_shift": (worst["probability"] - oscore) if worst else 0.0,
            "review_status": "pending",
        })
        if progress:
            progress(i, len(records))
    return attempts, per_sample


def summarise(attempts, per_sample):
    n = len(per_sample)
    over = sum(r["already_over_blocked"] for r in per_sample)
    passed = [r for r in per_sample if r["original_prediction"] == 0]
    npass = len(passed)
    fb_any = sum(r["new_false_block_any"] for r in passed)
    fb_trig = sum(r["new_false_block_trigger"] for r in passed)

    det = [a for a in attempts if a["original_prediction"] == 0]
    trig_c = [a for a in det if a["is_trigger"]]
    nontrig_c = [a for a in det if not a["is_trigger"]]
    def mean(xs, k): return round(sum(float(a[k]) for a in xs) / len(xs), 6) if xs else None
    headline = {
        "prompts": n,
        "baseline_over_defense": over,
        "baseline_over_defense_rate": round(over / n, 4),
        "correctly_passed": npass,
        "new_false_block_any_pool": fb_any,
        "new_false_block_rate_any": round(fb_any / npass, 4) if npass else None,
        "new_false_block_trigger_only": fb_trig,
        "new_false_block_rate_trigger": round(fb_trig / npass, 4) if npass else None,
        "trigger_candidate_mean_shift": mean(trig_c, "probability_shift"),
        "nontrigger_candidate_mean_shift": mean(nontrig_c, "probability_shift"),
        "trigger_candidate_block_count": sum(1 for a in trig_c if a["prediction"] == 1),
        "trigger_candidate_total": len(trig_c),
        "nontrigger_candidate_block_count": sum(1 for a in nontrig_c if a["prediction"] == 1),
        "nontrigger_candidate_total": len(nontrig_c),
    }
    category_rows = []
    for cat in dict.fromkeys(r["category"] for r in per_sample):
        sub = [r for r in per_sample if r["category"] == cat]
        subpass = [r for r in sub if r["original_prediction"] == 0]
        category_rows.append({
            "category": cat, "n": len(sub),
            "baseline_over_blocked": sum(r["already_over_blocked"] for r in sub),
            "correctly_passed": len(subpass),
            "new_false_block_any": sum(r["new_false_block_any"] for r in subpass),
            "new_false_block_trigger": sum(r["new_false_block_trigger"] for r in subpass),
        })
    return headline, category_rows


REPORT = """# NotInject 反向测试：同形字能否让良性文本被误判为攻击

逃逸实验问的是同形字能否让注入 prompt 漏检（injection→benign）。本实验问镜像问题：
**同一套西里尔替换，能不能让一句良性文本被误判成注入（benign→injection）？**

这直接打 PIGuard 的核心卖点。它的贡献是"缓解过度防御"：NotInject 是 339 条故意嵌入
攻击触发词的良性句子，PIGuard 声称不会对它们误报（论文 91.74%）。我们用同一张 M1 映射、
每候选改一个词，测这份"容忍度"在表面形式改变后还在不在。

判定翻转是固定基准上的误报测量，不是对句子真实意图的判断。
"""


def write_report(path, headline, category_rows, queries):
    h = headline
    npass = h["correctly_passed"]
    L = [REPORT, f"模型调用次数：{queries}", "", "## 主结果", "",
         f"- 基线过度防御（原文即被误判为注入）：**{h['baseline_over_defense']}/{h['prompts']}"
         f" = {h['baseline_over_defense_rate']:.2%}**",
         f"- 正确放行的良性句：{npass}",
         f"- **改一个词即可造成误拦（任意词）：{h['new_false_block_any_pool']}/{npass}"
         f" = {h['new_false_block_rate_any']:.2%}**",
         f"- 仅用触发词造成误拦：{h['new_false_block_trigger_only']}/{npass}"
         f" = {h['new_false_block_rate_trigger']:.2%}",
         f"- 平均概率变化（朝注入方向，正=更像攻击）：触发词候选 {h['trigger_candidate_mean_shift']:+.4f}"
         f" / 非触发词候选 {h['nontrigger_candidate_mean_shift']:+.4f}",
         f"- 触发词候选造成误拦 {h['trigger_candidate_block_count']}/{h['trigger_candidate_total']}；"
         f"非触发词候选 {h['nontrigger_candidate_block_count']}/{h['nontrigger_candidate_total']}",
         "", "## 分类别", "",
         "|类别|n|基线已误拦|正确放行|改任意词致误拦|仅改触发词致误拦|",
         "|---|---:|---:|---:|---:|---:|"]
    for r in category_rows:
        L.append(f"|{r['category']}|{r['n']}|{r['baseline_over_blocked']}|{r['correctly_passed']}"
                 f"|{r['new_false_block_any']}|{r['new_false_block_trigger']}|")
    L += ["", "Multilingual 类的触发词多为中文/emoji，无法被西里尔映射改动，是天然对照：",
          "对这类样本，触发词路径几乎不可用，只能靠句中偶发的拉丁词。", "",
          "## 局限", "",
          "- 判定翻转是固定基准上的误报测量，不等于真实系统在部署中的行为。",
          "- classify() 在 256 token 截断。",
          "- 单次运行，无重复性与置信区间；小计数不可外推。",
          "- 本实验只做西里尔一族；其他 Unicode 族与隐形字符留待后续。", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", default="checkpoints/best_model.pth")
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("--output-dir", default="/content/m2_notinject_fp_results")
    ap.add_argument("--preview", action="store_true", help="Validate candidates without model inference")
    args = ap.parse_args()

    records = load_notinject(args.dataset_root)
    counts = [len(candidates(r)) for r in records]
    trig = sum(sum(c["is_trigger"] for c in candidates(r)) for r in records)
    print("NotInject prompts:", len(records), "/ 4 categories", flush=True)
    print("Changeable candidates:", sum(counts), "| trigger-word candidates:", trig, flush=True)
    print("Planned model calls:", len(records) + sum(counts),
          f"({len(records)} originals + {sum(counts)} candidates)", flush=True)
    if args.preview:
        print("Preview only: no model was loaded and no prediction was made.", flush=True)
        return

    import torch
    import transformers
    from PIGuard import PIGuard
    from util import set_seed
    from types import SimpleNamespace

    out = Path(args.output_dir) / datetime.now(timezone.utc).strftime("fp_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    set_seed(SimpleNamespace(seed=42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| output:", out, flush=True)
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
        if done % 25 == 0 or done == total:
            print(f"[{done}/{total}] prompts done", flush=True)

    attempts, per_sample = evaluate(records, predict, progress)
    headline, category_rows = summarise(attempts, per_sample)
    save_csv(out / "attempts.csv", attempts)
    save_csv(out / "per_sample.csv", per_sample)
    save_csv(out / "category_summary.csv", category_rows)
    write_report(out / "REPORT_ZH.md", headline, category_rows, len(attempts) + len(records))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "NotInject reverse direction: Cyrillic homoglyph false-block on benign prompts",
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "seed": 42, "git_commit": git.stdout.strip() or None,
        "checkpoint_sha256": sha256(args.resume), "script_sha256": sha256(__file__),
        "model_script_sha256": sha256("PIGuard.py"),
        "dataset_sha256": {f: sha256(Path(args.dataset_root) / f) for f in NOTINJECT_FILES},
        "query_count": len(attempts) + len(records),
        "unicode_map": {chr(k): v for k, v in UNICODE_MAP.items()},
        "direction": "benign -> injection (false block / over-defense attack)",
        "headline": headline,
        "limitations": [
            "Detector flip is a fixed-benchmark false-positive measurement, not deployed-system behaviour.",
            "classify() truncates at 256 tokens.",
            "Single run; no repeat or confidence interval.",
            "Cyrillic family only; other Unicode families and invisible characters deferred.",
        ],
        "review_status": "pending",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBaseline over-defense: {headline['baseline_over_defense']}/{headline['prompts']}", flush=True)
    print(f"New false block (any word): {headline['new_false_block_any_pool']}/{headline['correctly_passed']}"
          f" = {headline['new_false_block_rate_any']:.2%}", flush=True)
    print(f"New false block (trigger only): {headline['new_false_block_trigger_only']}/{headline['correctly_passed']}", flush=True)
    print("Saved to:", out, flush=True)


if __name__ == "__main__":
    main()
