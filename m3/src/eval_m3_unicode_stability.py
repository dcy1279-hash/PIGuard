"""M3: paired Unicode stability and blind repair of the FIXED PIGuard.

Full official BIPIA text/code and NotInject; no fitting or checkpoint changes.
Each family edits the SAME eligible word occurrences independently. Full-pool
errors are adversarial existence rates, not real-world false-positive rates.
Repair takes only the current string, never the original, label or family.
Standard-library preview/tests run without torch. Inference uses official classify.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CYRILLIC = dict(zip("aceiopsxyACEIOPSXY", "асеіорѕхуАСЕІОРЅХУ"))
REVERSE = str.maketrans({v: k for k, v in CYRILLIC.items()})
FAMILIES = ("cyrillic", "fullwidth", "zero_width")
REPAIRS = ("nfkc", "mixed_script")
BUDGETS = (1, 3, 5, 10, None)
NI_FILES = ("NotInject_one.json", "NotInject_two.json", "NotInject_three.json")
CHECKPOINT_SHA256 = "ad7eed71bedf2d445a4d452d781bccab6b2101675c2b4f65edae86d444ca1c99"
ZWSP = "\u200b"
ASCII_WORD = re.compile(r"\b[A-Za-z]+\b")
UNICODE_WORD = re.compile(r"[^\W\d_]+(?:\u200b[^\W\d_]+)*", re.UNICODE)
PROTECTED = re.compile(
    r"```[\s\S]*?(?:```|$)|`[^`\n]*`|(?:https?://|www\.)[^\s<>]+"
    r"|[\w.+-]+@[\w.-]+|\"[^\"\n]*\"|(?<!\w)'[^'\n]*'(?!\w)")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def protected_spans(text):
    return [m.span() for m in PROTECTED.finditer(text)]


def overlaps(start, end, spans):
    return any(start < b and end > a for a, b in spans)


def load_records(root):
    root = Path(root)
    records = []
    for filename, dataset, expected_n, expected_c in (
        ("BIPIA_text.json", "text", 75, 15),
        ("BIPIA_code.json", "code", 50, 10),
    ):
        data = json.loads((root / filename).read_text(encoding="utf-8"))
        group = []
        for cat, texts in data.items():
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(f"Non-text in {filename}")
                group.append(dict(sample_id=f"BIPIA_{dataset.upper()}_{len(group):03d}",
                                  dataset=dataset, category=cat, category_index=i,
                                  label=1, source_file=filename, text=text))
        if len(group) != expected_n or len(data) != expected_c:
            raise ValueError(f"Incomplete official {dataset} dataset")
        records.extend(group)
    group = []
    for filename in NI_FILES:
        for item in json.loads((root / filename).read_text(encoding="utf-8")):
            group.append(dict(sample_id=f"NI_{len(group):03d}", dataset="benign",
                              category=item["category"], category_index=-1,
                              label=0, source_file=filename, text=item["prompt"]))
    if len(group) != 339 or len({r["category"] for r in group}) != 4:
        raise ValueError("Incomplete official NotInject dataset")
    records.extend(group)
    return records


def eligible_words(text):
    """Common support for all families, not positions picked from predictions.

    A >=2-letter ASCII word containing a M1-mappable character, outside code,
    URLs/emails and quoted literals. This safer scope differs from M2's full pool.
    """
    spans = protected_spans(text)
    return [(m.start(), m.end(), m.group()) for m in ASCII_WORD.finditer(text)
            if len(m.group()) >= 2 and any(c in CYRILLIC for c in m.group())
            and not overlaps(m.start(), m.end(), spans)]


def rewrite(token, family):
    if family == "cyrillic":
        return "".join(CYRILLIC.get(c, c) for c in token)
    if family == "fullwidth":
        # Identical substitution positions to Cyrillic, different representation.
        return "".join(chr(ord(c) + 0xFEE0) if c in CYRILLIC else c for c in token)
    if family == "zero_width":
        # One insertion inside the word. Edit count is NOT matched to other families.
        i = len(token) // 2
        return token[:i] + ZWSP + token[i:]
    raise ValueError(f"Unknown family {family}")


def candidates(text, family):
    result = []
    for rank, (a, b, token) in enumerate(eligible_words(text), 1):
        changed = rewrite(token, family)
        decoded = (changed.translate(REVERSE) if family == "cyrillic" else
                   unicodedata.normalize("NFKC", changed) if family == "fullwidth" else
                   changed.replace(ZWSP, ""))
        assert decoded == token, "Construction is not exactly reversible"
        result.append(dict(rank=rank, start=a, end=b, token=token,
                           rewritten_token=changed,
                           substitutions=sum(x != y for x, y in zip(token, changed))
                           if family != "zero_width" else 0,
                           insertions=int(family == "zero_width"),
                           changed_span_end=a + len(changed),
                           construction_reversible=1,
                           text=text[:a] + changed + text[b:]))
    return result


def repair(text, policy):
    """Blind conservative repair. Protected literals/addresses/code stay exact.

    NFKC normalises eligible Unicode letter runs, not entire source code.
    Mixed-script repair maps confusables only in a word with ASCII Latin AND
    mapped Cyrillic; pure Cyrillic is preserved. A completely Cyrillic-looking
    short word can remain unrepaired. ZWSP is removed only between ASCII letters
    after NFKC. Legitimate mixed-script text can still be harmed: measure controls.
    """
    if policy not in REPAIRS:
        raise ValueError(f"Unknown repair {policy}")
    spans = protected_spans(text)
    def replace(match):
        if overlaps(*match.span(), spans):
            return match.group()
        word = unicodedata.normalize("NFKC", match.group())
        if policy == "mixed_script":
            if any(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in word) \
                    and any(ord(c) in REVERSE for c in word):
                word = word.translate(REVERSE)
            word = re.sub(r"(?<=[A-Za-z])\u200b(?=[A-Za-z])", "", word)
        return word
    return UNICODE_WORD.sub(replace, text)


def validate_prediction(value):
    pred, score = value
    if pred not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid classifier output")
    return int(pred), float(score)


def audit_input(tokenizer, text, span=None):
    """Record full token length and actual offset coverage of the edited span.

    Official classify is unchanged (max_length=256). Offset mapping can be
    unavailable on a slow tokenizer; then visibility is unknown, never guessed.
    """
    ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    out = dict(token_count=len(ids), truncated=int(len(ids) > 256),
               edited_span_visibility="not_applicable" if span is None else "unknown")
    if span is not None and getattr(tokenizer, "is_fast", False):
        offsets = tokenizer(text, truncation=True, max_length=256,
                            return_offsets_mapping=True)["offset_mapping"]
        covered = {i for a, b in offsets for i in range(max(a, span[0]), min(b, span[1]))}
        expected = {i for i in range(*span) if not text[i].isspace() and text[i] != ZWSP}
        out["edited_span_visibility"] = "full" if expected <= covered else \
            "partial" if covered else "none"
    return out


def preview(records):
    rows = []
    for dataset in ("text", "code", "benign"):
        sub = [r for r in records if r["dataset"] == dataset]
        position_count = sum(len(eligible_words(r["text"])) for r in sub)
        # Per-sample caching of identical original/variant/repaired strings.
        calls = 0
        for r in sub:
            strings = {r["text"], *(repair(r["text"], p) for p in REPAIRS)}
            for f in FAMILIES:
                for c in candidates(r["text"], f):
                    strings.add(c["text"])
                    strings.update(repair(c["text"], p) for p in REPAIRS)
            calls += len(strings)
        rows.append(dict(dataset=dataset, prompts=len(sub),
                         prompts_with_candidates=sum(bool(eligible_words(r["text"])) for r in sub),
                         common_word_positions=position_count,
                         variants=position_count * len(FAMILIES),
                         planned_unique_model_calls=calls))
    return rows


def evaluate(records, predict, audit=None, completed=None):
    attempts, clean = [], []
    stats = Counter(model_calls=0, cache_hits=0)
    for record in records:
        cache = {}
        before = stats["model_calls"]
        def query(text):
            if text not in cache:
                cache[text] = validate_prediction(predict(text))
                stats["model_calls"] += 1
            else:
                stats["cache_hits"] += 1
            return cache[text]
        original = record["text"]
        op, oscore = query(original)
        base = {k: record[k] for k in ("sample_id", "dataset", "category", "label", "source_file")}
        original_audit = audit(original, None) if audit else {}
        cr = dict(**base, original=original, original_prediction=op,
                  original_probability=oscore, baseline_correct=int(op == record["label"]),
                  candidates_per_family=len(eligible_words(original)), **original_audit)
        for p in REPAIRS:
            fixed = repair(original, p)
            rp, rs = query(fixed)
            cr.update({f"{p}_prediction": rp, f"{p}_probability": rs,
                       f"{p}_changes_clean_text": int(fixed != original),
                       f"{p}_new_error": int(op == record["label"] and rp != record["label"])})
        local = []
        for family in FAMILIES:
            for candidate in candidates(original, family):
                pred, score = query(candidate["text"])
                row = dict(**base, **candidate, original_prediction=op,
                           original_probability=oscore, prediction=pred, probability=score,
                           probability_shift=score-oscore,
                           baseline_correct=cr["baseline_correct"], family=family,
                           new_error=int(op == record["label"] and pred != record["label"]),
                           baseline_error_corrected=int(op != record["label"] and pred == record["label"]),
                           original_token_count=original_audit.get("token_count", ""),
                           original_truncated=original_audit.get("truncated", ""),
                           review_status="pending")
                if audit:
                    row.update(audit(candidate["text"], (candidate["start"], candidate["changed_span_end"])))
                for p in REPAIRS:
                    fixed = repair(candidate["text"], p)
                    rp, rs = query(fixed)
                    row.update({f"{p}_prediction": rp, f"{p}_probability": rs,
                                f"{p}_exact_original": int(fixed == original),
                                f"{p}_restores_baseline_label": int(rp == op),
                                f"{p}_correct": int(rp == record["label"]),
                                f"{p}_introduces_error": int(pred == record["label"] and rp != record["label"])})
                local.append(row)
        cr["unique_model_calls"] = stats["model_calls"]-before
        attempts.extend(local)
        clean.append(cr)
        if completed:
            completed(cr, local, dict(stats))
    return attempts, clean, dict(stats)


def summarise(attempts, clean):
    by_sample = {}
    for a in attempts:
        by_sample.setdefault((a["sample_id"], a["family"]), []).append(a)
    clean_summary, family_summary, category_summary = [], [], []
    for dataset in ("text", "code", "benign"):
        group = [r for r in clean if r["dataset"] == dataset]
        if not group:
            continue
        correct = [r for r in group if r["baseline_correct"]]
        label = group[0]["label"]
        row = dict(dataset=dataset, expected_label=label, prompts=len(group),
                   baseline_correct=len(correct), baseline_errors=len(group)-len(correct),
                   eligible_baseline_correct=sum(r["candidates_per_family"] > 0 for r in correct),
                   originally_truncated=sum(r.get("truncated", 0) for r in group))
        for p in REPAIRS:
            row[f"{p}_new_clean_errors"] = sum(r[f"{p}_new_error"] for r in group)
            row[f"{p}_total_clean_errors"] = sum(r[f"{p}_prediction"] != label for r in group)
        clean_summary.append(row)
        for family in FAMILIES:
            for budget in BUDGETS:
                fs = dict(dataset=dataset, family=family, budget=budget or "all",
                          baseline_correct=len(correct),
                          eligible_baseline_correct=row["eligible_baseline_correct"],
                          new_error_samples=0, evaluated_variants=0, erroneous_variants=0,
                          new_errors_with_truncation=0, new_errors_not_fully_visible=0)
                for p in REPAIRS:
                    fs[f"{p}_first_error_recovered"] = 0
                    fs[f"{p}_postrepair_any_error_samples"] = 0
                    fs[f"{p}_introduced_error_variants"] = 0
                for r in correct:
                    sub = by_sample.get((r["sample_id"], family), [])[:budget]
                    bad = [a for a in sub if a["new_error"]]
                    fs["new_error_samples"] += bool(bad)
                    fs["evaluated_variants"] += len(sub)
                    fs["erroneous_variants"] += len(bad)
                    fs["new_errors_with_truncation"] += sum(bool(a.get("truncated") or a.get("original_truncated")) for a in bad)
                    fs["new_errors_not_fully_visible"] += sum(a.get("edited_span_visibility", "unknown") != "full" for a in bad)
                    for p in REPAIRS:
                        fs[f"{p}_first_error_recovered"] += bool(bad and bad[0][f"{p}_correct"])
                        fs[f"{p}_postrepair_any_error_samples"] += any(not a[f"{p}_correct"] for a in sub)
                        fs[f"{p}_introduced_error_variants"] += sum(a[f"{p}_introduces_error"] for a in sub)
                fs["new_error_rate"] = fs["new_error_samples"]/len(correct) if correct else None
                family_summary.append(fs)
            for category in dict.fromkeys(r["category"] for r in group):
                sub = [r for r in correct if r["category"] == category]
                category_summary.append(dict(dataset=dataset, category=category, family=family,
                    total_prompts=sum(r["category"] == category for r in group),
                    baseline_correct=len(sub),
                    with_candidates=sum(r["candidates_per_family"] > 0 for r in sub),
                    new_error_samples=sum(any(a["new_error"] for a in by_sample.get((r["sample_id"], family), [])) for r in sub)))
    return clean_summary, family_summary, category_summary


LIMITATIONS = [
    "Existing M2 datasets are reused; this is a paired robustness extension, not held-out generalisation.",
    "Official payload-only labels are retained. Detector flips do not prove downstream prompt-injection success.",
    "Common word positions/query budgets, but ZWSP insertion has a different edit count from substitutions.",
    "Protected code, quoted literals, URLs/emails and one-letter words excluded: not an exact M2 pool reproduction.",
    "Full-pool rates mean an adversarial version exists; they are not natural/deployment error rates.",
    "Reversible character construction is not semantic/functionality validation; manual review pending.",
    "Repairs are blind heuristic front-end controls, not changes to PIGuard and not guaranteed safe for all scripts.",
    "Official classify truncates at 256 tokens. Visibility logs identify risk, not causal attribution.",
    "Single fixed-model run. Repeated templates/candidates are dependent; no population-level causal claims.",
]


def write_report(path, cs, fs, stats):
    lines = ["# M3 字符变化、双向错误与还原对照", "",
             "## 这次做什么", "",
             "固定 PIGuard，对同一批词分别做西里尔同形字、全角字母、词内零宽字符变换。",
             "同时观察攻击漏检与正常文本误报。两个还原规则只读取当前文本，不读取原文或标签。",
             "三种方法位置和查询预算一致，但改动字符数量不完全一致，不能据此认定字符族的因果强弱。", "",
             f"实际模型调用：{stats['model_calls']}；相同字符串缓存复用：{stats['cache_hits']}。", "",
             "## 原文与正常输入的还原对照", "",
             "|数据|原文正确/总数|可变换且原文正确|NFKC新增错误|混合脚本还原新增错误|",
             "|---|---:|---:|---:|---:|"]
    for r in cs:
        lines.append(f"|{r['dataset']}|{r['baseline_correct']}/{r['prompts']}|{r['eligible_baseline_correct']}|"
                     f"{r['nfkc_new_clean_errors']}|{r['mixed_script_new_clean_errors']}|")
    lines += ["", "## 全候选池：两种错误与修复", "",
              "攻击数据里的新增错误是漏检；benign 里的新增错误是误报。分母均为原文判断正确的样本，",
              "原本已错不算新增成功。无候选的样本保留在总分母中，并另报覆盖数。", "",
              "|数据|变换|新增错误样本/原文正确|NFKC后仍存在错误的样本|混合脚本还原后仍存在错误的样本|",
              "|---|---|---:|---:|---:|"]
    for r in fs:
        if r["budget"] == "all":
            lines.append(f"|{r['dataset']}|{r['family']}|{r['new_error_samples']}/{r['baseline_correct']}|"
                         f"{r['nfkc_postrepair_any_error_samples']}|{r['mixed_script_postrepair_any_error_samples']}|")
    lines += ["", "这些是尝试多个版本后的最坏情况存在率，不是日常误报率。",
              "还原后仍有错误的统计也包括：原本修改后正确、但被还原规则弄错的候选。", "",
              "## 截断与位置可见性诊断（候选数，不是独立样本数）", "",
              "|数据|变换|错误候选数|其中原文或变体超256 token|其中修改位置不完整可见/未知|",
              "|---|---|---:|---:|---:|"]
    for r in fs:
        if r["budget"] == "all":
            lines.append(f"|{r['dataset']}|{r['family']}|{r['erroneous_variants']}|{r['new_errors_with_truncation']}|{r['new_errors_not_fully_visible']}|")
    lines += ["", "存在截断并不证明截断造成错误；需结合 attempts.csv 中原文/变体长度与具体案例检查。", "",
              "## 相同查询预算对比", "",
              "|数据|变换|每样本预算|新增错误/原文正确|原文正确样本上的变体查询机会数|",
              "|---|---|---|---:|---:|"]
    for r in fs:
        if r["budget"] != "all":
            lines.append(f"|{r['dataset']}|{r['family']}|{r['budget']}|{r['new_error_samples']}/{r['baseline_correct']}|{r['evaluated_variants']}|")
    lines += ["", "## 结果应该怎样解释", "",
              "- 若新变换有效：说明固定检测器存在另一种表面形式敏感性；不是下游成功证明。",
              "- 若新变换无效：它界定攻击范围，同样需要保留。",
              "- 若还原恢复判断：支持字符表示有关的解释；原样还原本就应得到原预测，不单独算新算法突破。",
              "- 若还原误伤正常输入：修复存在代价，不能只报恢复数量。",
              "- 若修改位置不可见或输入被截断：单独分析，不把变化一概归因于该词语义。", "",
              "## 局限与待人工检查", ""]
    lines.extend("- " + s for s in LIMITATIONS)
    lines += ["", "review_cases.json 提供每组固定顺序的代表性成功/失败案例，包含原文、修改后文本和还原文本。",
              "图表是以上结果的展示，不是额外独立实验。报告由程序输出，尚需结合案例撰写最终 M3 结论。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_review(path, attempts, clean):
    originals = {r["sample_id"]: r["original"] for r in clean}
    cases, count = [], Counter()
    for row in attempts:
        if not row["baseline_correct"]:
            continue
        key = (row["dataset"], row["family"], row["new_error"])
        if count[key] >= 3:
            continue
        count[key] += 1
        cases.append({**row, "original": originals[row["sample_id"]],
                      "nfkc_text": repair(row["text"], "nfkc"),
                      "mixed_script_text": repair(row["text"], "mixed_script"),
                      "manual_meaning_review": "pending", "manual_notes": ""})
    Path(path).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("--resume", default="checkpoints/best_model.pth")
    ap.add_argument("--output-dir", default="/content/m3_unicode_results")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    records = load_records(args.dataset_root)
    plan = preview(records)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    planned_calls = sum(r["planned_unique_model_calls"] for r in plan)
    print("Full dataset:", len(records), "| planned unique model calls:", planned_calls, flush=True)
    if args.preview:
        print("Preview only; no model loaded or experimental result obtained.")
        return

    import torch
    import transformers
    from types import SimpleNamespace
    from PIGuard import PIGuard
    from util import set_seed
    set_seed(SimpleNamespace(seed=42))
    if digest(args.resume) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the verified M2 author checkpoint; stop rather than compare different baselines.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = PIGuard("microsoft/deberta-v3-base", 2, device)
    model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True), strict=True)
    model.to(device).eval()
    def predict(text):
        with torch.inference_mode():
            logits = model.classify(text)
            return int(logits.argmax().item()), float(torch.softmax(logits, dim=-1)[0, 1].cpu())

    out = Path(args.output_dir) / datetime.now(timezone.utc).strftime("stability_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    partial = out / "partial_samples.jsonl"
    done = 0
    def completed(clean_row, rows, stats):
        nonlocal done
        done += 1
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(clean=clean_row, attempts=rows, stats=stats), ensure_ascii=False) + "\n")
        if done % 10 == 0 or done == len(records):
            print(f"[{done}/{len(records)}] calls={stats['model_calls']}/{planned_calls} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    attempts, clean, stats = evaluate(records, predict,
        audit=lambda t, s: audit_input(model.tokenizer, t, s), completed=completed)
    if stats["model_calls"] != planned_calls:
        raise RuntimeError("Planned and actual model calls differ")
    cs, fs, categories = summarise(attempts, clean)
    save_csv(out / "attempts.csv", attempts)
    save_csv(out / "clean_controls.csv", clean)
    save_csv(out / "clean_summary.csv", cs)
    save_csv(out / "family_budget_summary.csv", fs)
    save_csv(out / "category_summary.csv", categories)
    save_review(out / "review_cases.json", attempts, clean)
    write_report(out / "REPORT_ZH.md", cs, fs, stats)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = dict(experiment="M3 paired Unicode stability and blind repair", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), record_count=len(records),
        plan=plan, stats=stats, elapsed_seconds=time.monotonic()-started,
        device=str(device), seed=42, torch=torch.__version__, transformers=transformers.__version__,
        git_commit=git.stdout.strip() or None, script_sha256=digest(__file__),
        model_script_sha256=digest(Path(__file__).resolve().parent / "PIGuard.py"),
        checkpoint_sha256=digest(args.resume),
        datasets_sha256={name: digest(Path(args.dataset_root)/name) for name in
                         ("BIPIA_text.json", "BIPIA_code.json", *NI_FILES)},
        families=list(FAMILIES), repairs=list(REPAIRS), budgets=list(BUDGETS),
        cyrillic_map=CYRILLIC, protected_pattern=PROTECTED.pattern,
        baseline_decision="official classify, argmax, max_length=256",
        common_scope="ASCII word >=2 letters with a mapped character; exclude protected spans",
        repair_has_original_or_label_access=False, independent_holdout=False,
        manual_review_status="pending", limitations=LIMITATIONS)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print((out / "REPORT_ZH.md").read_text(encoding="utf-8"), flush=True)
    print("COMPLETE:", out, flush=True)


if __name__ == "__main__":
    main()
