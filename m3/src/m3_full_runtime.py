"""Fresh-model execution and English result readers for the full M3 workbook.

Historical cohort definitions may be reused; historical predictions never act
as the oracle. Existing protocol sources, weights and budgets are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


TASKS = ("baseline", "stability", "gradient", "combination", "families", "selector", "transfer")
CORE_TASKS = ("baseline", "combination", "families")
OPTIONAL_TASKS = ("gradient", "selector", "stability", "transfer")


def selected_tasks(options=None):
    """Keep whole fixed protocols; switches select experiments, not easier samples."""
    options = {} if options is None else options
    if set(options) - set(OPTIONAL_TASKS) or any(type(value) is not bool for value in options.values()):
        raise ValueError("Optional experiment switches must be known Boolean values.")
    if options.get("transfer") and not options.get("stability"):
        raise ValueError("Direct transfer requires the current-run stability experiment. Enable RUN_STABILITY too.")
    return tuple(task for task in TASKS if task in CORE_TASKS or options.get(task, False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_bundle(root):
    root = Path(root)
    inventory = read_json(root / "inventory.json")
    for name, expected in inventory["files_sha256"].items():
        if sha(root / name) != expected:
            raise RuntimeError(f"Source/data/plan differs from the fixed workbook: {name}")
    return inventory


def stream(command, cwd, logfile):
    """Stream real progress; keep complete legacy reports in the log, not the English display."""
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    with Path(logfile).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                # Existing benchmark text and source reports stay unchanged on disk.
                if not any("\u4e00" <= char <= "\u9fff" for char in line):
                    print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise subprocess.CalledProcessError(status, command)


def result_files(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def completed_result(session, task):
    session = Path(session)
    entry = read_json(session / "runs.json").get(task)
    if not entry or entry["status"] != "complete":
        raise RuntimeError(f"Run the fresh {task} stage first; old evidence is not a fallback.")
    path = Path(entry["result_dir"])
    if result_files(path) != entry["result_files_sha256"]:
        raise RuntimeError(f"Saved {task} result files changed or are missing.")
    return path


def run_stage(root, session, python, checkpoint, task, retry_incomplete=False):
    root, session, checkpoint = Path(root), Path(session), Path(checkpoint)
    if task not in TASKS:
        raise ValueError(task)
    verify_bundle(root)
    from eval_m3_unicode_stability import CHECKPOINT_SHA256
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Wrong author checkpoint.")
    session.mkdir(parents=True, exist_ok=True)
    identity = {"inventory_sha256": sha(root / "inventory.json"), "checkpoint_sha256": sha(checkpoint)}
    identity_file = session / "identity.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise RuntimeError("Session belongs to different code, plans or weights. Use a new session.")
    write_json(identity_file, identity)
    index_path = session / "runs.json"
    index = read_json(index_path) if index_path.exists() else {}
    if task in index and index[task]["status"] == "complete":
        result = completed_result(session, task)
        print(f"{task}: reusing this session's completed native run; no repeated inference: {result}", flush=True)
        return result
    if task in index and not retry_incomplete:
        raise RuntimeError("An earlier attempt was incomplete. Its partial files are preserved. Set RETRY_INCOMPLETE=True only if you accept restarting this stage.")
    output = session / task / time.strftime("attempt_%Y%m%dT%H%M%S", time.gmtime())
    output.mkdir(parents=True, exist_ok=False)
    source, plans = root / "src", root / "plans"
    index[task] = {"status": "running", "attempt_dir": str(output), "fresh_model_inference": True}
    write_json(index_path, index)
    result = None
    try:
        if task in ("baseline", "families"):
            command = [python, "-u", source / "m3_full_runtime.py", "--worker", task,
                       "--root", root, "--checkpoint", checkpoint, "--output", output]
            stream(command, source, output / "run.log")
            result = output / "result"
        elif task in ("gradient", "combination", "selector"):
            script, member, prefix = {
                "gradient": ("eval_m3_gradient_text_full.py", "text_plan.json", "gradient_text_"),
                "combination": ("eval_m3_blackbox_combo.py", "combo_plan.json", "blackbox_combo_"),
                "selector": ("eval_m3_selector.py", "selector_plan.json", "selector_")}[task]
            raw = (plans / member).read_bytes()
            plan = output / member
            plan.write_bytes(raw)
            write_json(output / "preparation.json", {"plan_sha256": sha(plan)})
            stream([python, "-u", source / script, "--plan", plan, "--repo", source,
                    "--checkpoint", checkpoint, "--output-dir", output], source, output / "run.log")
            matches = list(output.glob(prefix + "*"))
            result = next((p for p in matches if (p / "manifest.json").is_file()
                           and read_json(p / "manifest.json").get("complete") is True), None)
        elif task == "stability":
            stream([python, "-u", source / "eval_m3_unicode_stability.py", "--dataset-root", root / "datasets",
                    "--resume", checkpoint, "--output-dir", output], source, output / "run.log")
            result = next((p for p in output.glob("stability_*") if (p / "manifest.json").is_file()
                           and read_json(p / "manifest.json").get("complete") is True), None)
        elif task == "transfer":
            stability = completed_result(session, "stability")
            stream([python, "-u", source / "eval_m3_transfer.py", "--source", stability,
                    "--output-dir", output], source, output / "run.log")
            frozen = next(output.glob("frozen_*/frozen_plan.json"))
            stream([python, "-u", source / "eval_m3_transfer.py", "--plan", frozen,
                    "--batch-size", "4", "--output-dir", output], source, output / "run.log")
            result = next((p for p in output.glob("transfer_*") if (p / "manifest.json").is_file()
                           and read_json(p / "manifest.json").get("complete") is True), None)
        if result is None or read_json(result / "manifest.json").get("complete") is not True:
            raise RuntimeError("No complete new result was generated. Do not display the historical result instead.")
        verify_bundle(root)
        index[task].update(status="complete", result_dir=str(result), result_files_sha256=result_files(result))
        write_json(index_path, index)
        print(f"FRESH {task} complete: {result}", flush=True)
        return result
    except BaseException:
        index[task].update(status="incomplete", error=traceback.format_exc())
        write_json(index_path, index)
        raise


def baseline_worker(root, checkpoint, output):
    import torch
    import eval_m3_gradient_pilot as pilot
    import eval_m3_unicode_stability as rules
    model, device = pilot.load_piguard(root / "src", checkpoint)
    print("PIGuard architecture and checkpoint loaded; actual device:", device, flush=True)
    records = rules.load_records(root / "datasets")
    original = read_json(root / "plans/text_plan.json")["selected"][0]["original"]
    candidates = [{"condition": "original", "text": original}]
    for family in rules.FAMILIES:
        candidate = next(iter(rules.candidates(original, family)))
        candidates.append({"condition": family, "text": candidate["text"]})
    rows = []
    versions = [(p, p._version) for p in model.parameters()]
    for spec in candidates:
        with torch.inference_mode():
            logits = model.classify(spec["text"])
            row = {**spec, "prediction": int(logits.argmax(-1).item()),
                   "injection_probability": float(logits.softmax(-1)[0, 1].cpu()),
                   "visible_text": spec["text"].replace("\u200b", "[U+200B]"),
                   "escaped_text": spec["text"].encode("unicode_escape").decode()}
        rows.append(row)
        print(row["condition"], "label", row["prediction"], "score", row["injection_probability"], flush=True)
    if any(p._version != version for p, version in versions):
        raise RuntimeError("Baseline model parameter changed.")
    result = output / "result"
    result.mkdir()
    write_json(result / "predictions.json", rows)
    write_json(result / "manifest.json", {"complete": True, "fresh_model_inference": True,
               "model": "microsoft/deberta-v3-base + original PIGuard classifier checkpoint",
               "checkpoint_sha256": sha(checkpoint), "model_source_sha256": sha(root / "src/PIGuard.py"),
               "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
               "native_window": 256, "dataset_records": len(records), "native_predictions": len(rows),
               "parameter_updates": 0, "scope": "Illustrative native predictions, not an attack-success benchmark."})


def family_worker(root, checkpoint, output):
    import gc
    import torch
    import eval_m3_complementarity as family
    import eval_m3_gradient_pilot as pilot
    plan = read_json(root / "plans/family_plan.json")
    result = output / "result"
    result.mkdir()
    all_outcomes, stats = [], {}
    for name, selected_key, baseline_key in (("piguard", "source_selected", "source_baseline"),
                                               ("protectai", "target_selected", "target_baseline")):
        directory = result / name
        directory.mkdir()
        if name == "piguard":
            model, device = pilot.load_piguard(root / "src", checkpoint)
            tokenizer, limit = model.tokenizer, 256
        else:
            tokenizer, model, device = family.reader.load_target()
            limit = 512
        print(f"Fresh matched families: {name}, device={device}, window={limit}", flush=True)
        versions = [(p, p._version) for p in model.parameters()]
        cache, baseline, outcomes, traces = {}, {}, [], []
        def predict(text):
            if text not in cache:
                with torch.inference_mode():
                    if name == "piguard":
                        logits = model.classify(text)
                    else:
                        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=limit)
                        logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
                    cache[text] = (int(logits.argmax(-1).item()), float(logits.softmax(-1)[0, 1].cpu()))
            return cache[text]
        for i, record in enumerate(plan["originals"], 1):
            label, probability = predict(record["original"])
            baseline[record["sample_id"]] = {"prediction": label, "probability": probability}
            if label != plan[baseline_key][record["sample_id"]]["prediction"]:
                raise RuntimeError(f"{name} original changed: {record['sample_id']}; stop without changing the denominator.")
            if i % 15 == 0:
                print(f"{name}: original checks {i}/75", flush=True)
        write_json(directory / "baseline_verification.json", baseline)
        for i, record in enumerate(plan[selected_key], 1):
            calls = []
            def oracle(text):
                fresh = text not in cache
                value = predict(text)
                calls.append(fresh)
                return value
            rows, log, count = family.evaluate_record(record, oracle,
                lambda text, metadata: family.audit_window(tokenizer, text, metadata, record["original"], limit))
            it = iter(calls)
            for row in log:
                local = row["physical_cache_hit"]
                native = False if local else next(it)
                row.update(local_cache_hit=local, physical_cache_hit=not native, native_inference_performed=native)
            if count != len(calls) or any(p._version != v for p, v in versions):
                raise RuntimeError("Native call ledger or parameter-integrity check failed.")
            outcomes.extend(dict(r, model=name) for r in rows)
            traces.extend(log)
            write_json(directory / "outcomes.json", outcomes)
            write_json(directory / "query_trace.json", traces)
            print(f"{name}: matched families {i}/{len(plan[selected_key])}", flush=True)
        stats[name] = {"device": str(device), "physical_model_inputs": len(cache),
                       "logical_attack_queries": len(traces), "parameter_updates": 0,
                       "new_model_inference": True, "records": len(plan[selected_key])}
        all_outcomes.extend(outcomes)
        del model, tokenizer, versions, cache, predict, oracle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    budgets, complementary, costs, cross = family.analyse(plan, all_outcomes)
    for name, rows in (("all_outcomes.json", all_outcomes), ("budget_summary.json", budgets),
                       ("complementarity.json", complementary), ("paired_costs.json", costs), ("cross_model.json", cross)):
        write_json(result / name, rows)
    write_json(result / "manifest.json", {"complete": True, "fresh_model_inference": True,
               "protocol": "unchanged six matched-family strategies, fresh inference on BOTH detectors",
               "plan_sha256": sha(root / "plans/family_plan.json"), "stats": stats,
               "independent_holdout": False, "manual_semantic_review": "pending"})


def gradient_table(directory):
    rows, random_rows = csv_rows(Path(directory) / "budget_summary.csv"), csv_rows(Path(directory) / "random_seed_summary.csv")
    lookup = {(r["strategy"], int(r["budget"])): r for r in rows}
    return [{"Budget": int(r["budget"]), "Sequential": int(lookup["left_to_right", int(r["budget"])]["new_error_samples"]),
             "Gradient": int(lookup["gradient", int(r["budget"])]["new_error_samples"]),
             "Random_mean": float(r["mean_new_error_samples"]), "Denominator": int(r["denominator_per_seed"])} for r in random_rows]


def combination_table(directory):
    rows, random_rows = csv_rows(Path(directory) / "budget_summary.csv"), csv_rows(Path(directory) / "random_seed_summary.csv")
    lookup = {(r["strategy"], int(r["total_query_budget"])): r for r in rows}
    return [{"Budget": int(r["total_query_budget"]), "Single": int(lookup["single_only", int(r["total_query_budget"])]["audited_successes"]),
             "Feedback": int(lookup["feedback_combo", int(r["total_query_budget"])]["audited_successes"]),
             "Sequential_combo": int(lookup["sequential_combo", int(r["total_query_budget"])]["audited_successes"]),
             "Random_mean": float(r["mean_successes"]), "Denominator": int(r["denominator_per_seed"])} for r in random_rows]


def family_table(directory):
    return [{"Model": r["model"], "Strategy": r["strategy"], "Successes": r["successes"],
             "Denominator": r["denominator"], "Logical_queries": r["all_sample_queries_capped"]}
            for r in read_json(Path(directory) / "budget_summary.json") if r["total_query_budget"] == 30]


def conclusions(session, tasks=None):
    tasks = CORE_TASKS if tasks is None else tuple(tasks)
    if (len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS) or not set(CORE_TASKS) <= set(tasks)
            or "transfer" in tasks and "stability" not in tasks):
        raise ValueError("Invalid experiment selection for the conclusion.")
    paths = {task: completed_result(session, task) for task in tasks}
    combo, families = combination_table(paths["combination"])[-1], family_table(paths["families"])
    lines = ["# Conclusions from THIS session's selected native experiments", "",
             "Completed main workflow: real model predictions, full combination controls and the matched family comparison on both detectors.",
             f"Combination, {combo['Budget']} total queries: single {combo['Single']}/{combo['Denominator']}; feedback {combo['Feedback']}/{combo['Denominator']}; sequential combination {combo['Sequential_combo']}/{combo['Denominator']}; random mean {combo['Random_mean']}/{combo['Denominator']}.", ""]
    for name in ("piguard", "protectai"):
        rows = [r for r in families if r["Model"] == name and r["Strategy"].endswith("_feedback")]
        best_coverage = max(r["Successes"] for r in rows)
        winners = [r for r in rows if r["Successes"] == best_coverage]
        best_cost = min(r["Logical_queries"] for r in winners)
        best = [r["Strategy"] for r in winners if r["Logical_queries"] == best_cost]
        lines.append(f"{name}: best coverage among the three feedback families = {best_coverage}/{rows[0]['Denominator']}; lowest query cost among coverage-tied families = {best_cost}, strategies {best}.")
    if "gradient" in paths:
        grad = gradient_table(paths["gradient"])[-1]
        lines.extend(["", f"Optional white-box, {grad['Budget']} candidates: gradient {grad['Gradient']}/{grad['Denominator']}; sequential {grad['Sequential']}/{grad['Denominator']}; random mean {grad['Random_mean']}/{grad['Denominator']}.",
                      "Ranking has extra forward/backward cost. Better candidate coverage does not by itself establish lower total runtime."])
    if "selector" in paths:
        lines.extend(["", "Optional paid-probe selector outcomes (same-budget comparison):"])
        for row in read_json(paths["selector"] / "criteria.json"):
            lines.append(f"{row['model']}: fixed zero-width {row['zero_successes']}/{row['denominator']} at {row['zero_queries']} queries; selector {row['selector_successes']}/{row['denominator']} at {row['selector_queries']} queries.")
    if "stability" in paths:
        lines.extend(["", "Optional full-pool benign false blocks and repair controls:"])
        for row in csv_rows(paths["stability"] / "family_budget_summary.csv"):
            if row["dataset"] == "benign" and row["budget"] == "all":
                lines.append(f"{row['family']}: {row['new_error_samples']}/{row['baseline_correct']} originally correct benign prompts have a new error; after NFKC, {row['nfkc_postrepair_any_error_samples']} have any tested error; after targeted repair, {row['mixed_script_postrepair_any_error_samples']}.")
    if "transfer" in paths:
        lines.extend(["", "Optional direct frozen-string transfer (text subset):"])
        for row in csv_rows(paths["transfer"] / "transfer_summary.csv"):
            if row["dataset"] == "text" and row["scope"] == "all":
                lines.append(f"{row['family']}: target successes {row['transfer_success']}/{row['transfer_eligible']} eligible source-successful variants.")
    skipped = [task for task in OPTIONAL_TASKS if task not in paths]
    if skipped:
        lines.extend(["", "Optional experiments not executed in the selected workflow: " + ", ".join(skipped) + ". No results or findings are claimed for these sections."])
    lines.extend(["", "Every numerical finding above is read from the selected completed native stages, not copied historical numbers.",
                  "These are already-explored benchmark cohorts, not a held-out generalization study. Preserve negative outcomes.",
                  "Detector flips do not prove downstream execution or semantic preservation. Repairs are limited controls, not a universal defense.",
                  "Independent target search is not the same experiment as direct transfer of a frozen source-successful string.",
                  "No model training, threshold fitting, or weight updates were performed."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("baseline", "families"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_bundle(args.root)
    (baseline_worker if args.worker == "baseline" else family_worker)(args.root, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
