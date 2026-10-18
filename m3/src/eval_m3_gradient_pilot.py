"""Exploratory white-box PIGuard word-ranking feasibility pilot.

No training or weight updates. Same single-character Cyrillic candidate pool for
left-to-right, gradient word ranking and seeded random word ranking. Gradient is
a subword sensitivity heuristic, not a character derivative or exact HotFlip.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import eval_m3_transfer as transfer
import eval_m3_unicode_stability as rules

PROTOCOL = "m3-gradient-word-ranking-pilot-v1"
MODEL_FILE_SHA = "1ab8335e94916960ec1f03079cacee37aa37b4012a55602c7fabc33b19ad6f0a"
PER_DATASET = 6
SEED = 5230
STRATEGIES = ("left_to_right", "gradient", "random")
BUDGETS = (1, 3, 5, 10)
CONFIG = dict(per_dataset=PER_DATASET, budgets=list(BUDGETS), seed=SEED,
    max_candidate_queries=10, mutation="one Cyrillic confusable substitution",
    ranking="mean L2 norm of native-margin gradient over overlapping visible subword embeddings",
    within_word_order="left to right (not gradient character ranking)",
    eligibility="historically correct, ASCII original, >=3 eligible words",
    selection="category round robin, category and sample IDs ascending; no attack success consulted",
    threat_model="white-box for gradient strategy; no model/threshold/weight updates",
    independent_holdout=False)


def prepare(source):
    if transfer.digest(Path(rules.__file__).read_bytes()) != transfer.SOURCE_SCRIPT_SHA:
        raise ValueError("The original M3 Unicode module changed")
    # Checks complete source, checkpoint, labels, counts, candidate ranks and hashes.
    frozen = transfer.freeze(source)
    selected = []
    for ds in transfer.DATASETS:
        categories = defaultdict(list)
        for r in frozen["originals"]:
            if (r["dataset"] == ds and r["source_original_prediction"] == r["label"]
                and r["original"].isascii() and len(rules.eligible_words(r["original"])) >= 3):
                categories[r["category"]].append(r)
        for cat in categories:
            categories[cat].sort(key=lambda r:r["sample_id"])
        chosen = []
        depth = 0
        while len(chosen) < PER_DATASET:
            added = False
            for cat in sorted(categories):
                if depth < len(categories[cat]):
                    chosen.append(categories[cat][depth]); added = True
                    if len(chosen) == PER_DATASET:
                        break
            if not added:
                raise ValueError(f"Insufficient predeclared eligible {ds} pilot samples")
            depth += 1
        selected.extend(chosen)
    return dict(protocol=PROTOCOL, config=CONFIG, source_file_sha256=frozen["source_file_sha256"],
        checkpoint_sha256=transfer.CHECKPOINT_SHA, generator_sha256=transfer.SOURCE_SCRIPT_SHA,
        model_file_sha256=MODEL_FILE_SHA, selected=selected,
        selection_before_gradient_or_single_character_feedback=True,
        manual_validity_review="pending")


def validate_plan(plan):
    if plan.get("protocol") != PROTOCOL or plan.get("config") != CONFIG or plan.get("selection_before_gradient_or_single_character_feedback") is not True:
        raise ValueError("Pilot configuration/selection changed")
    if (plan.get("checkpoint_sha256") != transfer.CHECKPOINT_SHA or
        plan.get("generator_sha256") != transfer.SOURCE_SCRIPT_SHA or
        plan.get("model_file_sha256") != MODEL_FILE_SHA):
        raise ValueError("Pilot baseline changed")
    if Counter(r["dataset"] for r in plan["selected"]) != Counter({ds:6 for ds in transfer.DATASETS}):
        raise ValueError("Incomplete 18-record pilot")
    if len({r["sample_id"] for r in plan["selected"]}) != 18:
        raise ValueError("Duplicate pilot IDs")
    for r in plan["selected"]:
        if r["label"] != int(r["dataset"] != "benign") or r["source_original_prediction"] != r["label"] or not r["original"].isascii() or len(rules.eligible_words(r["original"])) < 3:
            raise ValueError("Sample does not meet fixed pilot eligibility")


def word_scores(text, offsets, token_norms):
    """Mean token-gradient norm per word; partial/unseen words never guessed."""
    if len(offsets) != len(token_norms) or any(not math.isfinite(v) or v < 0 for v in token_norms):
        raise ValueError("Invalid gradient alignment/norms")
    rows = []
    for start,end,token in rules.eligible_words(text):
        indices = [i for i,(a,b) in enumerate(offsets) if a < end and b > start and b > a]
        covered = {j for i in indices for j in range(max(start,offsets[i][0]),min(end,offsets[i][1]))}
        full = set(range(start,end)) <= covered
        rows.append(dict(start=start,end=end,token=token,visible=full,
            overlapping_tokens=len(indices),score=sum(token_norms[i] for i in indices)/len(indices) if full and indices else None))
    return rows


def single_char_candidates(text, words):
    candidates = []
    for w in words:
        if not w["visible"]:
            continue
        for relative,char in enumerate(w["token"]):
            if char not in rules.CYRILLIC:
                continue
            pos = w["start"] + relative
            changed = text[:pos] + rules.CYRILLIC[char] + text[pos+1:]
            assert sum(a!=b for a,b in zip(changed,text)) == 1 and len(changed)==len(text)
            assert changed[:pos] + char + changed[pos+1:] == text
            candidates.append(dict(word_start=w["start"],word=w["token"],position=pos,
                original_character=char,replacement_character=rules.CYRILLIC[char],text=changed))
    return candidates


def orders(words, sid):
    visible = [w for w in words if w["visible"]]
    left = sorted(visible,key=lambda w:w["start"])
    gradient = sorted(visible,key=lambda w:(-w["score"],w["start"]))
    shuffled = list(left)
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{sid}".encode()).digest()[:8],"big")
    random.Random(seed).shuffle(shuffled)
    return {"left_to_right":[w["start"] for w in left],
            "gradient":[w["start"] for w in gradient],
            "random":[w["start"] for w in shuffled]}


def gradient_probe(model, text):
    """Read a gradient at native word embeddings; restore flags and remove hook.

    Detached input leaf preserves numerical values and native classify path.
    Parameter gradients disabled; no optimizer or parameter in-place operation.
    autograd.grad takes ONE backward pass on the original native logit margin.
    """
    import torch
    flags = [(p,p.requires_grad) for p in model.parameters()]
    versions = [(p,p._version) for p in model.parameters()]
    captured = []
    handle = None
    started = time.perf_counter()
    if getattr(model,"device",None) is not None and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    try:
        for p,_ in flags:
            p.requires_grad_(False)
        def hook(module, args, output):
            leaf = output.detach().requires_grad_(True)
            captured.append(leaf)
            return leaf
        handle = model.deberta.get_input_embeddings().register_forward_hook(hook)
        with torch.enable_grad():
            logits = model.classify(text)
            if len(captured) != 1 or logits.shape != (1,2):
                raise RuntimeError("Native embedding hook/logit dimensions are unsupported")
            original_prediction = int(logits.detach().argmax(-1).item())
            margin = logits[0,original_prediction] - logits[0,1-original_prediction]
            gradient, = torch.autograd.grad(margin,captured[0])
        if not bool(torch.isfinite(gradient).all()) or float(gradient.norm()) == 0:
            raise RuntimeError("Missing, zero or nonfinite gradient: feasibility not verified")
        norms = gradient[0].norm(p=2,dim=-1).detach().cpu().tolist()
        if getattr(model,"device",None) is not None and str(model.device).startswith("cuda"):
            torch.cuda.synchronize()
        return dict(logits=logits.detach().cpu().tolist()[0],token_norms=norms,
            original_prediction=original_prediction,finite_nonzero_gradient=True,
            gradient_norm=float(gradient.norm().detach().cpu()),
            forward_passes=1,backward_passes=1,elapsed_seconds=time.perf_counter()-started)
    finally:
        if handle is not None:
            handle.remove()
        for p,value in flags:
            p.requires_grad_(value)
        if any(p._version != v for p,v in versions):
            raise RuntimeError("Model parameter changed during gradient read")


def evaluate_sample(record, words, query, audit=None):
    """Same pool / 10 candidate queries per method; shared physical cache."""
    pool = single_char_candidates(record["original"],words)
    schedules = orders(words,record["sample_id"])
    grouped = defaultdict(list)
    for c in pool:
        grouped[c["word_start"]].append(c)
    cache,logs,outcomes = {},[],[]
    for strategy in STRATEGIES:
        ordered = [c for start in schedules[strategy] for c in grouped[start]][:10]
        row = dict(sample_id=record["sample_id"],dataset=record["dataset"],category=record["category"],
            label=record["label"],strategy=strategy,candidate_pool_size=len(pool),
            visible_words=sum(w["visible"] for w in words),excluded_partial_words=sum(not w["visible"] for w in words),
            attempts=0,success=False,first_success_attempt=0,selected_word="",selected_position=-1,
            selected_character="",selected_text="",selected_truncated=False,
            selected_visibility="not_applicable",edited_codepoints=0)
        for index,c in enumerate(ordered,1):
            cached = c["text"] in cache
            if not cached:
                pred,prob = query(c["text"])
                if pred not in (0,1) or not math.isfinite(prob) or not 0 <= prob <= 1:
                    raise ValueError("Invalid candidate prediction")
                cache[c["text"]] = (pred,prob)
            pred,prob = cache[c["text"]]
            new_error = pred != record["source_original_prediction"]
            a = audit(c["text"],(c["position"],c["position"]+1)) if audit else dict(token_count=None,truncated=0,edited_span_visibility="unknown")
            logs.append(dict(sample_id=record["sample_id"],dataset=record["dataset"],strategy=strategy,
                attempt_index=index,**c,prediction=pred,probability=prob,new_error=new_error,
                physical_cache_hit=cached,edited_codepoints=1,**a))
            row["attempts"]=index
            if new_error:
                row.update(success=True,first_success_attempt=index,selected_word=c["word"],
                    selected_position=c["position"],selected_character=c["original_character"],
                    selected_text=c["text"],selected_truncated=bool(a["truncated"]),
                    selected_visibility=a["edited_span_visibility"],edited_codepoints=1)
                break
        outcomes.append(row)
    return outcomes,logs,len(cache),schedules


def summary_rows(outcomes):
    summary=[]
    for ds in transfer.DATASETS:
        for method in STRATEGIES:
            sub=[r for r in outcomes if r["dataset"]==ds and r["strategy"]==method]
            for b in BUDGETS:
                wins=[r for r in sub if r["success"] and r["first_success_attempt"]<=b]
                summary.append(dict(dataset=ds,strategy=method,budget=b,denominator=len(sub),
                    new_error_samples=len(wins),new_error_rate=len(wins)/len(sub) if sub else None,
                    logical_candidate_queries=sum(min(b,r["attempts"]) for r in sub),
                    original_forward_passes=len(sub),
                    ranking_forward_passes=len(sub) if method=="gradient" else 0,
                    ranking_backward_passes=len(sub) if method=="gradient" else 0,
                    success_edit_codepoints=1 if wins else None))
    return summary


def write_report(out, summary, feasibility, stats):
    lines=["# M3 梯度选词技术验证（小规模白盒预实验）","",
        "每组6条，按类别轮流挑选原文历史正确、ASCII、至少3个可改词的样本。",
        "选样本不读取过去攻击是否成功；本数据已被探索过，不是独立留出。",
        "三方法相同候选池：每次只换一个同形字；词内字母从左到右。",
        "梯度按原文分词单元的敏感度给词排序，不是直接知道哪个字母替换有效。", "",
        "## 技术可行性", "",
        f"原生分类一致、分词对齐、梯度非零有限、权重版本未改：{len(feasibility)}条通过。",
        "如果技术检查失败，程序停止且不产生complete=true的最终比较。", "",
        "## 同一字符修改规则下的比较", "",
        "|数据|选词方式|最多候选查询|新增错误/分母|比例|候选查询合计|额外排序前向/反向|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        rate=f"{100*r['new_error_rate']:.1f}%" if r["new_error_rate"] is not None else "N/A"
        lines.append(f"|{r['dataset']}|{r['strategy']}|{r['budget']}|{r['new_error_samples']}/{r['denominator']}|{rate}|{r['logical_candidate_queries']}|{r['ranking_forward_passes']}/{r['ranking_backward_passes']}|")
    lines += ["", "## 如何判断值不值得继续", "",
        "先看梯度能否读到、对齐是否正确，再看同一组样本、同样修改/查询预算时是否优于顺序和随机。",
        "若梯度没有改善，报告负结果，不临时挑成功样本或偷换预算；不能预设梯度一定更好。", "",
        "## 必须保留的限制与成本", "",
        "- 这是梯度范数引导的词级启发式，借鉴白盒敏感度思路，不是HotFlip直接复现或新算法声明。",
        "- 同词内各字母仍从左到右，未实现精确字母梯度；共享分词单元不提供独立字符导数。",
        "- 梯度方法需要内部权重及一次排序前向+反向；不能只比较10次候选查询而声称总计算相同。原文查询另计。",
        "- 候选预测跨方法缓存减少实际计算，日志保留逻辑查询；实际共享总耗时不能当各方法独立运行耗时。",
        "- 顺序对照也是单字符候选池，不能直接与第8节整词三族成功率比较。",
        "- 随机只用一个预先固定种子5230；18条小样本仅作技术和方向检查，不证明稳定优势/泛化。",
        "- 原文历史正确必须在当前checkpoint上复现，否则停止；不可通过动态删失败样本增加成功率。",
        "- 不训练、不更新权重、不改阈值；正常文本的新增错误是误拦而非注入执行。",
        "- 可逆字符修改不等于已证明下游语义/功能保留；人工审核pending、训练重叠unknown。",
        "- 此预实验只在原PIGuard上进行，新模型、对方队伍与全量验证均未包含。", "",
        f"共享评估实际候选输入数：{stats['candidate_physical_inputs']}；排序反向次数：{len(feasibility)}。"]
    (out/"REPORT_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def load_piguard(repo,checkpoint):
    import sys
    import torch
    import transformers
    if torch.__version__.split("+")[0]!="2.4.0" or transformers.__version__!="4.44.0":
        raise ValueError("Use the pinned original M3 runtime")
    if rules.digest(checkpoint)!=transfer.CHECKPOINT_SHA or rules.digest(repo/"PIGuard.py")!=MODEL_FILE_SHA:
        raise ValueError("Checkpoint/PIGuard.py differs from the verified fixed baseline")
    sys.path.insert(0,str(repo))
    from PIGuard import PIGuard
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PIGuard("microsoft/deberta-v3-base",2,device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    model.to(device).eval()
    if not getattr(model.tokenizer,"is_fast",False):
        raise RuntimeError("Native tokenizer has no offsets; do not guess gradient-to-word alignment")
    return model,device


def verify_gradient(model,record):
    import torch
    text=record["original"]
    with torch.inference_mode():
        native=model.classify(text).cpu()[0]
    native_pred=int(native.argmax().item())
    if native_pred!=record["source_original_prediction"]:
        raise RuntimeError("Pilot original no longer agrees with the historical source result")
    probe=gradient_probe(model,text)
    grad_logits=torch.tensor(probe["logits"])
    if not torch.allclose(native,grad_logits,rtol=1e-5,atol=1e-5) or probe["original_prediction"]!=native_pred:
        raise RuntimeError("Reading gradients changed native logits")
    # Both calls use the actual model tokenizer and exact native 256-token inputs.
    native_tokens=model.tokenizer(text,return_tensors="pt",truncation=True,padding="max_length",max_length=256)
    audit_tokens=model.tokenizer(text,truncation=True,padding="max_length",max_length=256,return_offsets_mapping=True)
    if native_tokens["input_ids"][0].tolist()!=audit_tokens["input_ids"] or native_tokens["attention_mask"][0].tolist()!=audit_tokens["attention_mask"]:
        raise RuntimeError("Offset audit does not match native inputs")
    words=word_scores(text,audit_tokens["offset_mapping"],probe["token_norms"])
    if sum(w["visible"] for w in words)<3:
        raise RuntimeError("Fewer than three fully visible words: pilot feasibility incomplete")
    return words,dict(sample_id=record["sample_id"],dataset=record["dataset"],
        native_prediction=native_pred,native_injection_probability=float(native.softmax(-1)[1]),
        native_logits_match=True,native_tokens_match=True,gradient_finite_nonzero=True,
        gradient_norm=probe["gradient_norm"],gradient_forward_passes=1,gradient_backward_passes=1,
        ranking_elapsed_seconds=probe["elapsed_seconds"],visible_words=sum(w["visible"] for w in words),
        partial_or_unseen_words=sum(not w["visible"] for w in words))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source",type=Path)
    group.add_argument("--plan",type=Path)
    p.add_argument("--repo",type=Path,default=Path("/content/PIGuard_M3"))
    p.add_argument("--checkpoint",type=Path,default=Path("/content/PIGuard_M3/checkpoints/best_model.pth"))
    p.add_argument("--output-dir",type=Path,default=Path("/content/m3_gradient_results"))
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.source:
        if args.smoke: p.error("--smoke requires a prepared --plan")
        plan=prepare(args.source);validate_plan(plan)
        out=args.output_dir/("prepared_"+transfer.stamp());out.mkdir(parents=True,exist_ok=False)
        transfer.json_write(out/"pilot_plan.json",plan)
        transfer.json_write(out/"preparation.json",dict(plan_sha256=transfer.digest((out/"pilot_plan.json").read_bytes())))
        print("Pilot selection frozen BEFORE gradients/single-character tests:",out,flush=True)
        for r in plan["selected"]: print(r["dataset"],r["sample_id"],r["category"],flush=True)
        print("18 selected records only; this is a feasibility pilot, not a reduced rerun of sections 1–8.",flush=True)
        return
    raw=args.plan.read_bytes();plan=json.loads(raw);validate_plan(plan)
    if transfer.digest(raw)!=json.loads((args.plan.parent/"preparation.json").read_text())["plan_sha256"]:
        raise ValueError("Pilot selection changed after preparation")
    started=time.monotonic()
    model,device=load_piguard(args.repo,args.checkpoint)
    if args.smoke:
        words,probe=verify_gradient(model,plan["selected"][0])
        print(json.dumps(probe,ensure_ascii=False,indent=2),flush=True)
        print("Most sensitive eligible words (heuristic, not attacks):",flush=True)
        for w in sorted([w for w in words if w["visible"]],key=lambda w:(-w["score"],w["start"]))[:5]: print(w,flush=True)
        print("Gradient smoke passed; no attack superiority established.",flush=True)
        return
    out=args.output_dir/("gradient_"+transfer.stamp());out.mkdir(parents=True,exist_ok=False)
    transfer.json_write(out/"pilot_plan.json",plan)
    transfer.json_write(out/"manifest.json",dict(complete=False,protocol=PROTOCOL,technical_verified=False))
    import torch
    all_outcomes,all_logs,all_rankings,feasibility=[],[],[],[]
    physical=0
    def predict(text):
        with torch.inference_mode():
            logits=model.classify(text)
            return int(logits.argmax().item()),float(logits.softmax(-1)[0,1].cpu())
    for i,r in enumerate(plan["selected"],1):
        words,probe=verify_gradient(model,r)
        feasibility.append(probe)
        outcomes,logs,calls,schedules=evaluate_sample(r,words,predict,
            audit=lambda text,span:rules.audit_input(model.tokenizer,text,span))
        all_outcomes.extend(outcomes);all_logs.extend(logs);physical+=calls
        for w in words:
            all_rankings.append(dict(sample_id=r["sample_id"],dataset=r["dataset"],**w,
                **{method+"_rank":schedules[method].index(w["start"])+1 if w["visible"] else None for method in STRATEGIES}))
        # Keep completed samples if a later technical check fails. No auto-resume.
        transfer.csv_write(out/"feasibility.csv",feasibility)
        transfer.csv_write(out/"outcomes.csv",all_outcomes)
        transfer.csv_write(out/"query_trace.csv",all_logs)
        transfer.csv_write(out/"word_rankings.csv",all_rankings)
        print(f"Pilot {i}/18 verified; candidate physical inputs={physical}",flush=True)
    summary=summary_rows(all_outcomes)
    transfer.csv_write(out/"budget_summary.csv",summary)
    stats=dict(candidate_physical_inputs=physical,original_native_forward_passes=18,
        gradient_ranking_forward_passes=18,gradient_ranking_backward_passes=18,
        candidate_logical_queries=sum(r["attempts"] for r in all_outcomes),
        parameter_updates=0,shared_caching_across_methods=True)
    write_report(out,summary,feasibility,stats)
    import transformers
    transfer.json_write(out/"manifest.json",dict(complete=True,technical_verified=True,protocol=PROTOCOL,
        config=CONFIG,pilot_records=18,checkpoint_sha256=transfer.CHECKPOINT_SHA,
        model_file_sha256=MODEL_FILE_SHA,source_file_sha256=plan["source_file_sha256"],
        plan_sha256=transfer.digest(raw),script_sha256=transfer.digest(Path(__file__).read_bytes()),
        generator_sha256=rules.digest(Path(rules.__file__)),device=str(device),
        torch=torch.__version__,transformers=transformers.__version__,
        elapsed_seconds=time.monotonic()-started,independent_holdout=False,
        manual_validity_review="pending",training_overlap="unknown",**stats))
    print((out/"REPORT_ZH.md").read_text(),flush=True)
    print("Completed gradient pilot:",out,flush=True)


if __name__=="__main__":
    main()
