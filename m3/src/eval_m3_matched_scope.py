"""New exploratory scope-matched experiment on the same ProtectAI detector.

Frozen previous target cohort, unchanged original model/budget/edit cap, complete
legal position support for every primary strategy, prefix8 diagnostic retained.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time
import zipfile

import eval_m3_combo_target as previous
import unicode_combo_matched_attack as engine

base,reader,rules=previous.base,previous.reader,previous.rules
PROTOCOL='m3-protectai-matched-full-position-scope-v1'
PREVIOUS_SHA='51f428fd35d5f4652317cefd420dca367cf5942053254022f47a0cfab30c4df8'
CONFIG=dict(attack=copy.deepcopy(engine.CONFIG),target_model=reader.MODEL_ID,
    target_revision=reader.MODEL_REVISION,target_max_length=512,
    cohort='ALL previously frozen target-correct eligible texts; verify all75 original labels again',
    primary_comparison='full legal position universe shared; query paths differ by strategy',
    diagnostic='prefix8 feedback is narrow-scope control, not a primary matched strategy',
    no_manual_case_position_override=True,independent_holdout=False,
    exploratory_after_prior_negative_result=True)


def check_modules():
    previous.check_modules()
    if rules.digest(previous.__file__)!=PREVIOUS_SHA:raise ValueError('Previous target evaluator changed')


def load_previous(source):
    names=('manifest.json','target_input_plan.json','target_baseline.json','target_cohort.json','outcomes.json')
    source=Path(source)
    if source.is_dir():payloads={n:(source/n).read_bytes() for n in names}
    else:
        with zipfile.ZipFile(source) as z:
            if any(z.namelist().count(n)!=1 or z.getinfo(n).file_size>40_000_000 for n in names):raise ValueError('Wrong/incomplete previous target ZIP')
            payloads={n:z.read(n) for n in names}
    data={n:json.loads(raw) for n,raw in payloads.items()}
    manifest=data['manifest.json'];input_plan=data['target_input_plan.json'];cohort=data['target_cohort.json']
    if manifest.get('complete') is not True or manifest.get('protocol')!=previous.PROTOCOL:
        raise ValueError('Use completed combo_target result, not source PIGuard ZIP')
    previous.validate_plan(input_plan)
    if manifest.get('config')!=previous.CONFIG:raise ValueError('Previous target settings mismatch')
    if manifest['plan_sha256']!=reader.digest(payloads['target_input_plan.json']):raise ValueError('Previous plan hash mismatch')
    if manifest['target_cohort_sha256']!=reader.digest(payloads['target_cohort.json']):raise ValueError('Previous cohort hash mismatch')
    if cohort.get('frozen_before_first_attack') is not True or cohort['baseline_sha256']!=reader.digest(payloads['target_baseline.json']):
        raise ValueError('Previous target cohort was not correctly frozen')
    selected,eligibility=previous.target_cohort(input_plan['originals'],data['target_baseline.json'])
    if cohort['records']!=selected or cohort['selected_ids']!=[r['sample_id'] for r in selected]:raise ValueError('Previous selection mismatch')
    if manifest.get('target_eligible_records')!=len(selected):raise ValueError('Previous cohort count mismatch')
    base.summary(data['outcomes.json'],cohort['selected_ids'])
    if manifest.get('target_model')!=reader.MODEL_ID or manifest.get('target_revision')!=reader.MODEL_REVISION or manifest.get('target_max_length')!=512:
        raise ValueError('Wrong previous target model')
    if manifest.get('target_evaluator_sha256')!=PREVIOUS_SHA or manifest.get('engine_sha256')!=previous.ENGINE_SHA:
        raise ValueError('Previous target/engine code mismatch')
    return dict(originals=input_plan['originals'],selected=selected,baseline=data['target_baseline.json'],
        previous_outcomes=data['outcomes.json'],source_manifest=manifest,
        previous_file_sha256={n:reader.digest(raw) for n,raw in payloads.items()})


def prepare(source):
    check_modules();data=load_previous(source)
    plan=dict(protocol=PROTOCOL,config=copy.deepcopy(CONFIG),**data,
        engine_sha256=rules.digest(engine.__file__),evaluator_sha256=rules.digest(__file__),
        previous_evaluator_sha256=PREVIOUS_SHA,selection_and_policy_frozen_before_new_trials=True,
        manual_validity_review='pending')
    validate_plan(plan);return plan


def validate_plan(plan):
    check_modules()
    if plan.get('protocol')!=PROTOCOL or plan.get('config')!=CONFIG or plan.get('selection_and_policy_frozen_before_new_trials') is not True:
        raise ValueError('Matched-scope protocol changed')
    if plan.get('engine_sha256')!=rules.digest(engine.__file__) or plan.get('evaluator_sha256')!=rules.digest(__file__):
        raise ValueError('New source changed after preparation')
    if len(plan['originals'])!=75 or len({r['sample_id'] for r in plan['originals']})!=75:raise ValueError('Missing source originals')
    selected,_=previous.target_cohort(plan['originals'],plan['baseline'])
    if selected!=plan['selected'] or not selected:raise ValueError('Changed/empty frozen target cohort')
    base.summary(plan['previous_outcomes'],[r['sample_id'] for r in selected])


def evaluate_record(record,predict,audit):
    outcomes,trace=[],[];cache={}
    for method in engine.STRATEGIES:
        hits=[]
        def oracle(text):
            hit=text in cache
            if not hit:cache[text]=predict(text)
            hits.append(hit);return cache[text]
        result,logs=engine.attack(record['original'],oracle,method,record['sample_id'])
        if logs[0]['prediction']!=record['source_original_prediction']:raise RuntimeError('Original baseline changed')
        if method!='feedback_prefix8' and result['allowed_positions']!=engine.old.positions(record['original']):
            raise AssertionError('Primary strategy did not receive the full shared position universe')
        for row,hit in zip(logs,hits):
            p=row['positions'];span=(min(p),max(p)+1) if p else None
            row.update(sample_id=record['sample_id'],strategy=method,physical_cache_hit=hit,**audit(row['text'],span))
        first=logs[result['first_success_query']-1] if result['success'] else None
        result.update(sample_id=record['sample_id'],category=record['category'],
            first_success_audited=bool(first and not logs[0]['truncated'] and not first['truncated'] and first['edited_span_visibility']=='full'),
            first_success_visibility=first['edited_span_visibility'] if first else 'not_applicable',
            first_success_truncated=bool(first['truncated']) if first else False)
        outcomes.append(result);trace.extend(logs)
    return outcomes,trace,len(cache)


def summarise(outcomes,ids):
    ids=set(ids);by={(r['sample_id'],r['strategy']):r for r in outcomes}
    if not ids or len(by)!=len(outcomes) or set(by)!={(sid,method) for sid in ids for method in engine.STRATEGIES}:
        raise ValueError('Missing/duplicate matched-scope results')
    rows,random_rows,pairs=[],[],[]
    for budget in base.BUDGETS:
        wins={}
        for method in engine.STRATEGIES:
            sub=[by[sid,method] for sid in sorted(ids)]
            if any(not 1<=r['total_queries']<=30 for r in sub):raise ValueError('Query budget violated')
            eligible=[r for r in sub if r['first_success_audited'] and r['first_success_query']<=budget]
            wins[method]={r['sample_id'] for r in eligible}
            rows.append(dict(strategy=method,total_query_budget=budget,denominator=len(ids),successes=len(eligible),
                success_rate=len(eligible)/len(ids),logical_queries=sum(min(budget,r['total_queries']) for r in sub),
                primary_matched_scope=method!='feedback_prefix8'))
        controls=[r for r in rows if r['total_query_budget']==budget and r['strategy'].startswith('random_full_')]
        rates=[r['success_rate'] for r in controls]
        random_rows.append(dict(total_query_budget=budget,denominator_per_seed=len(ids),seeds=len(controls),
            mean_successes=statistics.mean(r['successes'] for r in controls),min_successes=min(r['successes'] for r in controls),
            max_successes=max(r['successes'] for r in controls),seed_rate_sd=statistics.stdev(rates),
            mean_queries=statistics.mean(r['logical_queries'] for r in controls)))
        f=wins['feedback_full']
        for method in ('single_full','feedback_prefix8','sequential_full')+tuple(f'random_full_{s}' for s in engine.SEEDS):
            c=wins[method];pairs.append(dict(total_query_budget=budget,comparator=method,denominator=len(ids),
                both=len(f&c),feedback_only=len(f-c),comparator_only=len(c-f),neither=len(ids-(f|c))))
    return rows,random_rows,pairs


def write_report(out,plan,outcomes,rows,random_rows,stats):
    n=len(plan['selected']);idx={(r['strategy'],r['total_query_budget']):r for r in rows}
    lines=['# M3 第二模型：完整位置范围匹配对照','',
        f"仍是固定ProtectAI和原来全部{n}条合格文本；最多3个字符、30总调用，模型/精度/阈值不改。",
        '主对照都能从同一个完整合法位置池中选，不再把组合限制在前8个位置。',
        '组合先最多8次单点探测；后续增加字符时优先使用本方法尚未修改过的位置，仍按8/8/10分层配额。',
        '反馈选概率最低的两条父候选；顺序选最先观察的两条；随机按固定种子选两条。完整可选池相同，不代表预算内实际尝试路径相同或穷尽所有组合。',
        '单字符方法保留原先最多29个轮询单点版本，不削弱它来制造组合优势。另保留旧前8位置反馈控制作诊断，不混入主匹配对照。','',
        '|总查询上限|完整范围单点|完整范围顺序组合|完整范围反馈组合|完整范围随机平均（范围）|旧前8反馈诊断|',
        '|---:|---:|---:|---:|---:|---:|']
    for r in random_rows:
        b=r['total_query_budget']
        lines.append(f"|{b}|{idx['single_full',b]['successes']}/{n}|{idx['sequential_full',b]['successes']}/{n}|{idx['feedback_full',b]['successes']}/{n}|{r['mean_successes']:.2f}/{n}（{r['min_successes']}–{r['max_successes']}）|{idx['feedback_prefix8',b]['successes']}/{n}|")
    by={(r['sample_id'],r['strategy']):r for r in outcomes}
    wins=[r for r in outcomes if r['strategy']=='feedback_full' and r['first_success_audited']]
    new_vs_prefix=[r['sample_id'] for r in wins if not by[r['sample_id'],'feedback_prefix8']['first_success_audited']]
    new_vs_single=[r['sample_id'] for r in wins if not by[r['sample_id'],'single_full']['first_success_audited']]
    lost_vs_single=[r['sample_id'] for r in plan['selected'] if by[r['sample_id'],'single_full']['first_success_audited'] and not by[r['sample_id'],'feedback_full']['first_success_audited']]
    lines+=['','## 要回答的问题','',f"完整范围反馈成功、旧前8反馈失败的ID：{new_vs_prefix or '无'}。",
        f"完整范围反馈成功、完整范围单点失败的ID：{new_vs_single or '无'}。",
        f"单点成功但完整范围反馈仍失败的ID：{lost_vs_single or '无'}。",
        '先看前8限制是否有覆盖代价，再看完整范围下组合和反馈是否真的更有用。没有优势就保留负结果，不继续放宽字数/预算或挑成功样本。','',
        '## 边界和成本','',
        f"原文复现校验75次；攻击端逻辑请求{stats['logical_attack_queries']}（含每方法原文/探测/撤销）；实际目标输入{stats['physical_target_inputs']}。",
        '- 所有主方法可选位置集合完全相等，实际探测次序不同；最多30次不能保证测试过全部合法位置或所有组合。',
        '- 这次完整范围策略同时采用未修改位置优先扩展；不能把任意提升全都归因于一个参数，报告的是这套明确的新覆盖策略。',
        '- 单点仍29候选+原文，组合保留至多3次缩减，失败不回收预留。相同上限不要求实际调用数完全相等。',
        '- 旧8位置诊断重新运行，不把旧存档预测免费提供给新搜索；旧结果仅保留比较证据。',
        '- 全部75历史原文标签需当前模型重现，否则停止不换分母；13样本来自已探索基准，这次是观察负结果后的探索性调整，非新留出泛化验证。',
        '- 只看标签+概率，无模型/梯度供攻击端；审计仅在攻击返回后记录，主成功须无截断/位置可见。',
        '- Unicode可逆不证明语义保持或下游执行；人工pending、训练重叠unknown。',
        '- 目标模型、512输入、argmax、原保护范围/映射、30调用/3字符、随机种子全部保持；不做其他模型或其他队伍。',
        '- 最多3次撤销不是全局最小证明，种子波动不是独立样本置信区间，共享运行时间非各策略独立耗时。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source',type=Path);g.add_argument('--plan',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('/content/m3_matched_scope_results'));args=p.parse_args()
    if args.source:
        plan=prepare(args.source);out=args.output_dir/('prepared_'+reader.stamp());out.mkdir(parents=True,exist_ok=False)
        reader.json_write(out/'matched_plan.json',plan);reader.json_write(out/'preparation.json',dict(plan_sha256=rules.digest(out/'matched_plan.json')))
        print(f"Frozen all {len(plan['selected'])} previous target-eligible originals, full position scope and controls BEFORE new trials.",flush=True)
        print('Methods:',engine.STRATEGIES,flush=True);print('Prepared:',out,flush=True);return
    raw=args.plan.read_bytes();plan=json.loads(raw);validate_plan(plan)
    if reader.digest(raw)!=json.loads((args.plan.parent/'preparation.json').read_text())['plan_sha256']:raise ValueError('Plan changed after preparation')
    import torch,transformers
    if torch.__version__.split('+')[0]!='2.4.0' or transformers.__version__!='4.44.0':raise ValueError('Pinned runtime required')
    started=time.monotonic();out=args.output_dir/('matched_scope_'+reader.stamp());out.mkdir(parents=True,exist_ok=False)
    reader.json_write(out/'matched_plan.json',plan);reader.json_write(out/'manifest.json',dict(protocol=PROTOCOL,complete=False,stage='loading'))
    tokenizer,model,device=reader.load_target();versions=[(p,p._version) for p in model.parameters()]
    cache={};physical=0
    def predict(text):
        nonlocal physical
        if text not in cache:
            encoded=tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=512)
            with torch.inference_mode():
                logits=model(**{k:v.to(device) for k,v in encoded.items()}).logits
                cache[text]=(int(logits.argmax(-1).item()),float(logits.softmax(-1)[0,1].cpu()));physical+=1
        return cache[text]
    verified={}
    for i,r in enumerate(plan['originals'],1):
        pred,prob=predict(r['original']);verified[r['sample_id']]=dict(prediction=pred,probability=prob)
        reader.json_write(out/'baseline_verification.json',verified)
        if pred!=plan['baseline'][r['sample_id']]['prediction']:raise RuntimeError(f"Historical baseline changed for {r['sample_id']}; stop, do not resample")
        if i%10==0 or i==75:print(f'Original label verification {i}/75',flush=True)
    baseline_physical=physical
    outcomes,logs=[],[]
    for i,r in enumerate(plan['selected'],1):
        native_calls=[]
        def oracle(text):
            before=physical;response=predict(text);native_calls.append(physical>before);return response
        result,trace,local_count=evaluate_record(r,oracle,lambda text,span:previous.target_audit(tokenizer,text,span))
        it=iter(native_calls)
        for row in trace:
            local_hit=row['physical_cache_hit'];native=False if local_hit else next(it)
            row.update(local_cache_hit=local_hit,physical_cache_hit=not native,native_inference_performed=native)
        assert len(native_calls)==local_count
        if any(param._version!=version for param,version in versions):raise RuntimeError('Model parameter changed')
        outcomes.extend(result);logs.extend(trace)
        for name,data in (('outcomes.json',outcomes),('query_trace.json',logs)):
            reader.json_write(out/name,data)
        reader.csv_write(out/'outcomes.csv',base.flat_rows(outcomes));reader.csv_write(out/'query_trace.csv',base.flat_rows(logs))
        reader.json_write(out/'manifest.json',dict(protocol=PROTOCOL,complete=False,planned_records=len(plan['selected']),completed_records=i))
        print(f'Matched scope {i}/{len(plan["selected"])} complete; native inputs={physical}',flush=True)
    rows,random_rows,pairs=summarise(outcomes,[r['sample_id'] for r in plan['selected']])
    for name,data in (('budget_summary.csv',rows),('random_seed_summary.csv',random_rows),('paired_controls.csv',pairs)):
        reader.csv_write(out/name,data)
    stats=dict(baseline_verification_requests=75,baseline_physical_inputs=baseline_physical,
        logical_attack_queries=len(logs),physical_target_inputs=physical,attack_physical_inputs=physical-baseline_physical,
        logical_original_requests=sum(r['phase']=='original' for r in logs),logical_refinement_requests=sum(r['phase']=='refinement' for r in logs),
        gradient_backward_passes=0,parameter_updates=0,source_new_model_calls=0)
    write_report(out,plan,outcomes,rows,random_rows,stats)
    reader.json_write(out/'manifest.json',dict(protocol=PROTOCOL,complete=True,config=CONFIG,
        target_model=reader.MODEL_ID,target_revision=reader.MODEL_REVISION,target_max_length=512,
        target_eligible_records=len(plan['selected']),eligible_ids=[r['sample_id'] for r in plan['selected']],
        baseline_labels_reproduced=True,plan_sha256=reader.digest(raw),engine_sha256=rules.digest(engine.__file__),
        evaluator_sha256=rules.digest(__file__),previous_evaluator_sha256=PREVIOUS_SHA,
        previous_file_sha256=plan['previous_file_sha256'],device=str(device),torch=torch.__version__,transformers=transformers.__version__,
        elapsed_seconds=time.monotonic()-started,independent_holdout=False,exploratory_after_prior_result=True,
        manual_validity_review='pending',training_overlap='unknown',**stats))
    print((out/'REPORT_ZH.md').read_text(),flush=True);print('Complete:',out,flush=True)


if __name__=='__main__':main()
