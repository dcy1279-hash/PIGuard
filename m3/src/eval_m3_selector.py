"""Step2: paid two-probe selector vs freshly rerun fixed-zero baselines."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import zipfile

import eval_m3_complementarity as previous
import unicode_selector_attack as engine

reader,rules=previous.reader,previous.rules
PREVIOUS_SHA='5f97e72416547e4a74a6d64abd417a425604e535be07f371311b021ee19561dc'
PROTOCOL='m3-two-model-paid-probe-family-selector-step2-v1'
CONFIG=dict(attack=copy.deepcopy(engine.CONFIG),source_max_length=256,target_max_length=512,
    source_checkpoint_sha256=reader.CHECKPOINT_SHA,target_model=reader.MODEL_ID,target_revision=reader.MODEL_REVISION,
    cohorts='same FULL step1 source22/target13; verify75 originals on EACH model, no outcome selection',
    inference='fresh4-method comparison on BOTH models; old scores/outcomes never passed to attacker',
    nominal_goal='selector success count >= zero count AND fewer total logical queries',
    stricter_goal='retain ALL fixed-zero success IDs AND fewer total logical queries',
    independent_holdout=False,exploratory_after_step1=True)


def check_modules():
    previous.check_modules()
    if rules.digest(previous.__file__)!=PREVIOUS_SHA:raise ValueError('Step1 evaluator changed')


def verify_step1(data,payload):
    manifest=data['manifest.json'];old=data['complement_plan.json'];baseline=data['source_baseline_verification.json']
    previous.validate_plan(old)
    if manifest.get('complete') is not True or manifest.get('protocol')!=previous.PROTOCOL or manifest.get('config')!=previous.CONFIG:raise ValueError('Use complete complement ZIP')
    if manifest['plan_sha256']!=reader.digest(payload['complement_plan.json']):raise ValueError('Old plan hash mismatch')
    for k in ('engine_sha256','evaluator_sha256','source_checkpoint_sha256','source_model_file_sha256','target_source_file_sha256'):
        expected=old[k] if k!='evaluator_sha256' else PREVIOUS_SHA
        if manifest[k]!=expected:raise ValueError('Old code/model/provenance mismatch')
    if manifest['source_selected_ids']!=[r['sample_id'] for r in old['source_selected']] or manifest['target_selected_ids']!=[r['sample_id'] for r in old['target_selected']]:raise ValueError('Old cohort mismatch')
    if set(baseline)!={r['sample_id'] for r in old['originals']} or any(baseline[r['sample_id']]['prediction']!=r['source_original_prediction'] for r in old['originals']):raise ValueError('Old75 source labels not verified')
    actual={r['original']:(baseline[r['sample_id']]['prediction'],baseline[r['sample_id']]['probability']) for r in old['originals']}
    physical=len(actual);outcomes=[];trace=[]
    for record in old['source_selected']:
        lookup={};audits={}
        for row in data['source_query_trace.json']:
            if row['sample_id']!=record['sample_id']:continue
            value=(row['prediction'],row['injection_probability'])
            if row['text'] in lookup and lookup[row['text']]!=value:raise ValueError('Inconsistent old oracle')
            lookup[row['text']]=value
            audits[row['text']]={k:row[k] for k in ('token_count','truncated','edited_span_visibility','same_token_ids_as_original','zero_width_offset_covered','visibility_definition')}
        calls=[]
        def oracle(text):
            nonlocal physical
            new=text not in actual
            if new:actual[text]=lookup[text];physical+=1
            elif actual[text]!=lookup[text]:raise ValueError('Old cache mismatch')
            calls.append(new);return actual[text]
        result,logs,n=previous.evaluate_record(record,oracle,lambda text,meta:audits[text])
        if n!=len(calls):raise ValueError('Old count mismatch')
        it=iter(calls)
        for row in logs:
            local=row['physical_cache_hit'];new=False if local else next(it)
            row.update(local_cache_hit=local,physical_cache_hit=not new,native_inference_performed=new)
        outcomes.extend(result);trace.extend(logs)
    if outcomes!=data['source_outcomes.json'] or trace!=data['source_query_trace.json']:raise ValueError('Old source path replay failed')
    combined=[dict(r,model='piguard') for r in outcomes]+[dict(r,model='protectai') for r in old['target_outcomes']]
    if combined!=data['all_outcomes.json'] or data['reused_target_trace.json']!=old['target_trace']:raise ValueError('Old target reuse changed')
    previous.analyse(old,combined)
    if len(trace)!=manifest['logical_attack_queries'] or physical!=manifest['physical_source_inputs']:raise ValueError('Old source ledger changed')
    if any(manifest[k]!=0 for k in ('target_new_model_calls','gradient_backward_passes','parameter_updates')):raise ValueError('Old model updated')


def prepare(source):
    check_modules();source=Path(source)
    names=('manifest.json','complement_plan.json','source_baseline_verification.json','source_outcomes.json','source_query_trace.json','all_outcomes.json','reused_target_trace.json')
    if source.is_dir():payload={n:(source/n).read_bytes() for n in names}
    else:
        with zipfile.ZipFile(source) as z:
            if any(z.namelist().count(n)!=1 or z.getinfo(n).file_size>40_000_000 for n in names):raise ValueError('Upload complete complement ZIP, not mixed/revisit ZIP')
            if z.testzip() is not None:raise ValueError('Source CRC failure')
            payload={n:z.read(n) for n in names}
    data={n:json.loads(v) for n,v in payload.items()};verify_step1(data,payload);old=data['complement_plan.json']
    plan=dict(protocol=PROTOCOL,config=copy.deepcopy(CONFIG),previous_plan=old,originals=old['originals'],
        source_selected=old['source_selected'],target_selected=old['target_selected'],common_ids=old['common_ids'],
        source_baseline=data['source_baseline_verification.json'],target_baseline=old['target_baseline'],
        source_file_sha256={n:reader.digest(v) for n,v in payload.items()},
        engine_sha256=rules.digest(engine.__file__),evaluator_sha256=rules.digest(__file__),previous_evaluator_sha256=PREVIOUS_SHA,
        policy_and_cohorts_frozen_before_new_trials=True)
    plan['frozen_baselines_sha256']=previous.object_digest(dict(source=plan['source_baseline'],target=plan['target_baseline']))
    validate_plan(plan);return plan


def validate_plan(plan):
    check_modules()
    if plan.get('protocol')!=PROTOCOL or plan.get('config')!=CONFIG or plan.get('policy_and_cohorts_frozen_before_new_trials') is not True:raise ValueError('New policy changed')
    if plan.get('engine_sha256')!=rules.digest(engine.__file__) or plan.get('evaluator_sha256')!=rules.digest(__file__) or plan.get('previous_evaluator_sha256')!=PREVIOUS_SHA:raise ValueError('New code changed after freeze')
    old=plan['previous_plan'];previous.validate_plan(old)
    for key in ('originals','source_selected','target_selected','common_ids'):
        if plan[key]!=old[key]:raise ValueError('Cohort/originals changed')
    expected={r['sample_id'] for r in plan['originals']}
    for key in ('source_baseline','target_baseline'):
        if set(plan[key])!=expected:raise ValueError('Incomplete baseline')
    if any(plan['source_baseline'][r['sample_id']]['prediction']!=r['source_original_prediction'] for r in plan['originals']):raise ValueError('Changed source baseline')
    if plan['target_baseline']!=old['target_baseline']:raise ValueError('Changed target baseline')
    if plan.get('frozen_baselines_sha256')!=previous.object_digest(dict(source=plan['source_baseline'],target=plan['target_baseline'])):raise ValueError('Frozen baseline payload changed')


def evaluate_record(record,predict,audit):
    outcomes=[];trace=[];cache={}
    for method in engine.STRATEGIES:
        hits=[]
        def oracle(text):
            hit=text in cache
            if not hit:cache[text]=predict(text)
            hits.append(hit);return cache[text]
        result,logs=engine.attack(record['original'],oracle,method)
        if len(logs)!=len(hits) or len(logs)!=result['total_queries']:raise AssertionError('Uncounted oracle request')
        if logs[0]['prediction']!=record['source_original_prediction']:raise RuntimeError('Original changed')
        for row,hit in zip(logs,hits):row.update(sample_id=record['sample_id'],strategy=method,physical_cache_hit=hit,**audit(row['text'],row['edit_metadata']))
        first=logs[result['first_success_query']-1] if result['success'] else None
        result.update(sample_id=record['sample_id'],category=record['category'],
            first_success_audited=bool(first and not logs[0]['truncated'] and not first['truncated'] and first['edited_span_visibility']=='full' and not first['same_token_ids_as_original']),
            first_success_visibility=first['edited_span_visibility'] if first else 'not_applicable',
            first_success_truncated=bool(first['truncated']) if first else False,
            first_success_token_equivalent=bool(first['same_token_ids_as_original']) if first else False)
        outcomes.append(result);trace.extend(logs)
    return outcomes,trace,len(cache)


def analyse(plan,outcomes):
    cohorts={'piguard':[r['sample_id'] for r in plan['source_selected']], 'protectai':[r['sample_id'] for r in plan['target_selected']]}
    by={(r['model'],r['sample_id'],r['strategy']):r for r in outcomes}
    expected={(model,sid,method) for model,ids in cohorts.items() for sid in ids for method in engine.STRATEGIES}
    if set(by)!=expected or len(by)!=len(outcomes):raise ValueError('Incomplete/duplicate results')
    if any(not 1<=r['total_queries']<=30 or r['final_edits']>3 or r['selector_probe_queries']>2 for r in outcomes):raise ValueError('Shared budget/edit cap violated')
    rows=[];pairs=[];criteria=[];decisions=[];cross=[]
    for model,ids in cohorts.items():
        for budget in (5,10,20,30):
            wins={s:{sid for sid in ids if by[model,sid,s]['first_success_audited'] and by[model,sid,s]['first_success_query']<=budget} for s in engine.STRATEGIES}
            for method in engine.STRATEGIES:
                good=[by[model,sid,method] for sid in sorted(wins[method])]
                rows.append(dict(model=model,strategy=method,total_query_budget=budget,denominator=len(ids),successes=len(good),
                    all_sample_queries_capped=sum(min(budget,by[model,sid,method]['total_queries']) for sid in ids),
                    first_edit_mean_on_success=sum(r['first_success_edits'] for r in good)/len(good) if good else None))
            a=wins['score_selector']
            for method in ('zero_reference','cyr_reference','paired_zero_control'):
                b=wins[method];both=sorted(a&b)
                pairs.append(dict(model=model,total_query_budget=budget,comparator=method,denominator=len(ids),both=len(both),
                    selector_only_ids=sorted(a-b),comparator_only_ids=sorted(b-a),neither=len(set(ids)-(a|b)),
                    selector_first_queries_on_common=sum(by[model,sid,'score_selector']['first_success_query'] for sid in both),
                    comparator_first_queries_on_common=sum(by[model,sid,method]['first_success_query'] for sid in both)))
        a={sid for sid in ids if by[model,sid,'score_selector']['first_success_audited']};b={sid for sid in ids if by[model,sid,'zero_reference']['first_success_audited']}
        cost_a=sum(by[model,sid,'score_selector']['total_queries'] for sid in ids);cost_b=sum(by[model,sid,'zero_reference']['total_queries'] for sid in ids)
        criteria.append(dict(model=model,denominator=len(ids),selector_successes=len(a),zero_successes=len(b),selector_queries=cost_a,zero_queries=cost_b,
            selector_only_ids=sorted(a-b),zero_only_ids=sorted(b-a),success_count_not_lower=len(a)>=len(b),all_zero_success_ids_preserved=b<=a,
            queries_strictly_less=cost_a<cost_b,nominal_goal_met=len(a)>=len(b) and cost_a<cost_b,strict_paired_goal_met=b<=a and cost_a<cost_b))
        for sid in ids:
            r=by[model,sid,'score_selector'];control=by[model,sid,'paired_zero_control'];zero=by[model,sid,'zero_reference']
            decisions.append(dict(model=model,sample_id=sid,chosen_family=r['chosen_family'],choice_reason=r['choice_reason'],
                choice_after_query=r['choice_after_query'],choice_probabilities=r['choice_probabilities'],paid_probe_queries=r['selector_probe_queries'],
                selector_success=r['first_success_audited'],zero_success=zero['first_success_audited'],control_success=control['first_success_audited'],
                selector_total_queries=r['total_queries'],zero_total_queries=zero['total_queries'],control_total_queries=control['total_queries']))
    for sid in plan['common_ids']:
        for method in engine.STRATEGIES:
            a=by['piguard',sid,method];b=by['protectai',sid,method]
            cross.append(dict(sample_id=sid,strategy=method,piguard_success=a['first_success_audited'],protectai_success=b['first_success_audited'],
                piguard_queries=a['total_queries'],protectai_queries=b['total_queries'],piguard_chosen_family=a['chosen_family'],protectai_chosen_family=b['chosen_family']))
    return rows,pairs,criteria,decisions,cross


def write_report(out,plan,outcomes,rows,pairs,criteria,decisions,stats):
    idx={(r['model'],r['strategy'],r['total_query_budget']):r for r in rows}
    lines=['# M3第二轮：自动选类的试探成本是否值得？','',
        '同一选择规则原样用于两个模型，无训练/梯度/模型ID输入。不改变原来的保护范围、3单位编辑、30总调用或原生256/512窗口。',
        '选择器先查原文，然后在同一静态位置依次试ZW、Cyr；任一已漏检就停。两者都未成功时，比较两次注入概率，较低者获得后续预算，平局选ZW。',
        '选择出的纯方式从自己的已付费单点继续，不重新发送已知道的原文/该探针；其他实际请求即使跨方法缓存命中也计自己的总预算。不会给每种方式各30次再合并。',
        '同探针固定ZW对照也接受探针阶段已确认成功，只在两个探针都失败后固定ZW继续；用于区分早期试探与真正选类收益。','']
    for model in ('piguard','protectai'):
        n=next(r['denominator'] for r in criteria if r['model']==model)
        lines += [f'## {model} / 同一全部{n}条','|总预算|固定ZW|固定Cyr|自动选择|同探针后固定ZW|','|---:|---:|---:|---:|---:|']
        order=('zero_reference','cyr_reference','score_selector','paired_zero_control')
        for budget in (5,10,20,30):lines.append('|'+str(budget)+'|'+ '|'.join(f"{idx[model,s,budget]['successes']}/{n}" for s in order)+'|')
        c=next(r for r in criteria if r['model']==model)
        lines += [f"- 全部样本总逻辑调用：固定ZW={c['zero_queries']}，自动选择={c['selector_queries']}，差值（选择器减ZW）={c['selector_queries']-c['zero_queries']}。含原文/试探/失败/缩减。",
            f"- 选择器新增成功：{c['selector_only_ids'] or '无'}；固定ZW成功却被选择器丢失：{c['zero_only_ids'] or '无'}。",
            f"- 成功数量不减少且总查询更少：{c['nominal_goal_met']}；更严格地保留所有ZW成功ID且总查询更少：{c['strict_paired_goal_met']}。"]
        pair=next(r for r in pairs if r['model']==model and r['total_query_budget']==30 and r['comparator']=='paired_zero_control')
        lines += [f"- 对同样试探但不按分数选类的控制：选择器独有{pair['selector_only_ids'] or '无'}，控制独有{pair['comparator_only_ids'] or '无'}。若无优势，不能把早期探针命中当成概率选择有用。",
            '- '+('本模型达到预定严格目标，但仍是已探索小基准，不保证通用省查询。' if c['strict_paired_goal_met'] else '本模型未达到预定严格目标；即便能自动选，也不能说比固定ZW更实用。')]
        choices=[r for r in decisions if r['model']==model]
        lines.append(f"- 选择ZW={sum(r['chosen_family']=='zw' for r in choices)}条、Cyr={sum(r['chosen_family']=='cyr' for r in choices)}条。逐例只用当时已付费反馈，见decisions.csv。\n")
    lines += ['## 验收与限制',
        '- 两个模型均重新跑四方法与各自75原文；原文历史标签任何变化即停止，不按新结果换22/13分母。共同4条另列，不直接把22/13成功率作强弱排名。',
        '- 预算曲线按实际首次成功查询序号，含选择阶段。原文/两个探针/后续搜索/至多3次撤销共用30；搜索最多27，不免费回收失败预留。',
        '- 主比较是固定ZW。选择器若只减少查询却丢掉成功，或只保留成功却增加查询，均不能说达到省成本目标。',
        '- 同探针控制用于解释收益来源；固定Cyr仅参照。失败或更多开销全部记录，不事后调位置、探针数、阈值或种子追结果。',
        '- 小概率探针不保证后续容易成功；这只是一个预先固定的简单选择规则，不是训练出的预测模型/新算法。',
        '- 当前数据已探索、设计在第一轮之后，无新留出；训练重叠unknown、人工语义pending。检测漏检非下游执行，字符可逆非语义保证。',
        '- 首次有效漏检须未截断、修改/ZW邻接可见、token IDs不等于原文；审计不提供给攻击器。',
        '- 缓存只省真实物理推理，不增加逻辑预算。各方法共享总耗时不能当独立速度，主张仅限逻辑查询成本。',
        f"- 新推理成本：{stats}。没有读取旧成功路径作为攻击端先验；旧ZIP只固定样本、历史原文标签及源码。"]
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_model(name,plan,out,repo,checkpoint):
    import torch
    import transformers
    if torch.__version__.split('+')[0]!='2.4.0' or transformers.__version__!='4.44.0':raise ValueError('Pinned runtime required')
    directory=out/name;directory.mkdir()
    selected=plan['source_selected'] if name=='piguard' else plan['target_selected']
    expected=plan['source_baseline'] if name=='piguard' else plan['target_baseline']
    limit=256 if name=='piguard' else 512;started=time.monotonic()
    if name=='piguard':model,device=previous.native.load_oracle_model(repo,checkpoint);tokenizer=model.tokenizer
    else:tokenizer,model,device=reader.load_target()
    versions=[(p,p._version) for p in model.parameters()];cache={};physical=0
    print(f'{name}: Actual device={device}, native window={limit}, cohort={len(selected)}, four fresh methods',flush=True)
    def predict(text):
        nonlocal physical
        if text not in cache:
            with torch.inference_mode():
                if name=='piguard':logits=model.classify(text)
                else:
                    encoded=tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=512)
                    logits=model(**{k:v.to(device) for k,v in encoded.items()}).logits
                value=(int(logits.argmax(-1).item()),float(logits.softmax(-1)[0,1].cpu()));cache[text]=value;physical+=1
        return cache[text]
    baseline={}
    for i,r in enumerate(plan['originals'],1):
        label,prob=predict(r['original']);baseline[r['sample_id']]=dict(prediction=label,probability=prob)
        reader.json_write(directory/'baseline_verification.json',baseline)
        if label!=expected[r['sample_id']]['prediction']:raise RuntimeError(f'{name} original changed: {r["sample_id"]}; STOP, no resampling')
        if i%10==0 or i==75:print(f'{name}: original verification {i}/75',flush=True)
    baseline_physical=physical;outcomes=[];logs=[]
    for i,record in enumerate(selected,1):
        calls=[]
        def oracle(text):
            before=physical;value=predict(text);calls.append(physical>before);return value
        result,trace,n=evaluate_record(record,oracle,lambda text,meta:previous.audit_window(tokenizer,text,meta,record['original'],limit))
        it=iter(calls)
        for row in trace:
            local=row['physical_cache_hit'];new=False if local else next(it)
            row.update(local_cache_hit=local,physical_cache_hit=not new,native_inference_performed=new)
        assert n==len(calls)
        if any(p._version!=v for p,v in versions):raise RuntimeError('Model parameter changed')
        outcomes.extend(result);logs.extend(trace)
        for filename,data in (('outcomes.json',outcomes),('query_trace.json',logs)):reader.json_write(directory/filename,data)
        reader.csv_write(directory/'outcomes.csv',previous.mixed.flat_rows(outcomes));reader.csv_write(directory/'query_trace.csv',previous.mixed.flat_rows(logs))
        reader.json_write(directory/'progress.json',dict(complete=False,completed_records=i,planned_records=len(selected)))
        print(f'{name}: selector comparison {i}/{len(selected)} complete; native inputs={physical}',flush=True)
    stats=dict(device=str(device),torch=torch.__version__,transformers=transformers.__version__,native_max_length=limit,
        elapsed_seconds=time.monotonic()-started,baseline_verification_requests=75,baseline_physical_inputs=baseline_physical,
        logical_attack_queries=len(logs),physical_model_inputs=physical,attack_physical_inputs=physical-baseline_physical,
        logical_original_requests=sum(r['phase']=='original' for r in logs),logical_refinement_requests=sum(r['phase']=='refinement' for r in logs),
        selector_probe_requests=sum(r.get('role')=='selection_probe' for r in logs),gradient_backward_passes=0,parameter_updates=0)
    reader.json_write(directory/'progress.json',dict(complete=True,completed_records=len(selected),stats=stats))
    return [dict(r,model=name) for r in outcomes],stats


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source',type=Path);g.add_argument('--plan',type=Path)
    p.add_argument('--repo',type=Path,default=Path('/content/m3_selector/native_piguard'));p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('/content/m3_selector_results'));args=p.parse_args()
    if args.source:
        plan=prepare(args.source);out=args.output_dir/('prepared_'+reader.stamp());out.mkdir(parents=True,exist_ok=False)
        reader.json_write(out/'selector_plan.json',plan);reader.json_write(out/'preparation.json',dict(plan_sha256=rules.digest(out/'selector_plan.json')))
        print('Frozen paid paired probe rule and unchanged source/target cohorts:',out,flush=True);return
    raw=args.plan.read_bytes();plan=json.loads(raw);validate_plan(plan)
    if reader.digest(raw)!=json.loads((args.plan.parent/'preparation.json').read_text())['plan_sha256']:raise ValueError('Plan changed')
    checkpoint=args.checkpoint or args.repo/'checkpoints/best_model.pth'
    if rules.digest(checkpoint)!=reader.CHECKPOINT_SHA or rules.digest(args.repo/'PIGuard.py')!=previous.native.MODEL_FILE_SHA:raise ValueError('PIGuard baseline differs')
    out=args.output_dir/('selector_'+reader.stamp());out.mkdir(parents=True,exist_ok=False);reader.json_write(out/'selector_plan.json',plan)
    outcomes=[];stats={}
    for name in ('piguard','protectai'):
        reader.json_write(out/'manifest.json',dict(protocol=PROTOCOL,complete=False,stage=name))
        result,info=run_model(name,plan,out,args.repo,checkpoint);outcomes.extend(result);stats[name]=info
        gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    rows,pairs,criteria,decisions,cross=analyse(plan,outcomes)
    reader.json_write(out/'all_outcomes.json',outcomes);reader.csv_write(out/'all_outcomes.csv',previous.mixed.flat_rows(outcomes))
    for filename,data in (('budget_summary.csv',rows),('paired_comparisons.csv',pairs),('decisions.csv',decisions),('cross_model_common.csv',cross)):
        if data:reader.csv_write(out/filename,previous.mixed.flat_rows(data))
    reader.json_write(out/'criteria.json',criteria);write_report(out,plan,outcomes,rows,pairs,criteria,decisions,stats)
    reader.json_write(out/'manifest.json',dict(protocol=PROTOCOL,complete=True,config=CONFIG,
        plan_sha256=reader.digest(raw),engine_sha256=rules.digest(engine.__file__),evaluator_sha256=rules.digest(__file__),previous_evaluator_sha256=PREVIOUS_SHA,
        source_file_sha256=plan['source_file_sha256'],source_checkpoint_sha256=rules.digest(checkpoint),source_model_file_sha256=rules.digest(args.repo/'PIGuard.py'),
        target_model=reader.MODEL_ID,target_revision=reader.MODEL_REVISION,common_ids=plan['common_ids'],stats=stats,
        source_selected_ids=[r['sample_id'] for r in plan['source_selected']],target_selected_ids=[r['sample_id'] for r in plan['target_selected']],
        independent_holdout=False,manual_validity_review='pending',training_overlap='unknown'))
    print((out/'REPORT_ZH.md').read_text(),flush=True);print('Complete:',out,flush=True)


if __name__=='__main__':main()
