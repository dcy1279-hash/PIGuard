"""Matched-site Cyrillic / U+200B / mixed oracle-only Unicode search."""
import math
import unicode_revisit_attack as revisit

old=revisit.old
ZW='\u200b'
SEEDS=revisit.SEEDS
STRATEGIES=('single_reference',)+tuple(f'{family}_{policy}' for family in ('cyr','zw','mixed') for policy in ('feedback','sequential'))+tuple(f'mixed_random_{s}' for s in SEEDS)
CONFIG=dict(max_total_queries=30,search_cap_including_original=27,max_refinement_queries=3,
    max_unit_edits_per_input=3,substitution_cost=1,insertion_cost=1,
    max_one_action_per_original_site=True,zero_width_character='U+200B',
    shared_sites='original round-robin Cyrillic-mappable letters strictly inside eligible words (not final letter)',
    search='section14 revisitable skeleton, same restart/add/swap/drop cycle and parent expansion tiers',
    initial_probes='up to8 identical sites; pure uses own operator, mixed alternates cyr/zw by site index',
    mixed_restart_order='round-robin sites with cyr then zw at each site; skip observed states',
    controls='cyr/zw/mixed each feedback and sequential; mixed random5230-5239; unchanged old single reference has its original broader site pool',
    random_seeds=list(SEEDS),budgets=[5,10,20,30],independent_holdout=False,
    exploratory_after_prior_results=True,threat_model='label+score oracle only, no tokenizer/model/gradient')


def sites(text):
    words=old.eligible_words(text)
    return [p for p in old.positions(text) if any(a<=p and p+1<b for a,b,w in words)]


def canonical(actions):return tuple(sorted(tuple(a) for a in actions))


def render(text,actions,reference=False):
    actions=canonical(actions);legal=set(old.positions(text) if reference else sites(text))
    if not text.isascii() or len(actions)>3 or len({p for kind,p in actions})!=len(actions):raise ValueError('Invalid unit-edit state')
    if any(kind not in ('c','z') or p not in legal or (reference and kind!='c') for kind,p in actions):raise ValueError('Illegal/protected action')
    by={p:kind for kind,p in actions};pieces=[];meta=[];index=0
    for p,c in enumerate(text):
        kind=by.get(p)
        if kind=='c':
            pieces.append(old.CYRILLIC[c]);meta.append(dict(kind='c',original_position=p,start=index,end=index+1,required_spans=[[index,index+1]]));index+=1
        else:
            pieces.append(c);index+=1
            if kind=='z':
                pieces.append(ZW);meta.append(dict(kind='z',original_position=p,start=index,end=index+1,required_spans=[[index-1,index],[index+1,index+2]]));index+=1
    changed=''.join(pieces)
    if changed.replace(ZW,'').translate(str.maketrans({v:k for k,v in old.CYRILLIC.items()}))!=text:raise AssertionError('Not exactly reversible')
    assert len(changed)==len(text)+sum(k=='z' for k,p in actions)
    return changed,meta


def neighbors(parent,pool,operation,used):
    present=set(parent);occupied={p for kind,p in parent}
    additions=[a for a in pool if a not in used]+[a for a in pool if a in used]
    if operation=='add' and len(parent)<3:
        for a in additions:
            if a[1] not in occupied:yield canonical((*parent,a))
    elif operation=='swap' and len(parent)>1:
        for a in additions:
            if a in present:continue
            for removed in reversed(parent):
                reduced=present-{removed}
                if a[1] not in {p for k,p in reduced}:yield canonical((*reduced,a))
    elif operation=='drop' and len(parent)>1:
        for removed in reversed(parent):yield canonical(present-{removed})


def annotate(result,trace,text,family,policy):
    for row in trace:
        reference=family=='reference_cyr'
        actions=canonical(row.get('actions',[('c',p) for p in row['positions']]))
        rebuilt,meta=render(text,actions,reference)
        assert rebuilt==row['text']
        row.update(actions=[list(a) for a in actions],edit_metadata=meta,
            escaped_text=row['text'].encode('unicode_escape').decode('ascii'),
            cyr_edits=sum(k=='c' for k,p in actions),zw_edits=sum(k=='z' for k,p in actions))
    first=trace[result['first_success_query']-1] if result['success'] else None
    result.update(family=family,policy=policy,shared_sites=sites(text),
        allowed_sites=old.positions(text) if family=='reference_cyr' else sites(text),
        first_success_operation=first['phase'] if first else '',
        first_success_actions=first['actions'] if first else [],
        first_success_is_mixed=bool(first and first['cyr_edits'] and first['zw_edits']))
    return result,trace


def attack(text,oracle,strategy,sample_id='input'):
    if strategy not in STRATEGIES:raise ValueError('Unknown mixed strategy')
    if strategy=='single_reference':
        r,t=old.attack(text,oracle,'single_only',sample_id);r['strategy']=strategy
        r['final_actions']=[['c',p] for p in r['final_positions']]
        return annotate(r,t,text,'reference_cyr','reference')
    family,policy=strategy.split('_',1);seed=int(policy.rsplit('_',1)[1]) if policy.startswith('random_') else None
    support=sites(text)
    if not support:raise ValueError('No common legal site; do not silently resample')
    kinds=('c','z') if family=='mixed' else ('c',) if family=='cyr' else ('z',)
    pool=[(k,p) for p in support for k in kinds]
    initial=[(kinds[i%len(kinds)],p) for i,p in enumerate(support[:8])]
    trace=[];observed={};expansions={};best=None;winner=None;first_query=0;refinement=0
    def query(actions,phase,parent=(),requested='initial'):
        nonlocal best
        if len(trace)>=30:raise RuntimeError('Query budget exceeded')
        actions=canonical(actions);parent=canonical(parent);changed,meta=render(text,actions)
        label,prob=oracle(changed)
        if label not in (0,1) or not math.isfinite(prob) or not 0<=prob<=1:raise ValueError('Invalid oracle')
        prior=observed.get(parent)
        row=dict(query_index=len(trace)+1,phase=phase,requested_operation=requested,depth=len(actions),
            actions=[list(a) for a in actions],positions=sorted(p for k,p in actions),
            parent_actions=[list(a) for a in parent],parent_positions=sorted(p for k,p in parent),
            removed_actions=[list(a) for a in sorted(set(parent)-set(actions))],
            added_actions=[list(a) for a in sorted(set(actions)-set(parent))],
            parent_query_index=prior['query_index'] if prior else 0,parent_probability=prior['injection_probability'] if prior else None,
            text=changed,prediction=int(label),injection_probability=float(prob))
        trace.append(row)
        if best is None or (prob,len(actions),actions)<(best['injection_probability'],best['depth'],canonical(best['actions'])):best=row
        if phase!='refinement':observed[actions]=row
        return row
    original=query((),'original')
    if original['prediction']!=1:raise RuntimeError('Original label changed; stop, no resampling')
    for action in initial:
        row=query((action,),'single')
        if row['prediction']==0:winner=row;first_query=row['query_index'];break
    step=0;cycle=('restart','add','swap','drop')
    while winner is None and len(trace)<27:
        used={tuple(a) for row in trace for a in row['actions']};proposal=None
        requested=cycle[step%4]
        for operation in cycle[step%4:]+cycle[:step%4]:
            if operation=='restart':
                options=[(a,) for a in pool if (a,) not in observed]
                if options:proposal=(options[0],(),operation);break
            else:
                candidates=[]
                for parent,row in observed.items():
                    if not parent:continue
                    child=next((c for c in neighbors(parent,pool,operation,used) if c not in observed),None)
                    if child is not None:candidates.append((parent,row,child))
                if not candidates:continue
                tier=min(expansions.get((p,operation),0)//2 for p,r,c in candidates)
                candidates=[x for x in candidates if expansions.get((x[0],operation),0)//2==tier]
                if policy=='feedback':candidates.sort(key=lambda x:(x[1]['injection_probability'],len(x[0]),x[0]))
                elif policy=='sequential':candidates.sort(key=lambda x:x[1]['query_index'])
                else:candidates=old.random_order(candidates,seed,sample_id,100+step)
                parent,row,child=candidates[0];expansions[parent,operation]=expansions.get((parent,operation),0)+1
                proposal=(child,parent,operation);break
        if proposal is None:break
        actions,parent,operation=proposal;row=query(actions,operation,parent,requested);step+=1
        if row['prediction']==0:winner=row;first_query=row['query_index']
    first=winner
    if winner:
        for action in reversed(canonical(winner['actions'])):
            if winner['depth']<=1 or refinement>=3:break
            row=query(tuple(a for a in canonical(winner['actions']) if a!=action),'refinement',canonical(winner['actions']),'refinement');refinement+=1
            if row['prediction']==0:winner=row
    selected=winner if winner else best
    result=dict(strategy=strategy,random_seed=seed,success=winner is not None,total_queries=len(trace),
        search_queries=len(trace)-refinement,refinement_queries=refinement,first_success_query=first_query,
        first_success_edits=first['depth'] if first else 0,first_success_positions=first['positions'] if first else [],
        first_success_text=first['text'] if first else '',final_edits=winner['depth'] if winner else 0,
        final_positions=winner['positions'] if winner else [],final_actions=winner['actions'] if winner else [],
        final_text=winner['text'] if winner else '',final_is_mixed=bool(winner and {k for k,p in winner['actions']}=={'c','z'}),
        candidate_positions=len(support),best_probability=best['injection_probability'],best_positions=best['positions'],
        best_text=best['text'],selected_probability=selected['injection_probability'],original_probability=original['injection_probability'],
        global_minimum_proven=False)
    return annotate(result,trace,text,family,policy)
