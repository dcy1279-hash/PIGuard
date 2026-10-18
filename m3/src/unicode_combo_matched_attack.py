"""Full-position eligible universe, fixed query/edit caps; old engine unmodified.

Initial probes remain at most eight. Later edits may use ALL legal positions,
prioritising never-edited positions. Position availability != exhaustive testing.
"""
import math

import unicode_combo_attack as old

MAX_QUERIES=old.MAX_QUERIES
MAX_EDITS=old.MAX_EDITS
STAGE_LIMITS=old.STAGE_LIMITS
SEEDS=old.RANDOM_SEEDS
STRATEGIES=('single_full','feedback_prefix8','sequential_full','feedback_full')+tuple(f'random_full_{s}' for s in SEEDS)
CONFIG=dict(max_total_queries=MAX_QUERIES,max_edited_codepoints=MAX_EDITS,
    original_query_counted=True,max_refinement_queries=old.REFINEMENT_QUERIES,
    stage_caps=list(STAGE_LIMITS),beam_width=old.BEAM_WIDTH,seeds=list(SEEDS),
    eligible_universe='ALL static protected-scope Cyrillic-mappable letter positions, for every full method',
    initial_probes='up to eight round-robin positions; random changes order, not initial set',
    extension_order='never-edited legal positions first, then previously edited positions; static round-robin within group',
    parent_selection='feedback: two lowest observed p1; sequential: first two observed; random: seeded two observed',
    prefix8_control='exact historical feedback search, restricted to first eight positions',
    single_control='historical single-only schedule, up to 29 positions',
    budgets=[5,10,20,30],independent_holdout=False,
    adaptation_after_prior_target_results=True)


def annotate(result,trace,text,method,scope):
    support=old.positions(text);initial=support[:8]
    used=sorted({p for row in trace for p in row['positions']})
    result.update(strategy=method,candidate_scope=scope,
        anchor_positions=initial,
        allowed_positions=support if scope=='full' else initial,
        allowed_position_count=len(support if scope=='full' else initial),
        initial_catalog=initial,queried_positions=used,
        queried_beyond_initial8=sorted(set(used)-set(initial)))
    for row in trace:
        row['outside_initial8']=bool(set(row['positions'])-set(initial))
    return result,trace


def extension_order(support,trace,seed,sid,depth):
    used={p for row in trace for p in row['positions']}
    unseen=[p for p in support if p not in used]
    seen=[p for p in support if p in used]
    if seed is not None:
        unseen=old.random_order(unseen,seed,sid,depth*10+2)
        seen=old.random_order(seen,seed,sid,depth*10+3)
    return unseen+seen


def attack(text,oracle,strategy,sample_id='input'):
    if strategy not in STRATEGIES:raise ValueError('Unknown matched-scope strategy')
    if strategy in ('single_full','feedback_prefix8'):
        original='single_only' if strategy=='single_full' else 'feedback_combo'
        result,trace=old.attack(text,oracle,original,sample_id)
        return annotate(result,trace,text,strategy,'full' if strategy=='single_full' else 'prefix8')
    if not text.isascii():raise ValueError('Frozen experiment requires ASCII originals')
    support=old.positions(text)
    if len(support)<3:raise ValueError('Insufficient legal positions')
    seed=int(strategy.rsplit('_',1)[1]) if strategy.startswith('random_full_') else None
    trace=[];best=None;winner=None;first_success=0
    def query(edits,phase,parent=()):
        nonlocal best
        if len(trace)>=MAX_QUERIES:raise RuntimeError('Total query cap exceeded')
        changed=old.apply_edits(text,edits);pred,prob=oracle(changed)
        if pred not in (0,1) or not math.isfinite(prob) or not 0<=prob<=1:
            raise ValueError('Invalid label+score oracle response')
        row=dict(query_index=len(trace)+1,phase=phase,depth=len(edits),positions=list(edits),
            parent_positions=list(parent),text=changed,prediction=int(pred),injection_probability=float(prob))
        trace.append(row)
        if best is None or (prob,len(edits),tuple(edits))<(best['injection_probability'],best['depth'],tuple(best['positions'])):
            best=row
        return row
    original=query((),'original')
    if original['prediction']!=1:raise RuntimeError('Original no longer detected; stop, do not resample')
    previous=[]
    for depth,cap in enumerate(STAGE_LIMITS,1):
        if depth==1:
            initial=support[:cap]
            if seed is not None:initial=old.random_order(initial,seed,sample_id,1)
            proposals=[((p,),()) for p in initial]
        else:
            if strategy=='feedback_full':
                parents=sorted(previous,key=lambda r:(r['injection_probability'],tuple(r['positions'])))[:old.BEAM_WIDTH]
            elif strategy=='sequential_full':parents=previous[:old.BEAM_WIDTH]
            else:parents=old.random_order(previous,seed,sample_id,depth*10+1)[:old.BEAM_WIDTH]
            additions=extension_order(support,trace,seed,sample_id,depth)
            proposals=old.children([tuple(r['positions']) for r in parents],additions)
        observed=[]
        for edits,parent in proposals[:cap]:
            if len(trace)>=old.SEARCH_QUERY_CAP:break
            row=query(edits,'single' if depth==1 else 'combination',parent)
            observed.append(row)
            if row['prediction']==0:winner=row;first_success=row['query_index'];break
        if winner is not None:break
        previous=observed
        if not previous:break
    first_winner=winner;refinement=0
    if winner:
        for p in reversed(winner['positions']):
            if len(winner['positions'])<=1 or refinement>=old.REFINEMENT_QUERIES:break
            reduced=tuple(i for i in winner['positions'] if i!=p)
            row=query(reduced,'refinement',tuple(winner['positions']));refinement+=1
            if row['prediction']==0:winner=row
    selected=winner if winner else best
    result=dict(strategy=strategy,random_seed=seed,success=winner is not None,
        total_queries=len(trace),search_queries=len(trace)-refinement,refinement_queries=refinement,
        first_success_query=first_success,first_success_edits=first_winner['depth'] if first_winner else 0,
        first_success_positions=first_winner['positions'] if first_winner else [],
        first_success_text=first_winner['text'] if first_winner else '',
        final_edits=winner['depth'] if winner else 0,final_positions=winner['positions'] if winner else [],
        final_text=winner['text'] if winner else '',candidate_positions=len(support),
        best_probability=best['injection_probability'],best_positions=best['positions'],best_text=best['text'],
        selected_probability=selected['injection_probability'],original_probability=original['injection_probability'],
        global_minimum_proven=False)
    return annotate(result,trace,text,strategy,'full')
