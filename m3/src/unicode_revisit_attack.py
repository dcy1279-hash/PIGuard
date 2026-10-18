"""Bounded revisitable Unicode state search; accepts label+score oracle only.

All states are sets of original offsets. Add, swap and drop do not accumulate
irreversible edits: every queried text is rebuilt from the original, <=3 edits.
Observed states remain available, including temporarily worse-scoring routes.
"""
import math

import unicode_combo_matched_attack as matched

old=matched.old
SEEDS=matched.SEEDS
STRATEGIES=('single_full','matched_feedback','feedback_add_only','sequential_revisit',
            'feedback_revisit')+tuple(f'random_revisit_{s}' for s in SEEDS)
CONFIG=dict(max_total_queries=30,max_edited_codepoints=3,search_cap_including_original=27,
    refinement_cap=3,initial_single_probes=8,initial_order='static round-robin, identical for new methods',
    operation_cycle=['restart','add','swap','drop'],
    add_only_cycle=['restart','add','add','add'],
    eligible_universe='same ALL protected-scope original Cyrillic-mappable positions',
    parent_selection='least per-operation expansion tier first (two children per tier), then score / observation order / seeded random',
    archive='retain ALL queried unsuccessful nonempty states; worse scores are not discarded',
    neighbor_order='unused positions before used; original static round-robin within each group',
    fallback='if requested operation exhausted, try later operations cyclically without queries',
    duplicate_policy='skip already observed search states without oracle calls; every actual request including cached refinement counts',
    single_control='unchanged historical up to 29 single candidates plus original',
    matched_feedback_control='unchanged section13 feedback_full with 8/8/10 allocation',
    random_seeds=list(SEEDS),budgets=[5,10,20,30],
    threat_model='simulated score-visible black box, no model/tokenizer/gradient access',
    per_input_edit_cap_not_lifetime_path_cap=True,
    exploratory_after_prior_target_results=True,independent_holdout=False)


def neighbors(parent,support,operation,used):
    """Canonical unique neighbors with a declared edit operation."""
    p=tuple(parent);present=set(p)
    additions=[a for a in support if a not in present and a not in used]
    additions += [a for a in support if a not in present and a in used]
    if operation=='add' and len(p)<3:
        for a in additions:yield tuple(sorted((*p,a)))
    elif operation=='swap' and len(p)>1:
        for a in additions:
            for removed in reversed(p):yield tuple(sorted((present-{removed})|{a}))
    elif operation=='drop' and len(p)>1:
        for removed in reversed(p):yield tuple(a for a in p if a!=removed)


def attack(text,oracle,strategy,sample_id='input'):
    if strategy not in STRATEGIES:raise ValueError('Unknown revisit strategy')
    if strategy in ('single_full','matched_feedback'):
        method='single_full' if strategy=='single_full' else 'feedback_full'
        result,trace=matched.attack(text,oracle,method,sample_id)
        result.update(strategy=strategy,search_policy='unchanged_reference',
            first_success_operation=trace[result['first_success_query']-1]['phase'] if result['success'] else '')
        return result,trace
    if not text.isascii():raise ValueError('Frozen experiment requires ASCII originals')
    support=old.positions(text)
    if len(support)<3:raise ValueError('Insufficient legal positions')
    seed=int(strategy.rsplit('_',1)[1]) if strategy.startswith('random_revisit_') else None
    feedback=strategy in ('feedback_revisit','feedback_add_only')
    cycle=CONFIG['add_only_cycle'] if strategy=='feedback_add_only' else CONFIG['operation_cycle']
    trace=[];observed={};expansions={};best=None;winner=None;first_success=0;refinement=0

    def query(edits,phase,parent=(),requested='initial',parent_rank=0):
        nonlocal best
        if len(trace)>=30:raise RuntimeError('Total query cap exceeded')
        edits=tuple(sorted(edits));parent=tuple(parent)
        changed=old.apply_edits(text,edits);label,prob=oracle(changed)
        if label not in (0,1) or not math.isfinite(prob) or not 0<=prob<=1:raise ValueError('Invalid label+score oracle')
        parent_row=observed.get(parent)
        row=dict(query_index=len(trace)+1,phase=phase,depth=len(edits),positions=list(edits),
            parent_positions=list(parent),text=changed,prediction=int(label),injection_probability=float(prob),
            requested_operation=requested,removed_positions=sorted(set(parent)-set(edits)),
            added_positions=sorted(set(edits)-set(parent)),parent_archive_rank=parent_rank,
            parent_query_index=parent_row['query_index'] if parent_row else 0,
            parent_probability=parent_row['injection_probability'] if parent_row else None,
            score_improved_vs_parent=bool(parent_row and prob<parent_row['injection_probability']),
            returned_to_older_state=bool(parent_row and parent_row['query_index']<len(trace)),
            search_policy='feedback' if feedback else ('random' if seed is not None else 'sequential'))
        trace.append(row)
        if best is None or (prob,len(edits),edits)<(best['injection_probability'],best['depth'],tuple(best['positions'])):best=row
        if phase!='refinement':observed[edits]=row
        return row

    original=query((),'original')
    if original['prediction']!=1:raise RuntimeError('Original no longer detected; stop, do not resample')
    for p in support[:8]:
        row=query((p,),'single')
        if row['prediction']==0:winner=row;first_success=row['query_index'];break
    step=0
    while winner is None and len(trace)<27:
        used={p for row in trace for p in row['positions']}
        requested=cycle[step%len(cycle)];proposal=None
        for operation in cycle[step%len(cycle):]+cycle[:step%len(cycle)]:
            if operation=='restart':
                options=[(p,) for p in support if (p,) not in observed]
                if options:proposal=(options[0],(),operation,0);break
            else:
                candidates=[]
                for parent,row in observed.items():
                    if not parent:continue
                    child=next((n for n in neighbors(parent,support,operation,used) if n not in observed),None)
                    if child is not None:candidates.append((parent,row,child))
                if not candidates:continue
                tier=min(expansions.get((p,operation),0)//2 for p,r,c in candidates)
                candidates=[x for x in candidates if expansions.get((x[0],operation),0)//2==tier]
                if feedback:candidates.sort(key=lambda x:(x[1]['injection_probability'],len(x[0]),x[0]))
                elif seed is None:candidates.sort(key=lambda x:x[1]['query_index'])
                else:candidates=old.random_order(candidates,seed,sample_id,100+step)
                parent,parent_row,child=candidates[0]
                expansions[parent,operation]=expansions.get((parent,operation),0)+1
                proposal=(child,parent,operation,1);break
        if proposal is None:break
        edits,parent,operation,rank=proposal
        row=query(edits,operation,parent,requested,rank);step+=1
        if row['prediction']==0:winner=row;first_success=row['query_index']
    first=winner
    if winner:
        for p in reversed(winner['positions']):
            if len(winner['positions'])<=1 or refinement>=3:break
            row=query(tuple(a for a in winner['positions'] if a!=p),'refinement',tuple(winner['positions']),'refinement')
            refinement+=1
            if row['prediction']==0:winner=row
    selected=winner if winner else best
    result=dict(strategy=strategy,random_seed=seed,success=winner is not None,
        total_queries=len(trace),search_queries=len(trace)-refinement,refinement_queries=refinement,
        first_success_query=first_success,first_success_edits=first['depth'] if first else 0,
        first_success_positions=first['positions'] if first else [],first_success_text=first['text'] if first else '',
        final_edits=winner['depth'] if winner else 0,final_positions=winner['positions'] if winner else [],
        final_text=winner['text'] if winner else '',candidate_positions=len(support),
        best_probability=best['injection_probability'],best_positions=best['positions'],best_text=best['text'],
        selected_probability=selected['injection_probability'],original_probability=original['injection_probability'],
        global_minimum_proven=False,search_policy='revisitable_archive',
        first_success_operation=first['phase'] if first else '',
        operation_counts={op:sum(r['phase']==op for r in trace) for op in ('restart','add','swap','drop')},
        worse_score_states_retained=sum(r['phase'] in ('add','swap','drop') and not r['score_improved_vs_parent'] for r in trace))
    return matched.annotate(result,trace,text,strategy,'full')
