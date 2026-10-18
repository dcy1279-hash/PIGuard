"""Two-probe family selection with ONE shared30-query budget.

No training, model ID, historical outcome, tokenization, or gradient input.
The chosen pure-family state machine resumes the already queried probe.
"""
import math
import unicode_mixed_attack as fixed

STRATEGIES=('zero_reference','cyr_reference','score_selector','paired_zero_control')
CONFIG=dict(max_total_queries=30,search_cap_including_original=27,max_refinement_queries=3,
    max_unit_edits_per_input=3,initial_probe_pairs=1,probe_order=['z','c'],
    probe_site='first static legal shared word-internal site; same site for both families',
    early_stop='any native label0 stops immediately, including a selection probe',
    selection='after two unsuccessful probes, lower observed injection probability; ties choose zero',
    continuation='resume selected immutable pure-family search order; saved own probe not sent again',
    control='same paired probes/early stopping, but force zero continuation if both probes unsuccessful',
    no_free_queries=True,no_model_id_or_historical_results_input=True,
    no_mixed_family_edit_sets=True,methods=list(STRATEGIES),budgets=[5,10,20,30],
    independent_holdout=False,exploratory_after_step1=True)


class FamilySearch:
    """Proposal-only equivalent of fixed pure feedback search, resumable."""
    def __init__(self,support,kind):
        self.kind=kind;self.pool=[(kind,p) for p in support];self.initial=self.pool[:8]
        self.observed={};self.expansions={};self.warm_index=0;self.step=0

    def observe(self,row):
        actions=fixed.canonical(row['actions'])
        if actions and all(k==self.kind for k,p in actions):self.observed[actions]=row

    def next_proposal(self):
        while self.warm_index<len(self.initial):
            action=self.initial[self.warm_index];self.warm_index+=1
            if (action,) not in self.observed:return (action,),(),'single','initial'
        cycle=('restart','add','swap','drop');requested=cycle[self.step%4]
        used={a for state in self.observed for a in state}
        for operation in cycle[self.step%4:]+cycle[:self.step%4]:
            if operation=='restart':
                option=next(((a,) for a in self.pool if (a,) not in self.observed),None)
                if option is not None:
                    self.step+=1;return option,(),operation,requested
            else:
                candidates=[]
                for parent,row in self.observed.items():
                    child=next((c for c in fixed.neighbors(parent,self.pool,operation,used) if c not in self.observed),None)
                    if child is not None:candidates.append((parent,row,child))
                if not candidates:continue
                tier=min(self.expansions.get((p,operation),0)//2 for p,r,c in candidates)
                candidates=[x for x in candidates if self.expansions.get((x[0],operation),0)//2==tier]
                parent,row,child=min(candidates,key=lambda x:(x[1]['injection_probability'],len(x[0]),x[0]))
                self.expansions[parent,operation]=self.expansions.get((parent,operation),0)+1
                self.step+=1;return child,parent,operation,requested
        return None


def attack(text,oracle,strategy,sample_id='input'):
    if strategy not in STRATEGIES:raise ValueError('Unknown selector strategy')
    if strategy in ('zero_reference','cyr_reference'):
        family='zw' if strategy=='zero_reference' else 'cyr'
        result,trace=fixed.attack(text,oracle,f'{family}_feedback',sample_id)
        result.update(strategy=strategy,chosen_family=family,choice_reason='fixed_reference',choice_after_query=0,
            selector_probe_queries=0,choice_probabilities={},own_probe_reused=False)
        return result,trace
    if not text.isascii():raise ValueError('ASCII originals required')
    support=fixed.sites(text)
    if not support:raise ValueError('No shared legal site; stop, no resampling')
    searches={k:FamilySearch(support,k) for k in ('z','c')}
    trace=[];observed={};best=None;winner=None;first_query=0;refinement=0
    chosen=None;reason='';choice_query=0;scores={}
    def query(actions,phase,parent=(),requested='initial',role='continuation'):
        nonlocal best
        if len(trace)>=30:raise RuntimeError('One global total-query cap exceeded')
        actions=fixed.canonical(actions);parent=fixed.canonical(parent)
        changed,meta=fixed.render(text,actions);label,prob=oracle(changed)
        if label not in (0,1) or not math.isfinite(prob) or not 0<=prob<=1:raise ValueError('Invalid label+probability')
        previous=observed.get(parent)
        row=dict(query_index=len(trace)+1,phase=phase,requested_operation=requested,role=role,
            depth=len(actions),actions=[list(a) for a in actions],positions=sorted(p for k,p in actions),
            parent_actions=[list(a) for a in parent],parent_positions=sorted(p for k,p in parent),
            removed_actions=[list(a) for a in sorted(set(parent)-set(actions))],
            added_actions=[list(a) for a in sorted(set(actions)-set(parent))],
            parent_query_index=previous['query_index'] if previous else 0,
            parent_probability=previous['injection_probability'] if previous else None,
            text=changed,prediction=int(label),injection_probability=float(prob))
        trace.append(row)
        if best is None or (prob,len(actions),actions)<(best['injection_probability'],best['depth'],fixed.canonical(best['actions'])):best=row
        if phase!='refinement':observed[actions]=row
        return row
    original=query((),'original',role='original')
    if original['prediction']!=1:raise RuntimeError('Original no longer detected; stop, no resampling')
    for kind in ('z','c'):
        row=query(((kind,support[0]),),'single',role='selection_probe')
        searches[kind].observe(row);scores[kind]=row['injection_probability']
        if row['prediction']==0:
            winner=row;first_query=row['query_index'];chosen=kind;reason='confirmed_probe_success';choice_query=len(trace);break
    if winner is None:
        chosen='z' if strategy=='paired_zero_control' or scores['z']<=scores['c'] else 'c'
        reason='forced_zero_after_same_probes' if strategy=='paired_zero_control' else 'lower_probe_probability_zero_on_tie'
        choice_query=len(trace)
        while len(trace)<27:
            proposal=searches[chosen].next_proposal()
            if proposal is None:break
            actions,parent,phase,requested=proposal
            row=query(actions,phase,parent,requested);searches[chosen].observe(row)
            if row['prediction']==0:winner=row;first_query=row['query_index'];break
    first=winner
    if winner:
        for action in reversed(fixed.canonical(winner['actions'])):
            if winner['depth']<=1 or refinement>=3:break
            row=query(tuple(a for a in fixed.canonical(winner['actions']) if a!=action),'refinement',fixed.canonical(winner['actions']),'refinement','refinement')
            refinement+=1
            if row['prediction']==0:winner=row
    selected=winner if winner else best
    family='zw' if chosen=='z' else 'cyr'
    result=dict(strategy=strategy,random_seed=None,success=winner is not None,total_queries=len(trace),
        search_queries=len(trace)-refinement,refinement_queries=refinement,first_success_query=first_query,
        first_success_edits=first['depth'] if first else 0,first_success_positions=first['positions'] if first else [],
        first_success_text=first['text'] if first else '',final_edits=winner['depth'] if winner else 0,
        final_positions=winner['positions'] if winner else [],final_actions=winner['actions'] if winner else [],
        final_text=winner['text'] if winner else '',final_is_mixed=False,candidate_positions=len(support),
        best_probability=best['injection_probability'],best_positions=best['positions'],best_text=best['text'],
        selected_probability=selected['injection_probability'],original_probability=original['injection_probability'],
        global_minimum_proven=False,chosen_family=family,choice_reason=reason,choice_after_query=choice_query,
        selector_probe_queries=sum(r.get('role')=='selection_probe' for r in trace),choice_probabilities=scores,
        own_probe_reused=reason!='confirmed_probe_success')
    return fixed.annotate(result,trace,text,family,'adaptive' if strategy=='score_selector' else 'paired_control')
