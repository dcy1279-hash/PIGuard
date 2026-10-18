"""Read-only archive replay. Uses stored scores; no fresh native model inference."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import eval_m3_blackbox_combo as m


def main(archive,source):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        read=lambda name:json.loads(z.read(name))
        manifest=read('manifest.json');plan=read('combo_plan.json')
        outcomes=read('outcomes.json');trace=read('query_trace.json')
        assert manifest['complete'] is True and manifest['originals_reproduced'] is True
        m.validate_plan(plan);assert plan==m.prepare(source)
        assert manifest['plan_sha256']==hashlib.sha256(z.read('combo_plan.json')).hexdigest()
        for key in ('protocol','config','engine_sha256','evaluator_sha256','reader_sha256',
                    'generator_sha256','checkpoint_sha256','model_file_sha256','source_file_sha256'):
            assert manifest[key]==plan[key],key
        assert len(outcomes)==22*13 and len(trace)==manifest['logical_oracle_queries']
        allout,alltrace=[],[];physical=0
        for record in plan['selected']:
            sub=[r for r in trace if r['sample_id']==record['sample_id']]
            lookup={};audits={}
            for row in sub:
                value=(row['prediction'],row['injection_probability'])
                if row['text'] in lookup:assert lookup[row['text']]==value
                lookup[row['text']]=value
                audit={k:row[k] for k in ('token_count','truncated','edited_span_visibility')}
                if row['text'] in audits:assert audits[row['text']]==audit
                audits[row['text']]=audit
            replay,logs,n=m.evaluate_record(record,lambda t:lookup[t],lambda t,s:audits[t])
            allout.extend(replay);alltrace.extend(logs);physical+=n
        assert allout==outcomes and alltrace==trace
        assert physical==manifest['physical_model_inputs']
        rows,random_rows,pairs=m.summary(outcomes,[r['sample_id'] for r in plan['selected']])
        for filename,expected in (('budget_summary.csv',rows),('random_seed_summary.csv',random_rows),('paired_comparisons.csv',pairs),
                                  ('outcomes.csv',m.flat_rows(outcomes)),('query_trace.csv',m.flat_rows(trace))):
            saved=list(csv.DictReader(io.StringIO(z.read(filename).decode())))
            assert len(saved)==len(expected)
            for a,b in zip(expected,saved):
                assert set(a)==set(b)
                for key,value in a.items():
                    if isinstance(value,float):
                        assert math.isclose(float(b[key]),value,rel_tol=1e-12,abs_tol=1e-12),(filename,key,value,b[key])
                    else:
                        assert b[key]==('' if value is None else str(value)),(filename,key,value,b[key])
        eligible=list(csv.DictReader(io.StringIO(z.read('eligibility.csv').decode())))
        assert eligible==[dict(sample_id=r['sample_id'],reason=m.eligibility(r)) for r in plan['originals']]
        assert sum(r['phase']=='original' for r in trace)==manifest['logical_original_queries']==286
        assert sum(r['phase']=='refinement' for r in trace)==manifest['logical_refinement_queries']
        assert manifest['gradient_forward_passes']==manifest['gradient_backward_passes']==manifest['parameter_updates']==0
        print('PASS: CRC; source/frozen cohort/code/model hashes; all 286 outcomes and oracle paths;')
        print('52 budget rows, 4 seed summaries, 48 paired rows and JSON/CSV copies; cache and total budgets.')
        print('Stored native scores replayed; no new inference or independent gradient calculation.')
        print('Device, seconds, logical, physical:',manifest['device'],manifest['elapsed_seconds'],len(trace),physical)
        print('Truncated / invisible variants:',sum(r['truncated'] for r in trace),
              sum(bool(r['positions']) and r['edited_span_visibility']!='full' for r in trace))
        by={(r['sample_id'],r['strategy']):r for r in outcomes}
        for method in m.engine.STRATEGIES:
            sub=[r for r in outcomes if r['strategy']==method]
            print(method,'success',sum(r['first_success_audited'] for r in sub),
                  'total queries',sum(r['total_queries'] for r in sub),
                  'refinement',sum(r['refinement_queries'] for r in sub))
        feedback=[r for r in outcomes if r['strategy']=='feedback_combo']
        originals={r['sample_id']:r['original'] for r in plan['selected']}
        for r in feedback:
            if r['first_success_audited'] and not by[r['sample_id'],'single_only']['success']:
                singles=[x for x in trace if x['sample_id']==r['sample_id'] and x['strategy']=='single_only' and x['phase']=='single']
                legal=m.engine.positions(originals[r['sample_id']])
                assert {tuple(x['positions']) for x in singles}=={(pos,) for pos in legal}
                assert len(singles)==len(legal) and all(x['prediction']==1 for x in singles)
                assert r['final_edits']==2
                print('Exhaustive fixed-scope single-position failure, pair success:',r['sample_id'],len(legal),'positions')
        print('Feedback multi-edit first successes:')
        for r in feedback:
            if r['success'] and r['first_success_edits']>1:
                print(r['sample_id'],'first query',r['first_success_query'],'first/final edits',
                      r['first_success_edits'],r['final_edits'],'positions',r['final_positions'],
                      'single-only success',by[r['sample_id'],'single_only']['success'],
                      'full position pool',r['candidate_positions'],
                      'single queries',by[r['sample_id'],'single_only']['total_queries'])
        print('All feedback failures:',[r['sample_id'] for r in feedback if not r['success']])
        print('Feedback-only vs sequential at total 10:',[r['sample_id'] for r in feedback if r['success'] and r['first_success_query']<=10
            and (not by[r['sample_id'],'sequential_combo']['success'] or by[r['sample_id'],'sequential_combo']['first_success_query']>10)])
        print('Archive SHA256:',hashlib.sha256(Path(archive).read_bytes()).hexdigest())


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
