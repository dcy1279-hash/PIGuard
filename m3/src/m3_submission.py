"""English submission front end. Archived-score checks are not new inference."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(archive, member):
    with zipfile.ZipFile(archive) as z:
        return json.loads(z.read(member))


def read_csv(archive, member):
    with zipfile.ZipFile(archive) as z:
        return list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8'))))


def csv_types(rows):
    """Restore numeric fields written by the original CSV exporter."""
    def cast(value):
        if value == '':
            return None
        if value in ('True', 'False'):
            return value == 'True'
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return [{k: cast(v) for k, v in r.items()} for r in rows]


def assert_csv(expected, saved):
    assert len(expected) == len(saved), 'Different row counts'
    for a, b in zip(expected, saved):
        assert set(a) == set(b), 'Different columns'
        for k, value in a.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert math.isclose(float(b[k]), value, rel_tol=1e-11, abs_tol=1e-11), k
            elif isinstance(value, (list, dict)):
                assert json.loads(b[k]) == value, k
            else:
                assert b[k] == ('' if value is None else str(value)), k


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.inventory = json.loads((self.root / 'inventory.json').read_text())
        self.paths = {key: self.root / name for key, name in self.inventory['archives'].items()}
        self.output = self.root / 'submission_outputs'
        self.output.mkdir(exist_ok=True)

    def json(self, key, member):
        return read_json(self.paths[key], member)

    def csv(self, key, member):
        return read_csv(self.paths[key], member)

    def verify_files(self):
        for name, expected in self.inventory['files_sha256'].items():
            assert sha((self.root / name).read_bytes()) == expected, f'Changed file: {name}'
        rows = []
        for key, path in self.paths.items():
            with zipfile.ZipFile(path) as z:
                assert len(z.namelist()) == len(set(z.namelist()))
                assert z.testzip() is None, f'Invalid ZIP: {key}'
            manifest = self.json(key, 'manifest.json')
            assert manifest['complete'] is True, f'Incomplete experiment: {key}'
            rows.append(dict(Experiment=key, File=path.name,
                             Device=manifest.get('device', 'see per-model statistics'),
                             SHA256=sha(path.read_bytes())))
        return rows

    def verify_stability(self):
        import eval_m3_unicode_stability as rules
        manifest = self.json('stability', 'manifest.json')
        assert manifest['script_sha256'] == sha(Path(rules.__file__).read_bytes())
        for name, expected in manifest['datasets_sha256'].items():
            assert sha((self.root / 'datasets' / name).read_bytes()) == expected
        clean = csv_types(self.csv('stability', 'clean_controls.csv'))
        attempts = csv_types(self.csv('stability', 'attempts.csv'))
        records = rules.load_records(self.root / 'datasets')
        assert len(clean) == len(records) == 464 and len(attempts) == 16101
        by = {r['sample_id']: r for r in clean}
        observed = {(r['sample_id'], r['family'], r['rank']): r for r in attempts}
        assert len(observed) == len(attempts)
        for record in records:
            sid = record['sample_id']
            assert by[sid]['original'] == record['text'] and by[sid]['label'] == record['label']
            for family in rules.FAMILIES:
                for candidate in rules.candidates(record['text'], family):
                    row = observed[sid, family, candidate['rank']]
                    assert row['text'] == candidate['text']
                    assert row['new_error'] == int(by[sid]['baseline_correct'] and row['prediction'] != row['label'])
                    for policy in rules.REPAIRS:
                        repaired = rules.repair(row['text'], policy)
                        assert row[policy + '_exact_original'] == int(repaired == record['text'])
                        assert row[policy + '_correct'] == int(row[policy + '_prediction'] == row['label'])
        cs, fs, cats = rules.summarise(attempts, clean)
        for member, expected in [('clean_summary.csv', cs), ('family_budget_summary.csv', fs), ('category_summary.csv', cats)]:
            assert_csv(expected, self.csv('stability', member))
        return clean, attempts

    def verify_selector(self):
        import eval_m3_selector as module
        plan = self.json('selector', 'selector_plan.json')
        manifest = self.json('selector', 'manifest.json')
        module.validate_plan(plan)
        assert plan == module.prepare(self.paths['complement'])
        assert manifest['plan_sha256'] == sha(zipfile.ZipFile(self.paths['selector']).read('selector_plan.json'))
        all_outcomes = []
        for model, cohort, baseline_key in [('piguard', 'source_selected', 'source_baseline'), ('protectai', 'target_selected', 'target_baseline')]:
            baseline = self.json('selector', model + '/baseline_verification.json')
            assert set(baseline) == set(plan[baseline_key])
            assert all(baseline[s]['prediction'] == plan[baseline_key][s]['prediction'] for s in baseline)
            saved = self.json('selector', model + '/query_trace.json')
            cache = {r['original']: (baseline[r['sample_id']]['prediction'], baseline[r['sample_id']]['probability']) for r in plan['originals']}
            physical = len(cache)
            outcomes, traces = [], []
            for record in plan[cohort]:
                old = [r for r in saved if r['sample_id'] == record['sample_id']]
                lookup, audits = {}, {}
                for row in old:
                    value = (row['prediction'], row['injection_probability'])
                    assert row['text'] not in lookup or lookup[row['text']] == value
                    lookup[row['text']] = value
                    audits[row['text']] = {k: row[k] for k in ('token_count', 'truncated', 'edited_span_visibility', 'same_token_ids_as_original', 'zero_width_offset_covered', 'visibility_definition')}
                calls = []
                def oracle(text):
                    nonlocal physical
                    fresh = text not in cache
                    if fresh:
                        cache[text] = lookup[text]
                        physical += 1
                    assert cache[text] == lookup[text]
                    calls.append(fresh)
                    return cache[text]
                result, trace, n = module.evaluate_record(record, oracle, lambda text, meta: audits[text])
                assert n == len(calls)
                it = iter(calls)
                for row in trace:
                    local = row['physical_cache_hit']
                    fresh = False if local else next(it)
                    row.update(local_cache_hit=local, physical_cache_hit=not fresh, native_inference_performed=fresh)
                assert trace == old
                outcomes.extend(result)
                traces.extend(trace)
            assert outcomes == self.json('selector', model + '/outcomes.json')
            assert traces == saved
            assert physical == manifest['stats'][model]['physical_model_inputs']
            assert len(traces) == manifest['stats'][model]['logical_attack_queries']
            all_outcomes.extend(dict(r, model=model) for r in outcomes)
        assert all_outcomes == self.json('selector', 'all_outcomes.json')
        rows, pairs, criteria, decisions, cross = module.analyse(plan, all_outcomes)
        assert criteria == self.json('selector', 'criteria.json')
        for name, data in [('budget_summary.csv', rows), ('paired_comparisons.csv', pairs), ('decisions.csv', decisions), ('cross_model_common.csv', cross)]:
            assert_csv(module.previous.mixed.flat_rows(data), self.csv('selector', name))
        return all_outcomes

    def verify_all(self):
        """Regenerate candidates and paths using saved scores, not new predictions."""
        files = self.verify_files()
        self.verify_stability()
        import audit_m3_gradient_text_archive as grad_audit
        import audit_m3_blackbox_combo_archive as combo_audit
        with contextlib.redirect_stdout(io.StringIO()):
            grad_audit.main(self.paths['gradient'], self.paths['stability'])
            combo_audit.main(self.paths['combination'], self.paths['stability'])
        self.verify_selector()
        import eval_m3_transfer as transfer
        assert self.json('transfer', 'frozen_plan.json') == transfer.freeze(self.paths['stability'])
        print('Verified: immutable files, candidate generation, saved search paths and result tables.')
        print('Mode: analysis of archived real model runs. No new model inference or gradients.')
        return files

    def gradient_table(self):
        rows = self.csv('gradient', 'budget_summary.csv')
        random = self.csv('gradient', 'random_seed_summary.csv')
        lookup = {(r['strategy'], int(r['budget'])): r for r in rows}
        return [dict(Candidate_budget=int(r['budget']), Sequential=int(lookup['left_to_right', int(r['budget'])]['new_error_samples']),
                     Gradient=int(lookup['gradient', int(r['budget'])]['new_error_samples']),
                     Random_mean=float(r['mean_new_error_samples']), Random_range=f"{r['min_new_error_samples']}-{r['max_new_error_samples']}", Denominator=22)
                for r in random]

    def combination_table(self):
        rows = self.csv('combination', 'budget_summary.csv')
        random = self.csv('combination', 'random_seed_summary.csv')
        lookup = {(r['strategy'], int(r['total_query_budget'])): r for r in rows}
        return [dict(Total_budget=int(r['total_query_budget']), Single=int(lookup['single_only', int(r['total_query_budget'])]['audited_successes']),
                     Sequential_combo=int(lookup['sequential_combo', int(r['total_query_budget'])]['audited_successes']),
                     Feedback_combo=int(lookup['feedback_combo', int(r['total_query_budget'])]['audited_successes']),
                     Random_mean=float(r['mean_successes']), Denominator=22) for r in random]

    def joint_cases(self):
        import unicode_combo_attack as engine
        outcomes = self.json('combination', 'outcomes.json')
        trace = self.json('combination', 'query_trace.json')
        plan = self.json('combination', 'combo_plan.json')
        originals = {r['sample_id']: r['original'] for r in plan['selected']}
        by = {(r['sample_id'], r['strategy']): r for r in outcomes}
        rows = []
        for r in outcomes:
            sid = r['sample_id']
            if r['strategy'] != 'feedback_combo' or not r['first_success_audited'] or by[sid, 'single_only']['success']:
                continue
            singles = [s for s in trace if s['sample_id'] == sid and s['strategy'] == 'single_only' and s['phase'] == 'single']
            legal = engine.positions(originals[sid])
            assert {tuple(s['positions']) for s in singles} == {(p,) for p in legal}
            assert all(s['prediction'] == 1 for s in singles)
            rows.append(dict(Sample=sid, All_legal_single_edits_tested=len(legal), Single_successes=0,
                             Successful_pair=r['final_positions'], Pair_first_query=r['first_success_query']))
        assert len(rows) == 3
        return rows

    def family_table(self):
        rows = self.json('complement', 'all_outcomes.json')
        table = []
        for model, denominator in [('piguard', 22), ('protectai', 13)]:
            for method in ('cyr_feedback', 'zw_feedback', 'mixed_feedback'):
                sub = [r for r in rows if r['model'] == model and r['strategy'] == method]
                assert len(sub) == denominator
                table.append(dict(Model=model, Method=method, Successes=sum(r['first_success_audited'] for r in sub),
                                  Denominator=denominator, Total_queries=sum(r['total_queries'] for r in sub)))
        return table

    def selector_table(self):
        outcomes = self.json('selector', 'all_outcomes.json')
        rows = []
        for r in self.json('selector', 'criteria.json'):
            control = [s for s in outcomes if s['model'] == r['model'] and s['strategy'] == 'paired_zero_control']
            rows.append(dict(Model=r['model'], Fixed_zero_success=r['zero_successes'], Selector_success=r['selector_successes'],
                             Same_probe_zero_success=sum(s['first_success_audited'] for s in control),
                             Denominator=r['denominator'], Fixed_zero_queries=r['zero_queries'], Selector_queries=r['selector_queries'],
                             Same_probe_zero_queries=sum(s['total_queries'] for s in control), Lost_fixed_zero_cases=r['zero_only_ids']))
        return rows

    def stability_table(self):
        rows = self.csv('stability', 'family_budget_summary.csv')
        return [dict(Dataset=r['dataset'], Family=r['family'], Correct_originals=int(r['baseline_correct']),
                     New_error_exists=int(r['new_error_samples']), After_NFKC=int(r['nfkc_postrepair_any_error_samples']),
                     After_targeted_repair=int(r['mixed_script_postrepair_any_error_samples']))
                for r in rows if r['budget'] == 'all']

    def representative_examples(self):
        """Keep benchmark wording unchanged; display invisible edits explicitly."""
        plan = self.json('selector', 'selector_plan.json')
        originals = {r['sample_id']: r['original'] for r in plan['originals']}
        rows = []
        for model, sid in [('piguard', 'BIPIA_TEXT_056'), ('protectai', 'BIPIA_TEXT_036')]:
            outcome = next(r for r in self.json('selector', model + '/outcomes.json') if r['sample_id'] == sid and r['strategy'] == 'zero_reference')
            assert outcome['first_success_audited']
            text = outcome['first_success_text']
            rows.append(dict(Model=model, Sample=sid, Original=originals[sid],
                             Modified_visible=text.replace('\u200b', '[U+200B]'),
                             Modified_escaped=text.encode('unicode_escape').decode(),
                             First_success_query=outcome['first_success_query'], Edits=outcome['first_success_edits'],
                             Human_semantic_review='pending; detector bypass only'))
        clean = {r['sample_id']: r for r in self.csv('stability', 'clean_controls.csv')}
        benign = next(r for r in self.csv('stability', 'attempts.csv') if r['dataset'] == 'benign' and r['family'] == 'zero_width' and r['new_error'] == '1' and clean[r['sample_id']]['original'].isascii())
        rows.append(dict(Model='piguard', Sample=benign['sample_id'], Original=clean[benign['sample_id']]['original'],
                         Modified_visible=benign['text'].replace('\u200b', '[U+200B]'),
                         Modified_escaped=benign['text'].encode('unicode_escape').decode(),
                         Effect='benign false block, not injection success'))
        return rows

    def export_tables(self):
        tables = dict(gradient=self.gradient_table(), combination=self.combination_table(), joint_cases=self.joint_cases(),
                      families=self.family_table(), selector=self.selector_table(), stability=self.stability_table(), examples=self.representative_examples())
        for name, rows in tables.items():
            (self.output / (name + '.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        return self.output


def show_table(rows):
    from IPython.display import display, HTML
    from html import escape
    keys = list(rows[0])
    def value(v):
        if isinstance(v, float):
            return f'{v:.2f}'
        return str(v)
    head = ''.join('<th>' + escape(k.replace('_', ' ')) + '</th>' for k in keys)
    body = ''.join('<tr>' + ''.join('<td style="text-align:left;white-space:normal">' + escape(value(r[k])) + '</td>' for k in keys) + '</tr>' for r in rows)
    display(HTML('<div style="overflow-x:auto"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'))


def prepare_native_environment(root):
    """Separate Python 3.10 environment; never change an old working runtime."""
    root = Path(root)
    has_gpu = bool(__import__('shutil').which('nvidia-smi')) and subprocess.run(['nvidia-smi', '-L'], capture_output=True).returncode == 0
    suffix = 'cu121' if has_gpu else 'cpu'
    env = root / ('native-env-' + suffix)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    subprocess.run(['uv', 'python', 'install', '3.10'], check=True)
    if not (env / 'bin/python').is_file():
        subprocess.run(['uv', 'venv', '--python', '3.10', str(env)], check=True)
    python = str(env / 'bin/python')
    index = 'https://download.pytorch.org/whl/' + suffix
    subprocess.run(['uv', 'pip', 'install', '--python', python, '--index-url', index, 'torch==2.4.0'], check=True)
    subprocess.run(['uv', 'pip', 'install', '--python', python, 'transformers==4.44.0', 'tokenizers==0.19.1', 'numpy==1.26.4', 'sentencepiece==0.2.0', 'protobuf==4.25.4', 'gdown==5.2.0', 'pyyaml==6.0.2'], check=True)
    probe = "import sys,torch; gpu=sys.argv[1]=='1'; assert not gpu or torch.cuda.is_available(); x=torch.tensor([1.],device='cuda' if gpu else 'cpu'); assert float(x.sum())==1.; print('Actual device:',x.device,'Torch:',torch.__version__)"
    subprocess.run([python, '-c', probe, '1' if has_gpu else '0'], check=True)
    checkpoint = root / 'checkpoints/best_model.pth'
    expected = 'ad7eed71bedf2d445a4d452d781bccab6b2101675c2b4f65edae86d444ca1c99'
    if not checkpoint.exists():
        checkpoint.parent.mkdir(exist_ok=True)
        subprocess.run([python, '-m', 'gdown', '1JpiVb_wtnbBLNEjIx1KS7PHuvmARQKTu', '-O', str(checkpoint)], check=True)
    assert sha(checkpoint.read_bytes()) == expected, 'Wrong checkpoint; no model run allowed'
    return python, checkpoint


def rerun_native(evidence, key, python, checkpoint):
    """Run a full original protocol. Never overwrite archived evidence."""
    import eval_m3_transfer as reader
    from datetime import datetime, timezone
    root = evidence.root
    source = root / 'src'
    output = root / 'fresh_runs' / key
    output.mkdir(parents=True, exist_ok=True)
    def run(script, args):
        command = [python, '-u', str(source / script), *map(str, args)]
        # Original legacy reports are retained on disk; the notebook uses English summaries.
        showing_report = False
        with subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'}) as process:
            for line in process.stdout:
                if line.startswith('# M3'):
                    showing_report = True
                if line.startswith('Complete:'):
                    showing_report = False
                if not showing_report:
                    print(line, end='', flush=True)
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    if key in ('gradient', 'combination', 'strategy'):
        script, old_key, member = {
            'gradient': ('eval_m3_gradient_text_full.py', 'gradient', 'text_plan.json'),
            'combination': ('eval_m3_blackbox_combo.py', 'combination', 'combo_plan.json'),
            'strategy': ('eval_m3_selector.py', 'selector', 'selector_plan.json')}[key]
        plan_path = output / member
        with zipfile.ZipFile(evidence.paths[old_key]) as z:
            raw = z.read(member)
        plan_path.write_bytes(raw)
        (output / 'preparation.json').write_text(json.dumps(dict(plan_sha256=sha(raw))))
        run(script, ['--plan', plan_path, '--repo', source, '--checkpoint', checkpoint, '--output-dir', output])
        prefix = {'gradient': 'gradient_text_', 'combination': 'blackbox_combo_', 'strategy': 'selector_'}[key]
        completed = [p for p in sorted(output.glob(prefix + '*')) if (p / 'manifest.json').exists() and json.loads((p / 'manifest.json').read_text()).get('complete') is True]
    elif key == 'stability':
        run('eval_m3_unicode_stability.py', ['--dataset-root', root / 'datasets', '--resume', checkpoint, '--output-dir', output])
        completed = [p for p in sorted(output.glob('stability_*')) if (p / 'manifest.json').exists() and json.loads((p / 'manifest.json').read_text()).get('complete') is True]
    else:
        raise ValueError('Choose gradient, combination, strategy or stability')
    assert completed, 'No complete new run was produced'
    import shutil
    archive = Path(shutil.make_archive(str(completed[-1]), 'zip', root_dir=completed[-1]))
    print('FRESH model run completed:', archive)
    if key == 'strategy':
        show_table([dict(Model=r['model'], Fixed_zero=r['zero_successes'], Selector=r['selector_successes'], Denominator=r['denominator'],
                         Zero_queries=r['zero_queries'], Selector_queries=r['selector_queries']) for r in read_json(archive, 'criteria.json')])
    elif key == 'stability':
        show_table([r for r in read_csv(archive, 'family_budget_summary.csv') if r['budget'] == 'all'])
    else:
        show_table(read_csv(archive, 'budget_summary.csv'))
    return archive
