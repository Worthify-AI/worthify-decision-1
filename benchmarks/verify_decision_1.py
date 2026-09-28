"""Recompute Decision-1 headline evidence using only public text-free records.

No model loading, network requests, source datasets or private research imports.
Runtime and full-weight uncertainty remain hashed observations, not recomputed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('raw-42', 'raw-43', 'tuned-42', 'tuned-43')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def rows(path):
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(result), f'Empty rows: {path}')
    require(len({r['id'] for r in result}) == len(result), f'Duplicate row IDs: {path}')
    return result


def equal(actual, expected, label='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'{label}: keys differ')
        for key in expected:
            equal(actual[key], expected[key], f'{label}/{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: lengths differ')
        for i, (a, e) in enumerate(zip(actual, expected)):
            equal(a, e, f'{label}/{i}')
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), f'{label}: {actual} != {expected}')
    else:
        require(actual == expected, f'{label}: differs')


def checksums(folder):
    names = set()
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(name not in names and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Duplicate or unsafe checksum path')
        names.add(name)
        require(sha(folder / name) == digest, f'Checksum mismatch: {folder / name}')
    return names


def metrics(gold, predictions, labels):
    require(len(gold) == len(predictions) and bool(gold), 'Invalid metric rows')
    require(set(gold) <= set(labels) and set(predictions) <= set(labels), 'Unknown metric label')
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, predictions):
        confusion[g][p] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {'rows': len(gold), 'accuracy': sum(a == b for a, b in zip(gold, predictions)) / len(gold),
            'macro_f1': sum(c['f1'] for c in per_class.values()) / len(labels),
            'per_class': per_class, 'confusion_gold_rows_predicted_columns': confusion}


def paired(data, raw, tuned):
    result = dict(both_correct=0, both_wrong=0, raw_only_correct=0, tuned_only_correct=0)
    for row in data:
        a = row['predicted_option_id'][raw] == row['gold_option_id']
        b = row['predicted_option_id'][tuned] == row['gold_option_id']
        result['both_correct' if a and b else 'raw_only_correct' if a else 'tuned_only_correct' if b else 'both_wrong'] += 1
    return result


def verify_curve(curve, metric):
    steps = curve['grid_steps']; values = curve['grid_' + metric]
    require(steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] > 0 and
            len(steps) == len(values), 'Invalid validation grid')
    for name, sequence in curve.items():
        if name.startswith('grid_') and name != 'grid_steps':
            require(len(sequence) == len(steps) and all(math.isfinite(x) for x in sequence), 'Invalid curve values')
    require(all(0 <= x <= 1 for x in values), 'Invalid curve metric')
    area = sum((a+b)*(t-s)/2 for s,t,a,b in zip(steps, steps[1:], values, values[1:])) / steps[-1]
    for key, value in {'absolute_'+metric+'_auc': area, 'baseline_adjusted_'+metric+'_auc': area-values[0],
                       'step_zero_'+metric: values[0], 'fixed_budget_'+metric: values[-1]}.items():
        equal(curve[key], value, key)
    threshold, label = (.8, '0_80') if metric == 'accuracy' else (.6, '0_60')
    equal(curve['first_grid_step_at_'+label+'_'+metric],
          next((s for s,v in zip(steps, values) if v >= threshold), None), 'first threshold')
    if 'late_phase_accuracy_auc_194_to_388' in curve:
        require(metric == 'accuracy' and 194 in steps and steps[-1] == 388,
                'Invalid extended curve horizon')
        start = steps.index(194)
        late = sum((a+b)*(t-s)/2 for s,t,a,b in
                   zip(steps[start:], steps[start+1:], values[start:], values[start+1:])) / 194
        equal(curve['late_phase_accuracy_auc_194_to_388'], late, 'late-phase accuracy AUC')


def verify_extended_selection(report):
    """Recompute the frozen four-epoch grid and validation-only selection."""
    grid = [0] + [step for step in range(1, 389) if step % 25 == 0 or step % 97 == 0]
    rankings = {}
    for arm in NAMES:
        curve = report['validation_curves'][arm]
        equal(curve['grid_steps'], grid, 'extended fixed grid')
        verify_curve(curve, 'accuracy')
        require('late_phase_accuracy_auc_194_to_388' in curve, 'Missing late-phase AUC')
        points = {step: (curve['grid_accuracy'][i], curve['grid_macro_f1'][i],
                         -curve['grid_cross_entropy'][i]) for i,step in enumerate(grid)}
        best = max((step for step in grid if step >= 97), key=lambda step: (*points[step], -step))
        equal(report['runs'][arm]['selected_step'], best, 'validation-selected checkpoint')
        rankings[arm] = points[best]
    expected = {arm: max((42,43), key=lambda seed: (*rankings[f'{arm}-{seed}'], -seed))
                for arm in ('raw','tuned')}
    equal(report['selected_seed_by_arm'], expected, 'validation-selected seed')
    equal(report['faster_learning_gate'], all(
        report['validation_curves'][f'tuned-{seed}']['baseline_adjusted_accuracy_auc'] >
        report['validation_curves'][f'raw-{seed}']['baseline_adjusted_accuracy_auc']
        for seed in (42,43)), 'historical baseline-adjusted gate')


def verify_extended(root):
    name = 'foundation-banking77-extended-lora-20260927-v1'
    result = verify_transfer(root, name, True)
    report_path = root/'results/worthify'/name/'report.json'
    report = load(report_path)
    require(report['schema'] == 'openjev-banking77-extended-evidence-v1', 'Wrong extended schema')
    verify_extended_selection(report)
    sensitivity = report['sensitivity_excluding_v9_exposure']
    improvement = (report['selected_test_accuracy_delta_tuned_minus_raw'] > 0
        and all(v > 0 for v in report['per_seed_test_accuracy_delta_tuned_minus_raw'].values())
        and report['selected_pair_intent_bootstrap']['percentile_2_5'] > 0
        and sensitivity['selected_pair_accuracy_delta_tuned_minus_raw'] > 0
        and all(v > 0 for v in sensitivity['per_seed_accuracy_delta_tuned_minus_raw'].values()))
    equal(report['task_specific_improvement_gate'], improvement, 'task improvement gate')
    stem = root/'docs/assets/decision-1/banking77-validation-0-388-20260927-v1'
    manifest = load(stem.with_suffix('.json'))
    equal(manifest['public_report_sha256'], sha(report_path), 'chart report binding')
    for kind in ('selector', 'protocol'):
        equal(manifest[kind+'_sha256'], report['source_hashes'][kind+'_sha256'], 'chart '+kind)
    for suffix in ('png', 'svg'):
        equal(manifest[suffix+'_sha256'], sha(stem.with_suffix('.'+suffix)), 'chart '+suffix)
    equal(manifest['fixed_grid_end_update'], 388, 'chart horizon')
    equal(manifest['validation_rows'], 365, 'chart validation rows')
    equal(manifest['adapter_seeds'], [42,43], 'chart seeds')
    # Recompute every displayed polyline from public curves, independently of the image renderer.
    curves = report['validation_curves']
    minimum = math.floor((min(min(c['grid_accuracy']) for c in curves.values())-.01)*100)/100
    maximum = math.ceil((max(max(c['grid_accuracy']) for c in curves.values())+.01)*100)/100
    expected = []
    for arm in ('raw','tuned'):
        values = [curves[f'{arm}-{seed}']['grid_accuracy'] for seed in (42,43)]
        values.append([(a+b)/2 for a,b in zip(*values)])
        for series in values:
            expected.append(' '.join(f'{130+1320*step/388:.1f},{165+550*(maximum-v)/(maximum-minimum):.1f}'
                for step,v in zip(curves[f'{arm}-42']['grid_steps'],series)))
    svg = ET.fromstring(stem.with_suffix('.svg').read_text())
    actual = [line.attrib['points'] for line in svg.findall('{http://www.w3.org/2000/svg}polyline')]
    equal(actual, expected, 'chart validation polylines')
    return {**result, 'validation_updates': 388, 'chart_curves': len(actual)}


def bootstrap(data, raw, tuned, labels):
    counts = Counter(r['gold_option_id'] for r in data)
    deltas = Counter()
    for row in data:
        g = row['gold_option_id']
        deltas[g] += int(row['predicted_option_id'][tuned] == g) - int(row['predicted_option_id'][raw] == g)
    totals = [(counts[label], deltas[label]) for label in labels]
    rng = random.Random(20260926); samples = []
    for _ in range(10000):
        draw = [totals[rng.randrange(len(totals))] for _ in totals]
        samples.append(sum(d for _, d in draw) / sum(n for n, _ in draw))
    samples.sort()
    def percentile(fraction):
        i = (len(samples)-1) * fraction
        return samples[math.floor(i)] + (samples[math.ceil(i)]-samples[math.floor(i)]) * (i-math.floor(i))
    return dict(unit='original intent cluster', groups=len(labels), replicates=10000, seed=20260926,
                percentile_2_5=percentile(.025), percentile_97_5=percentile(.975))


def verify_transfer(root, name, bank):
    directory = root / 'results/worthify' / name
    require(checksums(directory) == {'report.json', 'paired-rows.jsonl'}, 'Unexpected transfer inventory')
    report = load(directory / 'report.json'); data = rows(directory / 'paired-rows.jsonl')
    aggregate = load(root / 'results/raw' / (name+'.json'))
    equal(aggregate['report_sha256'], sha(directory/'report.json'), 'report hash')
    equal(aggregate['paired_rows_sha256'], sha(directory/'paired-rows.jsonl'), 'rows hash')
    for key in (aggregate.keys() & report.keys()) - {'schema'}:
        equal(aggregate[key], report[key], key)
    require(set(report['runs']) == set(NAMES), 'Expected four matched runs')
    labels = report['intent_labels'] if bank else ['botnet', 'normal']
    metric = 'accuracy' if bank else 'macro_f1'
    allowed = {'id', 'gold_option_id', 'predicted_option_id', 'correct'} | ({'v9_exposure_flag'} if bank else set())
    for row in data:
        require(set(row) == allowed, 'Unexpected text-free evidence fields')
        require(set(row['predicted_option_id']) == set(NAMES) and set(row['correct']) == set(NAMES), 'Missing prediction arm')
        for arm in NAMES:
            require(row['correct'][arm] is (row['predicted_option_id'][arm] == row['gold_option_id']), 'Incorrect correctness flag')
    equal(report['test_rows'], len(data), 'test rows')
    measured = {}
    for arm in NAMES:
        m = metrics([r['gold_option_id'] for r in data], [r['predicted_option_id'][arm] for r in data], labels)
        if bank:
            m['per_intent_accuracy'] = {k: v['recall'] for k,v in m['per_class'].items()}
        else:
            m.update(abstentions=0, abstention_rate=0.0)
        equal(report['runs'][arm]['metrics'], m, arm)
        measured[arm] = m
        verify_curve(report['validation_curves'][arm], metric)
    require(len({tuple(c['grid_steps']) for c in report['validation_curves'].values()}) == 1, 'Unequal update grids')
    selected = {arm: arm+'-'+str(report['selected_seed_by_arm'][arm]) for arm in ('raw','tuned')}
    equal(report['selected_names'], selected, 'selected names')
    raw, tuned = selected['raw'], selected['tuned']
    equal(report['selected_test_'+metric+'_delta_tuned_minus_raw'], measured[tuned][metric]-measured[raw][metric], 'selected delta')
    equal(report['selected_pair_correctness'], paired(data, raw, tuned), 'selected paired outcomes')
    for seed in (42,43):
        equal(report['per_seed_test_'+metric+'_delta_tuned_minus_raw'][str(seed)],
              measured[f'tuned-{seed}'][metric]-measured[f'raw-{seed}'][metric], 'seed delta')
        equal(report['per_seed_correctness'][str(seed)], paired(data, f'raw-{seed}', f'tuned-{seed}'), 'seed paired outcomes')
    if bank:
        equal(report['selected_pair_intent_bootstrap'], bootstrap(data, raw, tuned, labels), 'intent bootstrap')
        clean = [r for r in data if r['v9_exposure_flag'] is False]
        require(len(data)-len(clean) == 28, 'Exposure exclusions differ')
        sensitivity = report['sensitivity_excluding_v9_exposure']
        accuracy = {arm: sum(r['correct'][arm] for r in clean)/len(clean) for arm in NAMES}
        equal(sensitivity['rows'], len(clean), 'sensitivity rows')
        equal(sensitivity['run_accuracy'], accuracy, 'sensitivity accuracy')
        equal(sensitivity['selected_pair_accuracy_delta_tuned_minus_raw'], accuracy[tuned]-accuracy[raw], 'sensitivity delta')
        equal(sensitivity['selected_pair_correctness'], paired(clean, raw, tuned), 'sensitivity pair')
        for seed in (42,43):
            equal(sensitivity['per_seed_accuracy_delta_tuned_minus_raw'][str(seed)],accuracy[f'tuned-{seed}']-accuracy[f'raw-{seed}'], 'sensitivity seed delta')
            equal(sensitivity['per_seed_correctness'][str(seed)],paired(clean,f'raw-{seed}',f'tuned-{seed}'), 'sensitivity seed pair')
    return {'rows': len(data), 'selected_raw': measured[raw][metric], 'selected_tuned': measured[tuned][metric]}


def verify_full_weight(root):
    directory = root/'results/worthify/full-weight-seed42-heldout-20260924'
    provenance = load(directory/'provenance.json')
    for name, digest in provenance['files_sha256'].items():
        require(sha(directory/name) == digest, f'Full-weight hash mismatch: {name}')
    report = load(root/'results/raw/full-weight-seed42-heldout-20260924.json')
    require(report['scope'] == 'provisional_single_seed_heldout' and report['two_seed_selection_complete'] is False,
            'Single-seed scope changed')
    identities = None; checked = 0
    for arm in ('frozen','prior_lora','full_provisional'):
        data = rows(directory/arm/'rows.jsonl')
        equal(report['source_rows_sha256'][arm], sha(directory/arm/'rows.jsonl'), 'heldout row hash')
        current = [(r['id'],r['family'],r['gold'],r['option_ids']) for r in data]
        if identities is None: identities = current
        else: equal(current, identities, 'matched heldout rows')
        require(len(data) == 3200, 'Heldout row count changed')
        by_family = defaultdict(list)
        for row in data:
            require('state' not in row and 'question' not in row and 'prompt' not in row, 'Unexpected source text')
            require(row['prediction'] == row['option_ids'][max(range(len(row['option_ids'])),key=lambda i: row['option_logits'][i])], 'Prediction differs from logits')
            by_family[row['family']].append(row)
        equal(sorted(by_family), sorted(report['arms'][arm]['family_metrics']), 'family coverage')
        for family, subset in by_family.items():
            recorded = report['arms'][arm]['family_metrics'][family]
            labels = sorted({label for r in subset for label in r['option_ids']})
            m = metrics([r['gold'] for r in subset], [r['prediction'] for r in subset], labels)
            for key in ('rows','accuracy','macro_f1'):
                equal(recorded[key], m[key], f'{arm}/{family}/{key}')
            equal(recorded['score'], m[recorded['metric']], f'{arm}/{family}/primary score')
            if recorded.get('macro_f1_class_set') is not None:
                equal(recorded['macro_f1_class_set'], labels, 'macro-F1 denominator')
            checked += 1
    return {'rows_per_arm':3200,'family_headlines':checked}


def main(root=ROOT):
    from benchmarks.verify_banking77_unadapted import verify as verify_unadapted

    checksums(root/'results/raw')
    result = {'status':'ok',
        'banking77':verify_transfer(root,'foundation-banking77-transfer-lora-20260926-v1',True),
        'banking77_extended':verify_extended(root),
        'banking77_before_after':verify_unadapted(
            root/'results/worthify/foundation-banking77-unadapted-transfer-20260927-v1',
            root/'results/worthify/foundation-banking77-extended-lora-20260927-v1'),
        'ctu13':verify_transfer(root,'foundation-transfer-lora-20260926-v1',False),
        'full_weight':verify_full_weight(root)}
    print(json.dumps(result,sort_keys=True))
    return result


if __name__ == '__main__':
    main()
