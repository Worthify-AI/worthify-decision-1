"""Check evidence recomputation and reject damaged or inconsistent records."""
import json
from pathlib import Path
import shutil

import pytest
from benchmarks import verify_decision_1 as evidence
from openjev_phase1.task_lora import read_rows, check_splits

ROOT = Path(__file__).resolve().parents[1]


def test_all_public_evidence_recomputes():
    assert evidence.main(ROOT)['full_weight']['family_headlines'] == 21


def test_macro_f1_includes_absent_offered_class():
    metrics = evidence.metrics(['a', 'a'], ['a', 'a'], ['a', 'b'])
    assert metrics['accuracy'] == 1
    assert metrics['macro_f1'] == .5


def test_unknown_label_rejected():
    with pytest.raises(ValueError,match='Unknown metric label'):
        evidence.metrics(['a'], ['missing'], ['a'])


def test_duplicate_predictions_rejected(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_text('{"id":"duplicate"}\n'*2)
    with pytest.raises(ValueError,match='Duplicate row IDs'):
        evidence.rows(path)


def test_tampered_curve_summary_rejected():
    report=evidence.load(ROOT/'results/worthify/foundation-banking77-transfer-lora-20260926-v1/report.json')
    curve=report['validation_curves']['raw-42']
    curve['absolute_accuracy_auc'] += .01
    with pytest.raises(ValueError,match='absolute_accuracy_auc'):
        evidence.verify_curve(curve,'accuracy')


def test_consistently_rehashed_false_headline_rejected(tmp_path):
    name='foundation-transfer-lora-20260926-v1'
    package=tmp_path/'results/worthify'/name
    shutil.copytree(ROOT/'results/worthify'/name, package)
    raw=tmp_path/'results/raw';raw.mkdir(parents=True)
    aggregate=evidence.load(ROOT/'results/raw'/(name+'.json'))
    report=evidence.load(package/'report.json')
    report['runs']['raw-42']['metrics']['accuracy'] += .1
    (package/'report.json').write_text(json.dumps(report))
    aggregate['report_sha256']=evidence.sha(package/'report.json')
    (raw/(name+'.json')).write_text(json.dumps(aggregate))
    (package/'SHA256SUMS').write_text(''.join(evidence.sha(package/f)+'  '+f+'\n' for f in ['paired-rows.jsonl','report.json']))
    with pytest.raises(ValueError,match='raw-42/accuracy'):
        evidence.verify_transfer(tmp_path,name,False)


def test_example_splits_are_valid_disjoint_and_authored():
    data={split:read_rows(ROOT/f'examples/support/{split}.jsonl',gold=True)
          for split in ('train','validation','test')}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check_splits(data[a],data[b])
    assert {r['gold_option_id'] for r in data['train']} == {'payments','account'}


def test_snapshot_rejects_uninventoried_release_file(tmp_path):
    from scripts.verify_snapshot import verify
    fixture={'source_revision':'a'*40,'snapshot_files_sha256':{},'files':{}}
    (tmp_path/'SOURCE_PROVENANCE.json').write_text(json.dumps(fixture))
    verify(tmp_path)
    (tmp_path/'unreviewed.txt').write_text('This must not ship unnoticed.')
    with pytest.raises(ValueError,match='inventory differs'):
        verify(tmp_path)


def test_snapshot_ignores_local_environment(tmp_path):
    from scripts.verify_snapshot import verify
    fixture={'source_revision':'a'*40,'snapshot_files_sha256':{},'files':{}}
    (tmp_path/'SOURCE_PROVENANCE.json').write_text(json.dumps(fixture))
    (tmp_path/'.venv').mkdir()
    (tmp_path/'.venv/local.txt').write_text('local environment')
    verify(tmp_path)


def test_tampered_late_phase_area_rejected():
    curve = {'grid_steps': [0, 194, 388], 'grid_accuracy': [.8, .9, 1.0],
             'absolute_accuracy_auc': .9, 'baseline_adjusted_accuracy_auc': .1,
             'step_zero_accuracy': .8, 'fixed_budget_accuracy': 1.0,
             'first_grid_step_at_0_80_accuracy': 0,
             'late_phase_accuracy_auc_194_to_388': .85}
    with pytest.raises(ValueError, match='late-phase accuracy AUC'):
        evidence.verify_curve(curve, 'accuracy')


def test_extended_selection_rejects_test_favored_checkpoint():
    report=evidence.load(ROOT/'results/worthify/foundation-banking77-extended-lora-20260927-v1/report.json')
    report['runs']['raw-42']['selected_step'] = 388
    with pytest.raises(ValueError, match='validation-selected checkpoint'):
        evidence.verify_extended_selection(report)


def test_extended_selection_rejects_test_favored_seed():
    report=evidence.load(ROOT/'results/worthify/foundation-banking77-extended-lora-20260927-v1/report.json')
    report['selected_seed_by_arm']['raw'] = 43
    with pytest.raises(ValueError, match='validation-selected seed'):
        evidence.verify_extended_selection(report)


def test_rehashed_inaccurate_extended_chart_rejected(tmp_path):
    name='foundation-banking77-extended-lora-20260927-v1'
    package=tmp_path/'results/worthify'/name
    shutil.copytree(ROOT/'results/worthify'/name, package)
    raw=tmp_path/'results/raw'; raw.mkdir(parents=True)
    shutil.copyfile(ROOT/'results/raw'/(name+'.json'),raw/(name+'.json'))
    stem='banking77-validation-0-388-20260927-v1'
    assets=tmp_path/'docs/assets/decision-1'; assets.mkdir(parents=True)
    for suffix in ('svg','png','json'):
        shutil.copyfile(ROOT/'docs/assets/decision-1'/(stem+'.'+suffix),assets/(stem+'.'+suffix))
    svg=assets/(stem+'.svg')
    text=svg.read_text().replace('polyline points="130.0,','polyline points="131.0,',1)
    svg.write_text(text)
    manifest=assets/(stem+'.json'); metadata=evidence.load(manifest)
    metadata['svg_sha256']=evidence.sha(svg); manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='chart validation polylines'):
        evidence.verify_extended(tmp_path)
