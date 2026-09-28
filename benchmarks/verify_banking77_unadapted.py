"""Recompute public BANKING77 unadapted-to-LoRA results from text-free rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


SCHEMA = 'openjev-banking77-unadapted-transfer-v1'
SELECTOR_SHA256 = 'bd0d8abc79bc15c9be7e4fd8a1a0a07b458fe7d1a2d57e71c643928b4af602c5'
DATA_SHA256 = 'b9e434686de8ad0670bb3ca5223df78cee39af9697d09e1b068ed3ae114f714b'
TEST_SHA256 = 'c8ad8834a98d817198459404f874a71b9ec8848b94217448fc3a045380fe1686'
PRIOR_REPORT_SHA256 = '3a636551d84ad308ac07eb1366e9b76a4b746b7e14de60e2cb02b5ba34dd596c'
PRIOR_ROWS_SHA256 = '88c71f6ff9f3d998a53b5d9c9d24cc8a1aff927530437b7d52d2e085321fdc1a'


def _paired(gold: list[str], base: list[str], lora: list[str]) -> dict:
    return {'both_correct': sum(b == l == g for g,b,l in zip(gold,base,lora,strict=True)),
            'base_only_correct': sum(b == g and l != g for g,b,l in zip(gold,base,lora,strict=True)),
            'lora_only_correct': sum(b != g and l == g for g,b,l in zip(gold,base,lora,strict=True)),
            'both_wrong': sum(b != g and l != g for g,b,l in zip(gold,base,lora,strict=True))}


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def _intent_bootstrap(gold: list[str], base: list[str], lora: list[str],
                      labels: tuple[str, ...]) -> dict:
    groups = {label: [i for i,value in enumerate(gold) if value == label] for label in labels}
    if len(groups) != 77 or any(not indices for indices in groups.values()):
        raise ValueError('Official test must cover all 77 original intents')
    totals = [(len(indices), sum((lora[i] == gold[i])-(base[i] == gold[i])
                                 for i in indices)) for indices in groups.values()]
    randomizer = random.Random(20260926)
    samples = []
    for _ in range(10_000):
        drawn = [totals[randomizer.randrange(len(totals))] for _ in totals]
        samples.append(sum(delta for _,delta in drawn)/sum(size for size,_ in drawn))
    samples.sort()
    return {'unit':'original intent cluster', 'groups':77, 'replicates':10_000,
            'seed':20260926, 'percentile_2_5':_percentile(samples,.025),
            'percentile_97_5':_percentile(samples,.975)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory: Path, prior_directory: Path | None = None) -> dict:
    report_path, rows_path = directory/'report.json', directory/'paired-rows.jsonl'
    expected = f'{_sha(report_path)}  report.json\n{_sha(rows_path)}  paired-rows.jsonl\n'
    if (directory/'SHA256SUMS').read_text() != expected:
        raise ValueError('Evidence SHA256SUMS mismatch')
    report = json.loads(report_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    labels = report['labels']
    if (report['schema'] != SCHEMA or report['test_rows'] != 3080
            or report['source_hashes']['selector_sha256'] != SELECTOR_SHA256
            or report['source_hashes']['data_manifest_sha256'] != DATA_SHA256
            or report['source_hashes']['test_sha256'] != TEST_SHA256
            or report['source_hashes']['prior_report_sha256'] != PRIOR_REPORT_SHA256
            or report['source_hashes']['prior_paired_rows_sha256'] != PRIOR_ROWS_SHA256
            or report['numeric_policy'] != 'fp32-final-softcap-cap30'
            or report['quantization'] != 'nf4' or report['dtype'] != 'bfloat16'
            or report['max_tokens'] != 2048 or report['prompt_version'] != 'direct-options-v1'
            or len(labels) != 77 or len(set(labels)) != 77 or len(rows) != 3080
            or len({row['id'] for row in rows}) != 3080
            or Counter(row['gold_option_id'] for row in rows) != {label:40 for label in labels}
            or sum(row['v9_exposure_flag'] for row in rows) != 28):
        raise ValueError('Unexpected official test evidence')
    for row in rows:
        if (len(row['option_ids']) != 16 or len(set(row['option_ids'])) != 16
                or row['gold_option_id'] not in row['option_ids']
                or not set(row['option_ids']) <= set(labels)
                or len(row['prompt_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['prompt_sha256'])
                or set(row['zero_shot_predicted_option_id']) != {'raw','tuned'}
                or set(row['lora_predicted_option_id']) != {'raw','tuned'}
                or set(row['lora_all_seed_predicted_option_id']) !=
                   {'raw-42','raw-43','tuned-42','tuned-43'}):
            raise ValueError(f"Invalid paired row: {row['id']}")
        if any(value not in row['option_ids'] for value in
               list(row['zero_shot_predicted_option_id'].values())+
               list(row['lora_predicted_option_id'].values())+
               list(row['lora_all_seed_predicted_option_id'].values())):
            raise ValueError(f"Prediction outside row options: {row['id']}")
    if prior_directory is None:
        prior_directory = (Path(__file__).resolve().parents[1] / 'results/worthify/'
                           'foundation-banking77-extended-lora-20260927-v1')
    old_report, old_rows = prior_directory/'report.json', prior_directory/'paired-rows.jsonl'
    if _sha(old_rows) != PRIOR_ROWS_SHA256 or (prior_directory/'SHA256SUMS').read_text() != (
            f'{_sha(old_report)}  report.json\n{_sha(old_rows)}  paired-rows.jsonl\n'):
        raise ValueError('Prior public LoRA evidence changed')
    previous = [json.loads(line) for line in old_rows.read_text().splitlines()]
    if len(previous) != len(rows):
        raise ValueError('Prior public LoRA row count changed')
    for current, old in zip(rows, previous, strict=True):
        if (current['id'] != old['id'] or
                current['gold_option_id'] != old['gold_option_id'] or
                current['v9_exposure_flag'] != old['v9_exposure_flag'] or
                current['lora_all_seed_predicted_option_id'] != old['predicted_option_id']):
            raise ValueError(f"Prior public LoRA row mismatch: {current['id']}")
    gold = [row['gold_option_id'] for row in rows]
    for arm in ('raw','tuned'):
        actual = report['arms'][arm]
        base = [row['zero_shot_predicted_option_id'][arm] for row in rows]
        lora = [row['lora_predicted_option_id'][arm] for row in rows]
        base_correct = sum(a==b for a,b in zip(base,gold,strict=True))
        lora_correct = sum(a==b for a,b in zip(lora,gold,strict=True))
        retained = [i for i,row in enumerate(rows) if not row['v9_exposure_flag']]
        if len(retained) != 3052:
            raise ValueError('Exposure sensitivity count changed')
        clean_base = sum(base[i]==gold[i] for i in retained)
        clean_lora = sum(lora[i]==gold[i] for i in retained)
        checks = {'zero_shot_correct': base_correct, 'zero_shot_accuracy': base_correct/3080,
                  'lora_correct': lora_correct, 'lora_accuracy': lora_correct/3080,
                  'lift': (lora_correct-base_correct)/3080,
                  'paired_correctness': _paired(gold,base,lora),
                  'intent_bootstrap_95pct': _intent_bootstrap(gold,base,lora,tuple(labels)),
                  'exposure_excluded_rows': len(retained),
                  'exposure_excluded_zero_shot_correct': clean_base,
                  'exposure_excluded_zero_shot_accuracy': clean_base/len(retained),
                  'exposure_excluded_lora_correct': clean_lora,
                  'exposure_excluded_lora_accuracy': clean_lora/len(retained),
                  'exposure_excluded_lift': (clean_lora-clean_base)/len(retained)}
        for key,value in checks.items():
            if actual[key] != value:
                raise ValueError(f'Report {arm}.{key} differs from rows')
        name = actual['selected_lora_name']
        if name != f'{arm}-42' or actual['selected_lora_step'] != {'raw':350,'tuned':325}[arm]:
            raise ValueError(f'Unexpected selected {arm} LoRA')
        if any(row['lora_predicted_option_id'][arm] !=
               row['lora_all_seed_predicted_option_id'][name] for row in rows):
            raise ValueError(f'Selected {arm} LoRA differs from stored seed row')
        if not math.isfinite(actual['lift']):
            raise ValueError('Nonfinite lift')
    return {'schema': SCHEMA + '-verification', 'rows': len(rows),
            'report_sha256': _sha(report_path), 'paired_rows_sha256': _sha(rows_path),
            'arms': {arm: {'zero_shot_correct': report['arms'][arm]['zero_shot_correct'],
                           'lora_correct': report['arms'][arm]['lora_correct']}
                     for arm in ('raw','tuned')}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--prior-evidence', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence, args.prior_evidence), sort_keys=True))


if __name__ == '__main__':
    main()
