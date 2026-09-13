"""Independently replay saved comparisons and verify policy lineage (no LLM calls)."""
import hashlib
import json
from pathlib import Path
import sys
from harness import BASELINE, ROOT, compare, evaluate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(directory):
    report = read(directory / 'report.json')
    assert report['source_sha256'] == hashlib.sha256((ROOT / 'harness.py').read_bytes()).hexdigest()
    previous = dict(BASELINE)
    for generation in range(1, report['generations'] + 1):
        record = read(directory / f'generation-{generation}.json')
        assert record['before'] == previous
        seeds = range(10000 + generation * 1000, 10000 + generation * 1000 + 40)
        raw = read(directory / f'generation-{generation}-raw-scores.json')
        assert raw['old'] == evaluate(previous, seeds)
        assert raw['new'] == evaluate(record['candidate']['policy'], seeds)
        assert compare(raw['old'], raw['new']) == record['comparison']
        previous = record['candidate']['policy'] if record['comparison']['accepted'] else previous
        assert record['adopted_policy'] == previous
        for role in ('efficiency', 'robustness', 'falsification', 'synthesis'):
            request = read(directory / f'g{generation}-{role}.request.json')
            assert request['current_policy'] == record['before']
            assert request['inherited_proposal_guidance'] == record['before']['guidance']
    assert previous == report['final_policy']
    final_raw = read(directory / 'audit-raw-scores.json')
    assert final_raw['baseline'] == evaluate(BASELINE, range(900000, 900100))
    assert final_raw['final'] == evaluate(previous, range(900000, 900100))
    assert compare(final_raw['baseline'], final_raw['final']) == report['final_unseen_audit']
    calls, usage = 0, {}
    for path in directory.glob('*.events.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn.completed':
                calls += 1
                for key, value in event.get('usage', {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
    assert calls == report['generations'] * 4
    result = {'verified': True, 'real_llm_completed_calls': calls, 'usage': usage,
              'checks': ['source hash', 'exact simulation replay', 'adoption and rollback',
                         'next-generation policy and guidance inheritance', 'final untouched audit replay']}
    (directory / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]).resolve())
