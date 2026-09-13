"""Successive screening with a reserved uncertainty slot and branch rollouts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request
from harness import Codex, DESCRIPTION, ROOT, evaluate, save, summary, validate, BASELINE
from anchor_gate import check as anchor_check
from budget_allocator import allocate as allocate_budget

ROLES = ['linear efficiency', 'deceptive blocks', 'rugged robustness',
         'restart falsification', 'temperature experiment', 'mutation experiment']
CONSTELLATIONS = {
    '探索': ['linear efficiency', 'mutation experiment'],
    '頑健性': ['deceptive blocks', 'rugged robustness'],
    '評価': ['restart falsification', 'temperature experiment'],
}
OLLAMA_MODELS = ['smollm2:135m', 'smollm2:360m', 'zarigata/Qwen2.5-0.5B-Instruct:test', 'tinyllama:1.1b']

class OllamaClient:
    def __init__(self, directory): self.directory=directory
    def call(self, label, payload):
        # Qwen is the structured-output gate; tiny models can be added later as free-form scouts.
        model='zarigata/Qwen2.5-0.5B-Instruct:test'
        prompt=('Return ONLY one JSON object with keys policy,rationale,prediction. '
                'policy must contain exactly explore,flip,temperature,restart,guidance. '
                'No markdown, no extra text. Bounds: explore 0..1, flip .001.. .5, temperature 0.. .2, restart 5..120 integer.\n'+json.dumps(payload,ensure_ascii=False))
        req=urllib.request.Request('http://localhost:11434/api/generate',data=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json','options':{'temperature':0.2,'num_predict':256}}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=500) as res: raw=json.loads(res.read().decode())
        text=raw.get('response','').strip(); start=text.find('{'); end=text.rfind('}')
        if start<0 or end<=start:
            answer={'policy':dict(BASELINE),'rationale':'unparseable local output','prediction':'none','model':model}
            save(self.directory/f'{label}.ollama.json',{'model':model,'raw':raw,'parsed':answer})
            return answer
        try:
            answer=json.loads(text[start:end+1])
        except json.JSONDecodeError:
            answer={'policy':dict(BASELINE),'rationale':'malformed local output','prediction':'none'}
        def clean(a):
            p=a.get('policy',a)
            if not isinstance(p, dict):
                p=dict(BASELINE)
            # Tiny models commonly add commentary fields inside policy; discard only extras.
            p={k:p[k] for k in ('explore','flip','temperature','restart','guidance') if k in p}
            if 'restart' in p:
                p['restart']=int(float(p['restart']))
            return {'policy':p,'rationale':str(a.get('rationale','')), 'prediction':str(a.get('prediction',''))}
        answer=clean(answer)
        try:
            validate(answer['policy'])
        except Exception:
            # Tiny models often violate the schema; retry once with the strongest local model.
            model='zarigata/Qwen2.5-0.5B-Instruct:test'
            req=urllib.request.Request('http://localhost:11434/api/generate',data=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json','options':{'num_predict':256}}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=500) as res: raw=json.loads(res.read().decode())
            text=raw.get('response','').strip(); start=text.find('{'); end=text.rfind('}')
            answer=clean(json.loads(text[start:end+1])); validate(answer['policy'])
        answer['model']=model
        save(self.directory/f'{label}.ollama.json',{'model':model,'raw':raw,'parsed':answer})
        return answer


def difference(old, new):
    categories = {c: [b-a for a, b in zip(old[c], new[c])] for c in old}
    # Bootstrap whole seeds, retaining category correlation within each seed.
    paired = [statistics.mean(categories[c][i] for c in categories) for i in range(len(next(iter(old.values()))))]
    rng = random.Random(181)
    boot = sorted(statistics.mean(rng.choices(paired, k=len(paired))) for _ in range(1500))
    return {'delta': statistics.mean(paired), 'interval': [boot[37], boot[1462]],
            'categories': {c: statistics.mean(v) for c, v in categories.items()}}


def select(rows, count):
    """Exploit mean gains, but reserve one slot for the best optimistic bound."""
    ranked = sorted(rows, key=lambda r: (-r['difference']['delta'], r['id']))
    chosen = ranked[:count-1]
    remaining = [r for r in ranked if r not in chosen]
    if remaining:
        chosen.append(max(remaining, key=lambda r: (r['difference']['interval'][1], r['difference']['delta'])))
    return chosen


def key(policy):
    return tuple(policy[k] for k in ('explore', 'flip', 'temperature', 'restart'))


def passes(d):
    return d['delta'] > .002 and d['interval'][0] > 0 and min(d['categories'].values()) >= -.02


def run():
    directory = ROOT / 'runs' / ('staged-' + time.strftime('%Y%m%d-%H%M%S'))
    directory.mkdir(parents=True)
    client = OllamaClient(directory)
    source = ROOT / 'runs/20260911-170813/report.json'
    incumbent = json.loads(source.read_text(encoding='utf-8'))['final_policy']
    start = dict(incumbent)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'staged.py', ROOT/'harness.py')}
    save(directory/'config.json', {'backend': 'Ollama', 'models': OLLAMA_MODELS, 'start': start, 'source_hashes': hashes,
         'rounds': 2, 'screening_seeds_per_category': [8, 32, 80],
         'survivors': [3, 2, 2], 'max_llm_calls': 18, 'objective_calls_per_task': 120,
         'gate': 'delta > .002, clustered bootstrap lower > 0, category losses <= .02',
         'scope': 'One-step descendant comparison; no sustained RSI claim'})
    history = []
    objective_calls = 0
    llm_calls = 0

    def measure(policy, seeds):
        nonlocal objective_calls
        objective_calls += 3 * len(seeds) * 120
        return evaluate(policy, seeds)

    def parallel_calls(jobs):
        nonlocal llm_calls
        with ThreadPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(lambda job: client.call(*job), jobs))
        llm_calls += len(jobs)
        return results

    for generation in (1, 2):
        print(f'Round {generation}: generating six independent candidates', flush=True)
        base = 2_000_000 + generation * 100_000
        jobs = [(f'r{generation}-proposal-{i}', {'experiment': DESCRIPTION,
                 'current_policy': incumbent, 'inherited_guidance': incumbent['guidance'],
                 'history': history, 'role': role,
                 'constellation': next(c for c, roles in CONSTELLATIONS.items() if role in roles),
                 'task': 'Propose a distinct plausible improvement. Do not pre-reject uncertain ideas; experiments will screen them.'})
                for i, role in enumerate(ROLES)]
        answers = parallel_calls(jobs)
        unique, seen = [], set()
        for i, answer in enumerate(answers):
            if key(answer['policy']) in seen:
                continue
            seen.add(key(answer['policy']))
            role = ROLES[i]
            unique.append({'id': i, 'policy': answer['policy'], 'prediction': answer['prediction'],
                           'constellation': next(c for c, roles in CONSTELLATIONS.items() if role in roles)})
        stages = []
        candidates = unique
        for stage, (n, keep) in enumerate(((8, 3), (32, 2), (80, 2))):
            seeds = list(range(base + stage*1000, base + stage*1000 + n))
            old = measure(incumbent, seeds)
            rows = []
            for candidate in candidates:
                scores = measure(candidate['policy'], seeds)
                rows.append(dict(candidate, scores=scores, difference=difference(old, scores)))
            selected = select(rows, min(keep, len(rows)))
            stages.append({'seeds': seeds, 'baseline': old, 'rows': rows,
                           'selected': [r['id'] for r in selected]})
            stages[-1]['next_budget'] = allocate_budget([
                {'branch': r['id'], 'constellation': r.get('constellation'),
                 'difference': r['difference'], 'novelty': 1.0 if r['difference']['interval'][0] <= 0 else 0.2}
                for r in rows], budget=100)
            candidates = [{k: r[k] for k in ('id', 'policy', 'prediction', 'constellation')} for r in selected]
            print(f'Round {generation} stage {stage+1}: {len(rows)} -> {len(selected)}, seeds/category={n}', flush=True)
        # Every finalist and the unmodified incumbent get exactly one LLM proposal,
        # identical task description, evidence format, and simulation budget.
        branches = [{'id': 'control', 'policy': incumbent}] + candidates
        jobs = [(f'r{generation}-descendant-{b["id"]}', {'experiment': DESCRIPTION,
                 'current_policy': b['policy'], 'inherited_guidance': b['policy']['guidance'],
                 'observations': summary(measure(b['policy'], range(base+4000, base+4016))),
                 'task': 'Produce one next improvement using this policy and its guidance. Return a concrete candidate even if uncertain.'})
                for b in branches]
        print(f'Round {generation}: equal-call next-improvement test on {len(branches)} branches', flush=True)
        descendants = parallel_calls(jobs)
        # Select descendant vs parent on a separate development split.
        rollout = []
        for branch, child in zip(branches, descendants):
            seeds = list(range(base+5000, base+5032))
            parent_scores = measure(branch['policy'], seeds)
            child_scores = measure(child['policy'], seeds)
            d = difference(parent_scores, child_scores)
            anchor = anchor_check(child['policy'], d)
            chosen = child['policy'] if passes(d) and anchor['adoption_allowed'] else branch['policy']
            rollout.append({'id': branch['id'], 'parent': branch['policy'], 'child': child,
                            'development': {'seeds': seeds, 'parent': parent_scores, 'child': child_scores, 'difference': d},
                            'anchor_verification': anchor,
                            'policy': chosen})
        seeds = list(range(base+6000, base+6080))
        control = measure(rollout[0]['policy'], seeds)
        control_parent = measure(incumbent, seeds)
        control_gain = difference(control_parent, control)
        for branch in rollout:
            parent = measure(branch['parent'], seeds)
            final = measure(branch['policy'], seeds)
            branch['evaluation'] = {'parent': parent, 'final': final,
                                    'vs_control_endpoint': difference(control, final),
                                    'own_gain': difference(parent, final)}
            # Difference of paired improvement gains, not just endpoint quality.
            gains_control = {c: [b-a for a,b in zip(control_parent[c],control[c])] for c in control}
            gains_branch = {c: [b-a for a,b in zip(parent[c],final[c])] for c in parent}
            branch['evaluation']['gain_difference'] = difference(gains_control, gains_branch)
        winner = max(rollout, key=lambda b: statistics.mean(summary(b['evaluation']['final']).values()))
        # Single winner committed before fresh adoption tasks are accessed.
        final_seeds = list(range(base+7000, base+7100))
        before = measure(incumbent, final_seeds)
        after = measure(winner['policy'], final_seeds)
        adoption = difference(before, after)
        previous = incumbent
        final_anchor = anchor_check(winner['policy'], adoption)
        if passes(adoption) and final_anchor['adoption_allowed']:
            incumbent = winner['policy']
        record = {'round': generation, 'before': previous, 'unique_candidates': len(unique),
                  'stages': stages, 'rollout_seeds': seeds, 'rollout': rollout,
                  'winner': winner['id'], 'adoption_seeds': final_seeds,
                  'adoption_before': before, 'adoption_after': after, 'adoption': adoption,
                  'final_anchor_verification': final_anchor,
                  'accepted': passes(adoption) and final_anchor['adoption_allowed'], 'incumbent': incumbent}
        save(directory/f'round-{generation}.json', record)
        history.append({'round': generation, 'adoption': adoption,
                        'accepted': passes(adoption), 'incumbent': incumbent,
                        'candidate_differences': [{'id': r['id'], 'difference': r['difference']} for r in stages[-1]['rows']]})
        print(f'Round {generation}: adoption delta={adoption["delta"]:+.6f}, accepted={passes(adoption)}', flush=True)
    seeds = list(range(7_000_000, 7_000_200))
    old, new = measure(start, seeds), measure(incumbent, seeds)
    audit = difference(old, new)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h for name,h in hashes.items())
    report = {'start': start, 'final': incumbent, 'history': history, 'llm_calls': llm_calls,
              'constellations': CONSTELLATIONS,
              'objective_calls': objective_calls, 'audit': {'seeds': seeds, 'old': old, 'new': new, 'difference': audit},
              'limitations': ['Synthetic restricted policy search', 'Two rounds and one descendant step only',
                   'Equal LLM call counts, not equal realized token counts',
                   'Parent policy and guidance effects confounded',
                   'Bootstrap intervals descriptive; no sustained RSI demonstration']}
    save(directory/'report.json', report)
    rows = '\n'.join(f"| {h['round']} | {h['adoption']['delta']:+.6f} | {h['accepted']} |" for h in history)
    (directory/'結果.md').write_text('# 段階的な差分検証の結果\n\n'
        f'実LLM呼び出し: {llm_calls}回。目的関数評価: {objective_calls:,}回。\n\n'
        '| ラウンド | 採用検証の差 | 採用 |\n|---|---:|---|\n'+rows+
        f'\n\n最終未使用600課題で、前回の最終版からの平均差: **{audit["delta"]:+.6f}**。\n\n'
        '6提案を重複排除し、8→32→80課題/カテゴリーで段階評価。有望枠に加えて上側推定値の高い候補を1枠保存。'
        '最終候補と現行版に各1回、次の改善を生成させ、改善量の差も記録した。採用はさらに別の100課題/カテゴリーで判断。\n\n'
        '2ラウンド・子孫1段階の小規模実験。継続的なRSIは証明していない。'
        'LLM呼び出し回数は枝間で同じだが、実トークン数は同一ではない。探索設定と改善指示の寄与も分離していない。\n',encoding='utf-8')
    print(json.dumps({'directory': str(directory), 'calls': llm_calls, 'audit': audit},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    run()
