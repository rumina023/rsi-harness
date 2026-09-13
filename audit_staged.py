import json
import hashlib
from pathlib import Path
import sys
from harness import ROOT, evaluate
from staged import difference, passes, select

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def audit(directory):
    config=read(directory/'config.json')
    report=read(directory/'report.json')
    for name,h in config['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    incumbent=config['start']
    used=set()
    for generation in (1,2):
        r=read(directory/f'round-{generation}.json')
        assert r['before']==incumbent
        for stage in r['stages']:
            assert not used.intersection(stage['seeds'])
            used.update(stage['seeds'])
            assert stage['baseline']==evaluate(incumbent,stage['seeds'])
            for row in stage['rows']:
                assert row['scores']==evaluate(row['policy'],stage['seeds'])
                assert row['difference']==difference(stage['baseline'],row['scores'])
            assert stage['selected']==[x['id'] for x in select(stage['rows'],len(stage['selected']))]
        for b in r['rollout']:
            d=b['development']
            assert d['parent']==evaluate(b['parent'],d['seeds'])
            assert d['child']==evaluate(b['child']['policy'],d['seeds'])
            assert d['difference']==difference(d['parent'],d['child'])
            assert b['policy']==(b['child']['policy'] if passes(d['difference']) else b['parent'])
            assert b['evaluation']['parent']==evaluate(b['parent'],r['rollout_seeds'])
            assert b['evaluation']['final']==evaluate(b['policy'],r['rollout_seeds'])
        control=r['rollout'][0]['evaluation']
        for b in r['rollout']:
            e=b['evaluation']
            assert e['own_gain']==difference(e['parent'],e['final'])
            assert e['vs_control_endpoint']==difference(control['final'],e['final'])
            cg={c:[y-x for x,y in zip(control['parent'][c],control['final'][c])] for c in control['parent']}
            bg={c:[y-x for x,y in zip(e['parent'][c],e['final'][c])] for c in e['parent']}
            assert e['gain_difference']==difference(cg,bg)
        winner=next(b for b in r['rollout'] if b['id']==r['winner'])
        assert r['adoption_before']==evaluate(incumbent,r['adoption_seeds'])
        assert r['adoption_after']==evaluate(winner['policy'],r['adoption_seeds'])
        assert r['adoption']==difference(r['adoption_before'],r['adoption_after'])
        incumbent=winner['policy'] if passes(r['adoption']) else incumbent
        assert incumbent==r['incumbent']
    final=report['audit']
    assert final['old']==evaluate(config['start'],final['seeds'])
    assert final['new']==evaluate(incumbent,final['seeds'])
    assert final['difference']==difference(final['old'],final['new'])
    calls=0
    usage={}
    for p in directory.glob('*.events.jsonl'):
        for line in p.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e.get('type')=='turn.completed':
                calls+=1
                for k,v in e.get('usage',{}).items():
                    if isinstance(v,(int,float)):
                        usage[k]=usage.get(k,0)+v
    assert calls==report['llm_calls']
    result={'replay_verified':True,'completed_llm_calls':calls,'usage':usage}
    (directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    audit(Path(sys.argv[1]).resolve())
