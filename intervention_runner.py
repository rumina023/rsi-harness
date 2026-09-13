"""Execute Anchor Verifier interventions against the local semantic scaffold."""
from copy import deepcopy
from anchor_verifier import AnchorVerifier
from harness import BASELINE, evaluate
from staged import difference

def run_interventions(text: str) -> dict:
    verifier=AnchorVerifier()
    base=verifier.verify(text)
    graph=verifier.decompose(text)
    results=[]
    for test in verifier.interventions(graph):
        changed=deepcopy(graph)
        target=next((e for e in changed.entities if e['id']==test['target']),None)
        if target and target['attributes']:
            target['attributes'][0]['value']=test['change']['value']
            target['text']=test['change']['value']
        elif test['target'] not in [r['predicate'] for r in changed.relations]:
            continue
        else:
            changed.relations=[]
        rebuilt=verifier.reconstruct(changed)
        results.append({'test':test,'reconstructed':rebuilt,
                        'relation_count_before':len(graph.relations),
                        'relation_count_after':len(changed.relations),
                        'changed':rebuilt!=base['reconstruction'],
                        'falsifies_structure':bool(graph.relations) and not changed.relations})
    return {'source':text,'base_reconstruction':base['reconstruction'],
            'interventions':results,
            'all_tests_executed':len(results)==len(base['interventions']),
            'note':'介入で表現が変わることは因果的真理の証明ではない。'}

def run_policy_interventions(policy=None, seeds=range(500000, 500020)):
    """One-factor-at-a-time interventions against the fixed simulator."""
    policy=dict(policy or BASELINE)
    baseline=evaluate(policy,seeds)
    candidates=[]
    for field,value in [('explore',0.2),('flip',0.02),('temperature',0.05),('restart',60)]:
        child=dict(policy); child[field]=value
        scores=evaluate(child,seeds)
        candidates.append({'field':field,'from':policy[field],'to':value,
                           'difference':difference(baseline,scores)})
    return {'baseline':policy,'seeds':list(seeds),'interventions':candidates,
            'interpretation':'各差分は他のパラメータを固定した一因子介入。外部課題で再検証が必要。'}

if __name__=='__main__':
    import json,sys
    if sys.argv[1:] and sys.argv[1]=='--policy':
        print(json.dumps(run_policy_interventions(),ensure_ascii=False,indent=2))
    else:
        print(json.dumps(run_interventions(' '.join(sys.argv[1:]) or '私 は 赤い リンゴ が 好き だ'),ensure_ascii=False,indent=2))
