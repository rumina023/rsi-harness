"""Blind-ish benchmark for Anchor Verifier: consistency is not truth."""
from anchor_verifier import AnchorVerifier

CASES=[
 {'id':'supported','text':'リンゴは青い','evidence':True,'expected':'未反証'},
 {'id':'contradicted','text':'リンゴは青い','evidence':False,'expected':'反証あり'},
 {'id':'unknown_entity','text':'ザルゴンは軽い','evidence':None,'expected':'証拠不足'},
 {'id':'unsupported_but_coherent','text':'赤いリンゴは軽い','evidence':None,'expected':'証拠不足'},
]

def run():
    results=[]
    for case in CASES:
        evidence=[] if case['evidence'] is None else [{'claim':case['text'],'supports':case['evidence']}]
        out=AnchorVerifier(evidence).verify(case['text'])
        results.append({'id':case['id'],'expected':case['expected'],'actual':out['status'],
                        'passed':out['status']==case['expected'],'falsification_count':len(out['falsification_tests']),
                        'intervention_count':len(out['interventions'])})
    return {'cases':results,'accuracy':sum(x['passed'] for x in results)/len(results),
            'warning':'小規模で合成された評価。実世界の意味理解性能を表さない。'}

if __name__=='__main__':
    import json
    print(json.dumps(run(),ensure_ascii=False,indent=2))
