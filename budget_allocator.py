"""Allocate finite experiment budget across constellation branches."""
import math

def allocate(branches, budget=100):
    scored=[]
    for b in branches:
        d=b.get('difference',{}); delta=float(d.get('delta',0)); interval=d.get('interval',[delta,delta])
        uncertainty=max(0,float(interval[1])-float(interval[0]))
        novelty=float(b.get('novelty',0)); prior=float(b.get('prior_success',0))
        # Exploit observed gains, explore uncertain/novel branches, preserve a minimum.
        score=max(0,delta)+0.35*uncertainty+0.15*novelty+0.15*prior
        scored.append((score,b))
    total=sum(s for s,b in scored) or 1
    allocations=[]
    for s,b in scored:
        share=max(1,round(budget*s/total))
        allocations.append({'branch':b.get('branch'), 'constellation':b.get('constellation'),
                            'score':s, 'budget':share, 'reason':{'gain':b.get('difference',{}).get('delta',0),
                            'uncertainty':b.get('difference',{}).get('interval',[0,0]),'novelty':b.get('novelty',0)}})
    # Correct rounding while preserving >=1 per branch.
    while sum(x['budget'] for x in allocations)>budget:
        x=max((x for x in allocations if x['budget']>1),key=lambda z:z['budget'],default=None)
        if not x: break
        x['budget']-=1
    return {'total_budget':budget,'allocations':allocations,
            'policy':'gain + uncertainty + novelty; every surviving branch receives at least one unit'}

if __name__=='__main__':
    print(allocate([{'branch':'a','constellation':'探索','difference':{'delta':.15,'interval':[.1,.2]},'novelty':.2},
                    {'branch':'b','constellation':'評価','difference':{'delta':0,'interval':[-.02,.04]},'novelty':1}],20))
