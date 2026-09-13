"""Equal-budget control vs self-updating policy experiment (local, deterministic)."""
import json, statistics
from pathlib import Path
from harness import BASELINE, evaluate
from staged import difference, passes

def children(p):
    out=[]
    for name, change in [('small-flip',{'flip':max(.001,p['flip']*.6)}),('low-explore',{'explore':max(0,p['explore']*.5)}),('restart',{'restart:min':min(120,p['restart']+15)})]:
        c=dict(p)
        if 'restart:min' in change: c['restart']=change['restart:min']
        else: c.update(change)
        c['guidance']=p['guidance']+' | tested '+name
        out.append((name,c))
    return out

def main():
    fixed=dict(BASELINE); selfp=dict(BASELINE); rows=[]
    for gen in range(1,6):
        seeds=list(range(300000+gen*1000,300000+gen*1000+40))
        fixed_old=evaluate(fixed,seeds); self_old=evaluate(selfp,seeds)
        candidates=children(selfp)
        measured=[(n,c,difference(self_old,evaluate(c,seeds))) for n,c in candidates]
        winner=max(measured,key=lambda x:x[2]['delta'])
        if passes(winner[2]): selfp=winner[1]
        fixed_gain=difference(fixed_old,evaluate(fixed,seeds))
        self_gain=difference(self_old,evaluate(selfp,seeds))
        rows.append({'generation':gen,'candidate_deltas':[(n,d['delta']) for n,c,d in measured], 'self_gain':self_gain,'fixed_gain':fixed_gain,'self_policy':selfp})
    audit=list(range(399000,399200)); fixed_a=evaluate(fixed,audit); self_a=evaluate(selfp,audit)
    report={'design':'equal task budget, 5 generations, fixed policy control vs adaptive policy','generations':rows,'final_audit':difference(fixed_a,self_a),'self_update_beats_control':difference(fixed_a,self_a)['delta']>0,'limitations':['Candidate generator is deterministic, not an LLM','Synthetic objective only','No causal isolation of guidance text']}
    out=Path(__file__).with_name('control-report.json'); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'final_audit':report['final_audit'],'beats_control':report['self_update_beats_control']},ensure_ascii=False))
if __name__=='__main__': main()
