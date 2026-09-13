"""Continue a staged run without new model calls when the account limit is hit."""
import json, statistics
from pathlib import Path
from harness import evaluate, BASELINE
from staged import difference, passes, CONSTELLATIONS

def main(run_dir):
    run_dir=Path(run_dir)
    start=json.loads((Path('runs/20260911-170813')/'report.json').read_text(encoding='utf-8'))['final_policy']
    proposals=[]
    for p in sorted(run_dir.glob('r1-proposal-*.response.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        i=int(p.name.split('-')[-1].split('.')[0]); role=['linear efficiency','deceptive blocks','rugged robustness','restart falsification','temperature experiment','mutation experiment'][i]
        constellation=next(c for c,rs in CONSTELLATIONS.items() if role in rs)
        proposals.append({'id':i,'role':role,'constellation':constellation,'policy':d['policy']})
    seeds=list(range(880000,880080)); base=evaluate(start,seeds)
    rows=[]
    for p in proposals:
        child=dict(p['policy'])
        # Deterministic local descendant: one controlled parameter step.
        if p['constellation']=='探索': child['flip']=min(.5,max(.001,child['flip']*1.25))
        elif p['constellation']=='頑健性': child['temperature']=min(.2,child['temperature']+.01)
        else: child['restart']=min(120,child['restart']+15)
        scores=evaluate(child,seeds); d=difference(base,scores)
        rows.append({'id':p['id'],'role':p['role'],'constellation':p['constellation'],'parent':p['policy'],'child':child,'difference':d})
    rows.sort(key=lambda r:r['difference']['delta'],reverse=True)
    winner=rows[0]; adopted=winner['child'] if passes(winner['difference']) else start
    audit_seeds=list(range(990000,990100)); audit=difference(evaluate(start,audit_seeds),evaluate(adopted,audit_seeds))
    report={'mode':'offline continuation','reason':'LLM usage limit during descendant calls','source_run':str(run_dir),'start':start,'candidates':rows,'winner':winner['id'],'adopted':adopted,'audit':audit,'llm_calls_added':0,'limitations':['Descendant proposals are deterministic local perturbations, not new LLM reasoning','No claim of RSI']}
    (run_dir/'offline-continuation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (run_dir/'offline-結果.md').write_text('# 星座探索のオフライン継続\n\nLLM使用上限で停止したため、保存済み提案から星座別の子枝を決定的に生成し、別課題で差分を再計算した。\n\n'+ '\n'.join(f"- {r['constellation']} / {r['role']}: 差 {r['difference']['delta']:+.6f}" for r in rows)+f"\n\n最良枝: {winner['constellation']} / {winner['role']}。採用条件: {passes(winner['difference'])}。最終監査差: {audit['delta']:+.6f}。\n\nこれはLLMの再提案ではなく、実験を中断せず記録と検証を続けるためのローカル継続である。\n",encoding='utf-8')
    print(json.dumps({'winner':winner['role'],'delta':winner['difference']['delta'],'accepted':passes(winner['difference']),'audit':audit},ensure_ascii=False))
if __name__=='__main__': main(Path(__import__('sys').argv[1]))
