"""Reproduce math metrics and paired question outcomes from the packaged rollouts."""
import argparse
import gzip
import json
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
ARMS=['coldstart','rft','opsd_v3','opsd_v4_s1234','opsd_v5_s5678']

def summarize(root):
    test=json.loads((root/'data/evaluation/testset_zh_compact.json').read_text(encoding='utf-8'))
    runs={}
    metrics={}
    for arm in ARMS:
        with gzip.open(root/f'rollouts/evaluation/{arm}.jsonl.gz','rt',encoding='utf-8') as f:
            rows=[json.loads(l) for l in f if l.strip()]
        keys={(r['index'],r['sample_id']) for r in rows}
        if len(rows)!=len(test)*4 or keys!={(i,s) for i in range(len(test)) for s in range(4)}:
            raise ValueError('Incomplete/duplicate rollout coverage: '+arm)
        groups=defaultdict(list)
        for r in rows:
            item=test[r['index']]
            if r['problem']!=item['problem'] or str(r['ground_truth'])!=str(item['answer']):
                raise ValueError('Test set mismatch: '+arm)
            groups[r['index']].append(r)
        passed={i:any(r['correct'] for r in group) for i,group in groups.items()}
        runs[arm]=passed
        tokens=[r['completion_tokens'] for r in rows]
        metrics[arm]={'questions':len(test),'outputs':len(rows),'samples_per_question':4,
                      'pass_questions':sum(passed.values()),'correct_rollouts':sum(r['correct'] for r in rows),
                      'MathPass@4':sum(passed.values())/len(test),
                      'MeanRolloutAccuracy':sum(r['correct'] for r in rows)/len(rows),
                      'AnswerTagValidRate':sum(r['format_valid'] for r in rows)/len(rows),
                      'mean_completion_tokens':sum(tokens)/len(tokens)}
    pairs={}
    for arm in ARMS:
        if arm=='rft':continue
        cells=dict(base_only=0,other_only=0,both=0,neither=0)
        for i in range(len(test)):
            b,o=runs['rft'][i],runs[arm][i]
            cells['both' if b and o else 'base_only' if b else 'other_only' if o else 'neither']+=1
        cells['net']=cells['other_only']-cells['base_only']
        pairs[arm]=cells
    return {'baseline':'rft','metrics':metrics,'pairs_vs_rft':pairs,
            'per_question':[dict(index=i,**{arm:runs[arm][i] for arm in ARMS}) for i in range(len(test))]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=summarize(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['metrics'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
