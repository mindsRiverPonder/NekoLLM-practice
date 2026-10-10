#!/usr/bin/env python3
"""Freeze the one and only problem/trajectory sample used by RFT and OPSD."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
from persona_reward import answers_equivalent, parse_strict_answer, split_completion

DEFAULT_ROLLOUT = str(Path(__file__).resolve().parents[1] / 'data/source/coldstart_5k_8samples.jsonl')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/data_preparation'

def stable_seed(seed, key):
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rollouts',default=DEFAULT_ROLLOUT); ap.add_argument('--seed',type=int,default=20260915); ap.add_argument('--max-length',type=int,default=20480); ap.add_argument('--output-dir',default=str(OUT)); a=ap.parse_args()
    groups=defaultdict(list)
    with Path(a.rollouts).open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                x=json.loads(line); x['_line_no']=line_no
                groups[(x.get('source_index'), x.get('problem'), str(x.get('ground_truth')))].append(x)
    selected=[]; counts=Counter(); rejected=Counter()
    for key, rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if len(rows)!=8: rejected['not_8_rollouts']+=1; continue
        problem, gold = rows[0]['problem'], str(rows[0]['ground_truth'])
        verified=[]
        for x in rows:
            completion=str(x.get('completion','')); parsed=parse_strict_answer(completion)
            ok=bool(parsed['parse_success'] and answers_equivalent(parsed['answer'],gold))
            think,response=split_completion(completion)
            x['_verification']={'parse':parsed,'correct':ok}
            x['_thinking']=think; x['_response']=response
            if ok: verified.append(x)
        n=len(verified); counts[n]+=1
        if n<2 or n>5: rejected[f'correct_count_{n}']+=1; continue
        # Hash-seeded choice is stable across processes and Python versions.
        chosen=verified[random.Random(stable_seed(a.seed, key[0] or key[1])).randrange(n)]
        rec={'problem_id':key[0], 'question':problem, 'gold_answer':gold,
             'correct_count':n, 'num_rollouts':8, 'selected_rollout_id':{'index':chosen.get('index'),'sample_id':chosen.get('sample_id'),'source_index':chosen.get('source_index'),'line_no':chosen['_line_no']},
             'selected_completion':chosen['completion'], 'selected_reasoning':chosen['_thinking'], 'selected_response':chosen['_response'],
             'selected_parse':chosen['_verification']['parse'], 'selected_verified_correct':True,
             'all_rollout_ids':[{'index':x.get('index'),'sample_id':x.get('sample_id'),'verified_correct':x['_verification']['correct']} for x in sorted(rows,key=lambda z:int(z.get('sample_id',0)))],
             'source_rollout_file':str(Path(a.rollouts).resolve()), 'selection_seed':a.seed}
        selected.append(rec)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/'shared_core.jsonl').open('w',encoding='utf-8') as f:
        for x in selected: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    manifest={'rollout_file':str(Path(a.rollouts).resolve()),'seed':a.seed,'candidate_problem_groups':sum(counts.values()),'core_problems':len(selected),'correct_count_histogram':dict(sorted(counts.items())),'rejected':dict(sorted(rejected.items())),'verifier':'persona_reward.parse_strict_answer + answers_equivalent','selected_trajectory_contract':'same selected_completion in RFT and OPSD'}
    (out/'shared_core_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
