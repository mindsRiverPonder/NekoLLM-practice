#!/usr/bin/env python3
"""Four-port vLLM HTTP Pass@4 evaluator for the NMath test set."""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import sys
from persona_reward import answers_equivalent, parse_strict_answer, split_completion, token_count

def post(url, payload):
    last=None
    for attempt in range(5):
        try:
            req=Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
            with urlopen(req, timeout=1800) as r: return json.loads(r.read())
        except Exception as exc:
            last=exc; time.sleep(2**attempt)
    raise last

def main():
    p=argparse.ArgumentParser(); p.add_argument('--system-prompt',default=str(Path(__file__).resolve().parents[1]/'prompts/system.txt')); p.add_argument('--model',required=True); p.add_argument('--test-file',default=str(Path(__file__).resolve().parents[1]/'data/evaluation/testset_zh_compact.json')); p.add_argument('--output',required=True); p.add_argument('--ports',default='18000,18001,18002,18003'); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--max-tokens',type=int,default=4096); p.add_argument('--top-p',type=float,default=.95); p.add_argument('--temperature',type=float,default=1.0); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    rows=json.loads(Path(a.test_file).read_text(encoding='utf-8')); system=Path(a.system_prompt).read_text(encoding='utf-8').strip(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); ports=[int(x) for x in a.ports.split(',')];
    def worker(rank):
        path=out/f'predictions.rank{rank}.jsonl'; done={}
        if path.exists():
            for line in path.open(encoding='utf-8'):
                x=json.loads(line); done[(x['index'],x['sample_id'])]=x
        for (index, _), saved in done.items():
            if index >= len(rows) or saved.get('problem') != rows[index].get('problem') or str(saved.get('ground_truth')) != str(rows[index].get('answer')):
                raise RuntimeError(f'incompatible persisted result in {path}; use a clean output directory')
        assigned=list(range(rank,len(rows),len(ports)))
        def generate(job):
            i,sid=job; messages=[{'role':'system','content':system},{'role':'user','content':rows[i]['problem']}]
            payload={'model':a.model,'messages':messages,'temperature':a.temperature,'top_p':a.top_p,'max_tokens':a.max_tokens,'seed':a.seed+i*4+sid,'stream':False}
            result=post(f'http://127.0.0.1:{ports[rank]}/v1/chat/completions',payload); text=result['choices'][0]['message'].get('content',''); parsed=parse_strict_answer(text); correct=bool(parsed['parse_success'] and answers_equivalent(parsed['answer'],str(rows[i]['answer']))); think,response=split_completion(text)
            return {'index':i,'sample_id':sid,'problem':rows[i]['problem'],'ground_truth':str(rows[i]['answer']),'completion':text,'thinking':think,'response':response,'extracted_answer':parsed['answer'],'parse_success':parsed['parse_success'],'format_valid':parsed['format_valid'],'correct':correct,'completion_tokens':result.get('usage',{}).get('completion_tokens'),'finish_reason':result['choices'][0].get('finish_reason'),'sampling':{'temperature':a.temperature,'top_p':a.top_p,'max_tokens':a.max_tokens}}
        for off in range(0,len(assigned),a.batch_size):
            batch=assigned[off:off+a.batch_size]; jobs=[(i,sid) for i in batch for sid in range(4) if (i,sid) not in done]
            with ThreadPoolExecutor(max_workers=min(16,len(jobs) or 1)) as pool: generated=list(pool.map(generate,jobs))
            with path.open('a',encoding='utf-8') as f:
                for x in generated: f.write(json.dumps(x,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno()); done[(x['index'],x['sample_id'])]=x
        return rank
    with ThreadPoolExecutor(max_workers=len(ports)) as ex: list(ex.map(worker,range(len(ports))))
    allrows=[]
    for rank in range(len(ports)):
        shard=out/f'predictions.rank{rank}.jsonl'
        if shard.exists(): allrows += [json.loads(x) for x in shard.read_text(encoding='utf-8').splitlines()]
    allrows.sort(key=lambda x:(x['index'],x['sample_id'])); groups={}
    for x in allrows: groups.setdefault(x['index'],[]).append(x)
    if len(allrows)!=len(rows)*4 or any(len(g)!=4 for g in groups.values()): raise RuntimeError('evaluation incomplete')
    metrics={'outputs':len(allrows),'questions':len(rows),'samples_per_question':4,'MathPass@4':sum(any(x['correct'] for x in g) for g in groups.values())/len(rows),'MeanRolloutAccuracy':sum(x['correct'] for x in allrows)/len(allrows),'AnswerTagValidRate':sum(x['format_valid'] for x in allrows)/len(allrows)}
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in allrows)+'\n',encoding='utf-8'); (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(metrics,ensure_ascii=False))
if __name__=='__main__': main()
