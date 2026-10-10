#!/usr/bin/env python3
"""Create the explicit, matched no-truncation subset for veRL max_length."""
import argparse,json,os
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
MODEL=os.environ.get('COLDSTART_MODEL','')
SYSTEM_PATH=ROOT/'prompts/system.txt'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',default=MODEL); p.add_argument('--system-prompt',default=str(SYSTEM_PATH)); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--max-length',type=int,default=8192); a=p.parse_args();
 if not a.model: p.error('Set --model or COLDSTART_MODEL')
 SYSTEM=Path(a.system_prompt).read_text(encoding='utf-8').strip(); tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
 kept=[]; lengths=[]; dropped=[]
 for line in Path(a.input).read_text().splitlines():
  if not line.strip(): continue
  x=json.loads(line); msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':x['question']},{'role':'assistant','content':x['selected_completion']}]
  z=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=True)
  raw=z['input_ids'] if hasattr(z,'__getitem__') and not isinstance(z,(list,tuple)) else z
  if raw and isinstance(raw[0],(list,tuple)): raw=raw[0]
  n=len(raw); lengths.append(n); x['sequence_length']=n
  (kept if n<=a.max_length else dropped).append(x)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in kept))
 manifest={'input':str(Path(a.input).resolve()),'output':str(out.resolve()),'max_length':a.max_length,'input_problems':len(kept)+len(dropped),'kept_problems':len(kept),'filtered_problems':len(dropped),'min_sequence_length':min(lengths) if lengths else 0,'max_sequence_length':max(lengths) if lengths else 0,'mean_sequence_length':sum(lengths)/len(lengths) if lengths else 0,'filtered_problem_ids':[x.get('problem_id') for x in dropped]}
 out.with_name(out.stem+'_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
