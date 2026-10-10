#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
def main():
 p=argparse.ArgumentParser(); p.add_argument('--system-prompt',default=str(Path(__file__).resolve().parents[1]/'prompts/system.txt')); p.add_argument('--shared',required=True); p.add_argument('--output',required=True); a=p.parse_args(); system=Path(a.system_prompt).read_text(encoding='utf-8').strip(); rows=[]
 for line in Path(a.shared).read_text().splitlines():
  x=json.loads(line); assistant={'role':'assistant','content':x['selected_completion']}; rows.append({'messages':[{'role':'system','content':system},{'role':'user','content':x['question']},assistant],'problem':x['question'],'reference_answer':x['gold_answer'],'problem_id':x['problem_id'],'selected_rollout_id':x['selected_rollout_id'],'selected_verified_correct':True})
 pq.write_table(pa.Table.from_pylist(rows),a.output,compression='zstd'); print(f'wrote {len(rows)} rows to {a.output}')
if __name__=='__main__': main()
