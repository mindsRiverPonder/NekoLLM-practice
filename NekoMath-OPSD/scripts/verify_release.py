"""Offline checks of packaged files and the exact stored experimental evidence."""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['coldstart', 'rft', 'opsd_v3', 'opsd_v4_s1234', 'opsd_v5_s5678']
EXPECTED = {'coldstart': (360,785), 'rft': (402,908), 'opsd_v3': (429,945),
            'opsd_v4_s1234': (427,964), 'opsd_v5_s5678': (439,995)}
PERSONA = ['persona_consistency','character_specificity','naturalness','technical_clarity','overacting','repetition']
REASON = ['reasoning_correctness','solution_progress','answer_proximity','error_recoverability']

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read_rows(path):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def selected(rows):
    out={}
    for r in rows:
        key=(r['index'],r['sample_id'])
        if key not in out or (r.get('schema_ok') and not out[key].get('schema_ok')): out[key]=r
    require(all(r.get('schema_ok') and r.get('judge_error') is None for r in out.values()),'Invalid judge coverage')
    return out

def verify(root):
    report={'checks':{},'math':{},'data':{},'notes':['Offline evidence validation; no GPU training or external API calls.']}
    count=0
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()),'Unsafe manifest path')
        require(sha(path)==digest,'File hash mismatch: '+name)
        count+=1
    report['checks']['manifest_files']=count
    pyfiles=list((root/'scripts').glob('*.py'))
    for path in pyfiles: ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
    report['checks']['python_syntax_files']=len(pyfiles)
    # Search without emitting credential values or matching text.
    patterns=[re.compile(r'\b(?:sk-|nvapi-)[A-Za-z0-9_\-]{16,}'),
              re.compile(r'Bearer\s+[A-Za-z0-9_\-]{20,}'),
              re.compile(r'(?i)["\'](?:api_key|access_token|secret_key)["\']\s*:\s*["\'][A-Za-z0-9+/=_\-]{16,}["\']')]
    scanned=0
    for path in root.rglob('*'):
        if not path.is_file() or 'outputs' in path.relative_to(root).parts: continue
        if path.suffix=='.gz':
            with gzip.open(path,'rt',encoding='utf-8') as f: text=f.read()
        elif path.suffix=='.parquet': continue
        else:
            try: text=path.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError: continue
        require(not any(p.search(text) for p in patterns),'Credential-like value found in '+path.relative_to(root).as_posix())
        private_path=re.search(r'/data/[A-Za-z0-9_.-]+/(?:small_nmath|Nmath)',text)
        windows_profile=re.search(r'[A-Z]:\\Users\\[^\\\s"\']+',text)
        require(not private_path and not windows_profile,'Private path in '+path.name)
        scanned+=1
    report['checks']['credential_pattern_scan_files']=scanned
    test=json.loads((root/'data/evaluation/testset_zh_compact.json').read_text(encoding='utf-8'))
    require(len(test)==620,'Expected 620 evaluation records')
    core=read_rows(root/'data/train/shared_core.jsonl')
    final=read_rows(root/'data/train/shared_core_len8192.jsonl')
    require(len(core)==1848 and len(final)==1613,'Training set counts differ')
    coremap={r['problem_id']:r for r in core}
    require(len(coremap)==1848 and len({r['problem_id'] for r in final})==1613,'Duplicate training IDs')
    for r in final:
        require(2<=r['correct_count']<=5 and r['num_rollouts']==8 and r['selected_verified_correct'] is True,'Selection contract differs')
        require(r['sequence_length']<=8192,'Stored sequence length exceeds 8192')
        require(r['selected_completion']==coremap[r['problem_id']]['selected_completion'],'Selected answer changed')
    report['data']={'intermediate_questions':1848,'final_questions':1613,'length_filtered':235,'evaluation_records':620,
                    'unique_evaluation_questions':len({r['problem'] for r in test})}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report['checks']['rft_parquet']='SKIPPED: install pyarrow to compare parquet contents'
    else:
        table=pq.read_table(root/'data/train/rft_core.parquet').to_pylist()
        require(len(table)==1613,'Parquet count differs')
        system=(root/'prompts/system.txt').read_text(encoding='utf-8').strip()
        for row,r in zip(table,final):
            require(row['messages']==[{'role':'system','content':system},{'role':'user','content':r['question']},
                                     {'role':'assistant','content':r['selected_completion']}],'RFT/OPSD sample mismatch')
        report['checks']['rft_parquet']='1613 samples exactly match final questions, reference completions and system prompt'
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    paired={(r['arm'],r['index'],r['sample_id']):r for r in read_rows(root/'results/persona_equal_weight/paired_scores.jsonl')}
    require(len(paired)==12400,'Expected 12400 unique paired scores')
    models={'qwen':'qwen3.8-flash-next-2','deepseek':'deepseek-v4-flash-ascend'}
    coverage={(i,s) for i in range(620) for s in range(4)}
    for arm in ARMS:
        path=root/manifest['arms'][arm]['file']
        require(sha(path)==manifest['arms'][arm]['sha256'],'Rollout manifest hash differs')
        rows=read_rows(path)
        require(len(rows)==2480 and {(r['index'],r['sample_id']) for r in rows}==coverage,'Rollout coverage differs: '+arm)
        for r in rows:
            require(r['problem']==test[r['index']]['problem'] and str(r['ground_truth'])==str(test[r['index']]['answer']),'Evaluation order mismatch')
        groups=[[r for r in rows if r['index']==i] for i in range(620)]
        passes=sum(any(r['correct'] for r in g) for g in groups)
        correct=sum(r['correct'] for r in rows)
        require((passes,correct)==EXPECTED[arm],'Stored result changed: '+arm)
        metrics={'outputs':len(rows),'questions':620,'samples_per_question':4,'MathPass@4':passes/620,
                 'MeanRolloutAccuracy':correct/2480,'AnswerTagValidRate':sum(r['format_valid'] for r in rows)/2480}
        stored=json.loads((root/f'results/math/{arm}_metrics.json').read_text(encoding='utf-8'))
        for k,v in metrics.items(): require(math.isclose(stored[k],v,abs_tol=1e-12),'Metric mismatch: '+arm+'/'+k)
        judges={label:selected(read_rows(root/f'results/judges/{label}/{arm}.jsonl.gz')) for label in models}
        for label,j in judges.items(): require(set(j)==coverage,'Judge coverage mismatch: '+arm+'/'+label)
        for r in rows:
            key=(r['index'],r['sample_id'])
            pair=paired[(arm,*key)]
            require(hashlib.sha256(r['completion'].encode('utf-8')).hexdigest()==pair['source_completion_sha256'],'Completion content mismatch')
            for label,j in judges.items():
                q=j[key]
                require(q['judge_model']==models[label] and q['arm']==arm and q['source_correct']==r['correct'] and q['source_format_valid']==r['format_valid'],'Judge identity/routing mismatch')
            for d in PERSONA+(REASON if not r['correct'] else []):
                expected=(judges['qwen'][key]['judge_parsed'][d]+judges['deepseek'][key]['judge_parsed'][d])/2
                require(pair['scores'][d]==expected,'Equal-weight score mismatch')
        report['math'][arm]=dict(metrics,pass_questions=passes,correct_rollouts=correct)
    report['checks']['paired_scores']='12400 completions and all per-dimension 50%-50% scores verified'
    require(not any('v1_backup' in p.name or 'pre_leakfix' in p.name or p.name in ['opsd','opsd_v2'] for p in root.rglob('*')),'Obsolete version included')
    require(all(p.stat().st_size<100*1024*1024 for p in root.rglob('*') if p.is_file()),'Individual release file exceeds 100 MiB')
    report['checks']['obsolete_versions']='excluded'
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--report',type=Path)
    a=ap.parse_args()
    result=verify(a.root.resolve())
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
