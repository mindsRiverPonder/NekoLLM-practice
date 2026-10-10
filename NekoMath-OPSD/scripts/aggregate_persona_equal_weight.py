"""Pair two complete persona judges and average each numeric rubric dimension."""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ARMS = ['coldstart', 'rft', 'opsd_v3', 'opsd_v4_s1234', 'opsd_v5_s5678']
PERSONA = ['persona_consistency', 'character_specificity', 'naturalness',
           'technical_clarity', 'overacting', 'repetition']
REASON = ['reasoning_correctness', 'solution_progress', 'answer_proximity',
          'error_recoverability']
JUDGES = {'qwen': ('judge_out', 'qwen3.8-flash-next-2'),
          'deepseek': ('judge_out_ds', 'deepseek-v4-flash-ascend')}
LABELS = {'persona_consistency': '人格持续性↑', 'character_specificity': '角色辨识度↑',
          'naturalness': '自然度↑', 'technical_clarity': '技术清晰度↑',
          'overacting': '过度表演↓', 'repetition': '重复↓',
          'reasoning_correctness': '推理正确程度↑', 'solution_progress': '解法推进程度↑',
          'answer_proximity': '答案接近程度↑', 'error_recoverability': '错误可修复性↑'}


def read_rows(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')) as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def select_valid(rows):
    # Same selection as the source package's summarize.py: first valid judgement
    # wins, replacing an earlier invalid attempt; valid retries are not counted twice.
    best = {}
    for r in rows:
        key = (r['index'], r['sample_id'])
        if key not in best or (r.get('schema_ok') and not best[key].get('schema_ok')):
            best[key] = r
    if any(not r.get('schema_ok') or r.get('judge_error') is not None for r in best.values()):
        raise ValueError('A judge does not have complete usable coverage')
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve prior aggregates')
    package = args.package.resolve()
    records, inputs, counts = [], {}, {}
    for arm in ARMS:
        source_path = package / 'rollouts/evaluation' / (arm + '.jsonl.gz')
        source_rows = read_rows(source_path)
        source = {(r['index'], r['sample_id']): r for r in source_rows}
        if len(source_rows) != 2480 or len(source) != 2480:
            raise ValueError('Expected 2480 distinct source rollouts per arm')
        if set(source) != {(i, s) for i in range(620) for s in range(4)}:
            raise ValueError('Question/sample coverage differs from the expected protocol')
        inputs[source_path.relative_to(package).as_posix()] = sha256(source_path)
        selected = {}
        for label, (directory, model) in JUDGES.items():
            path = package / 'results/judges' / label / (arm + '.jsonl.gz')
            selected[label] = select_valid(read_rows(path))
            inputs[path.relative_to(package).as_posix()] = sha256(path)
            if selected[label].keys() != source.keys():
                raise ValueError('Judge and source rollout keys do not match')
            if any(r['judge_model'] != model for r in selected[label].values()):
                raise ValueError('Unexpected judge model')
        counts[arm] = {'all': 2480, 'correct': 0, 'failed': 0}
        for index, sid in sorted(source):
            src = source[(index, sid)]
            route = 'correct' if src['correct'] else 'failed'
            counts[arm][route] += 1
            dims = PERSONA + (REASON if route == 'failed' else [])
            original = {}
            for label, judged in selected.items():
                r = judged[(index, sid)]
                if r['arm'] != arm or r['source_correct'] != src['correct'] or r['prompt_kind'] != route:
                    raise ValueError('Judge routing/source mismatch')
                if r['source_format_valid'] != src['format_valid']:
                    raise ValueError('Judge format label differs from source')
                original[label] = {d: r['judge_parsed'][d] for d in dims}
                if any(type(v) is not int or not 0 <= v <= 4 for v in original[label].values()):
                    raise ValueError('Invalid original numeric rubric scores')
            labels = {}
            if route == 'failed':
                for field in ['primary_error_type', 'error_stage']:
                    values = {label: judged[(index, sid)]['judge_parsed'][field]
                              for label, judged in selected.items()}
                    labels[field] = dict(values, consensus=values['qwen'] if values['qwen'] == values['deepseek'] else None)
            records.append({'arm': arm, 'index': index, 'sample_id': sid,
                'prompt_kind': route, 'source_correct': src['correct'],
                'source_format_valid': src['format_valid'],
                'source_finish_reason': src.get('finish_reason'),
                'source_completion_sha256': hashlib.sha256(src['completion'].encode('utf-8')).hexdigest(),
                'judge_models': {k: v[1] for k, v in JUDGES.items()},
                'judge_weights': {'qwen': 0.5, 'deepseek': 0.5},
                'scores': {d: (original['qwen'][d] + original['deepseek'][d]) / 2 for d in dims},
                'scores_by_judge': original,
                'absolute_score_differences': {d: abs(original['qwen'][d] - original['deepseek'][d]) for d in dims},
                'categorical_labels': labels})
            if src.get('original_completion_sha256'):
                records[-1]['original_source_completion_sha256'] = src['original_completion_sha256']
                records[-1]['source_privacy_redacted'] = True
    if len(records) != 12400:
        raise ValueError('Expected 12400 paired results')

    summary = []
    for arm in ARMS:
        for subset in ['all', 'correct', 'failed']:
            rows = [r for r in records if r['arm'] == arm and
                    (subset == 'all' or r['prompt_kind'] == subset)]
            for d in PERSONA + (REASON if subset == 'failed' else []):
                values = [r['scores'][d] for r in rows]
                q = [r['scores_by_judge']['qwen'][d] for r in rows]
                ds = [r['scores_by_judge']['deepseek'][d] for r in rows]
                if not values:
                    continue
                summary.append({'arm': arm, 'subset': subset, 'metric': d, 'n': len(values),
                    'mean': statistics.mean(values), 'std_population': statistics.pstdev(values),
                    'qwen_mean': statistics.mean(q), 'deepseek_mean': statistics.mean(ds),
                    'deepseek_minus_qwen_mean': statistics.mean(b-a for a, b in zip(q, ds)),
                    'mean_absolute_judge_difference': statistics.mean(abs(b-a) for a, b in zip(q, ds)),
                    'judge_difference_ge_2_rate': sum(abs(b-a) >= 2 for a, b in zip(q, ds))/len(q)})
    output.mkdir(parents=True)
    with (output / 'paired_scores.jsonl').open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / 'summary.json').write_text(json.dumps({'counts': counts, 'summary': summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    index = {(r['arm'], r['subset'], r['metric']): r for r in summary}
    text = ['# 双判官等权评分', '', 'Qwen 与 DeepSeek 各占 50%。每条相同 rollout 的相同数值维度先平均，再按实验臂汇总。',
            '', '原始判官：`qwen3.8-flash-next-2`、`deepseek-v4-flash-ascend`。共 12,400 条有效配对。',
            '', '分数范围 0–4；↑ 越高越好，↓ 越低越好。不构造跨维度总分。',
            '', '重试记录按原包 summarize.py 的规则去重：保留首条有效评分。分类标签不做数值平均；仅双方一致时填写 consensus。',
            '', '标准差根据逐条合并评分重新计算，不是两位判官标准差的平均。']
    for subset, title, dims in [('all', '全部回答：人格六维', PERSONA),
                                ('correct', '正确回答：人格六维', PERSONA),
                                ('failed', '失败回答：人格六维', PERSONA),
                                ('failed', '失败回答：推理四维', REASON)]:
        text += ['', '## ' + title, '', '| 实验臂 | 条数 | ' + ' | '.join(LABELS[d] for d in dims) + ' |',
                 '|---|---:' + '|---:'*len(dims) + '|']
        for arm in ARMS:
            values = [index[(arm, subset, d)]['mean'] for d in dims]
            text.append('| ' + arm + ' | ' + str(counts[arm][subset]) + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    text += ['', '## 使用口径', '',
             '- 原始评分、两位判官的差值与模型名称均保留在 paired_scores.jsonl。',
             '- 失败组包含严格答案标签/格式校验失败；不同实验臂的失败组并非相同样本集合。',
             '- overacting、repetition 是负向维度，角色表达减少也可能降低这两项；应与人格持续性、辨识度共同解释。',
             '- 每题四次采样相互相关；后续置信区间应按题目聚类，而非把所有 rollout 当成独立样本。',
             '- 未运行新的 API 评测，只使用归档中的两套完整评分。']
    (output / 'summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    manifest = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_package': '.', 'models': {k: v[1] for k, v in JUDGES.items()},
                'weights': {'qwen': 0.5, 'deepseek': 0.5}, 'paired_records': len(records),
                'counts': counts, 'source_sha256': inputs,
                'dedup_rule': 'same as source summarize.py: first valid judgement per key',
                'output_sha256': {f.name: sha256(f) for f in output.iterdir() if f.is_file()},
                'script_sha256': sha256(Path(__file__))}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'records': len(records), 'counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
