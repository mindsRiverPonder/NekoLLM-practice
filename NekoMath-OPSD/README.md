# Neko-OPSD：猫娘数学模型的自蒸馏实验

从同一个 MiniCPM5-2B 猫娘数学冷启动检查点出发，比较 RFT 与 Neko-OPSD。
学生独立生成回答；冻结的初始模型额外查看正确参考回答，在学生轨迹上提供下一 token 分布。

本目录收录最终实现、共同训练集、完整评测 rollout、两位完整评委的原始评分与等权汇总工具。
旧版 OPSD v1、泄漏版 v2、相关历史启动脚本及其结果均不收录。

## 目录

```text
scripts/                     最终 OPSD、RFT 启动、数据构造、数学与人格评测、汇总
data/train/                  1848 题中间集、1613 题最终集、RFT parquet
data/evaluation/             固定顺序的 620 条评测题目记录
data/provenance/             筛选与采样统计
prompts/                     训练人设、正确/未通过回答的评测模板
rollouts/evaluation/         五个正式模型的完整回答（生成外链已脱敏），gzip JSONL
results/math/                原始数学指标与重新计算的对照
results/judges/{qwen,deepseek}/ 两套完整评分（服务诊断已脱敏），gzip JSONL
results/persona_equal_weight/ 逐条 50%-50% 配对与汇总
docs/                        方法、复现说明、来源和整理检查报告
manifest.json                人格评测输入清单及压缩文件哈希
MANIFEST.sha256              整个发布目录的文件校验清单
```

## 模型与结果

保留原实验臂 ID，便于对应已有文章、评分和案例。

| 实验臂 | 含义 | Pass@4 | Rollout 正确率 |
|---|---|---:|---:|
| coldstart | 共同冷启动起点 | 58.06% | 31.65% |
| rft | 自生成正确回答的监督微调 | 64.84% | 36.61% |
| opsd_v3 | 最终 Neko-OPSD，seed 42 | 69.19% | 38.10% |
| opsd_v4_s1234 | 相同实现，seed 1234 | 68.87% | 38.87% |
| opsd_v5_s5678 | 相同实现，seed 5678 | 70.81% | 40.12% |

每模型 620 条题目记录 × 4 次采样 = 2480 条回答；共 12400 条。
数学验证要求完整输出中恰有一个非空 `<answer>...</answer>`，且答案通过等价校验。
620 条记录对应 619 个唯一题目；存在 1 个与最终训练集相同的题目。数据顺序保留原样。

## 训练

Linux、CUDA、多 GPU 环境运行。不要用原始 MiniCPM 基座替代猫娘冷启动检查点来声称复现本实验。
冷启动的数据集由作者开源：[NekoMath](https://huggingface.co/datasets/liumindmind/NekoMath)。
本目录不包含模型权重或冷启动 SFT 全流程。

```bash
export COLDSTART_MODEL=/path/to/coldstart/merged_hf
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/train_rft_core.sh
SEED=42 bash scripts/train_opsd_core.sh
SEED=1234 bash scripts/train_opsd_core.sh
SEED=5678 bash scripts/train_opsd_core.sh
```

三个 OPSD seed 共用 `scripts/train_opsd_core.py`。导出权重：

```bash
python scripts/export_opsd_hf.py --base "$COLDSTART_MODEL" \
  --state outputs/opsd_seed42/student.pt --output outputs/opsd_seed42/merged_hf
```

RFT 使用 veRL 的 SFT trainer。原包未记录 veRL 版本/commit；先确认本地版本支持脚本中的配置。
训练与 vLLM 评测的依赖环境分开，已记录的核心版本见 requirements 文件。

## 评测与重新汇总

```bash
bash scripts/evaluate_hf_vllm.sh /path/to/merged_hf outputs/new_eval
# 不调用 API，重新计算已有的两套评分：
python scripts/aggregate_persona_equal_weight.py --package . --output outputs/persona_recomputed
python scripts/summarize_math.py --output outputs/math_recomputed.json
# 校验文件、题目/采样覆盖、数学指标、双评委配对与训练数据：
python scripts/verify_release.py
```

需要换站点重新进行人格评测时，使用自己的密钥和模型：

```bash
export JUDGE_API_KEY='YOUR_KEY'
python scripts/eval_persona.py --bundle . --prompt-dir prompts/evaluation \
  --base-url 'https://YOUR_ENDPOINT/v1' --model 'YOUR_MODEL' \
  --out-dir outputs/new_judge --concurrency 8 --verify-hashes
```

密钥从环境变量读取，发布目录没有真实密钥。`--model` 会记录在评分结果中。
默认评测输入已支持 `.jsonl.gz`；无需先解压。数学正文与评分保留，生成外链和服务诊断已脱敏。
`summarize.py` 与 `compare_judges.py` 也支持压缩评分文件；`reparse.py` 用于
新评测输出的未压缩 JSONL，不直接改动发布的压缩历史评分。

## 数据构造与已包含的 rollout

DAPO-Math-17k 的 5000 道随机候选题由作者译为中文；冷启动每题采样 8 次，
筛选答对 2～5 次的 1848 题，每题固定选择一个通过验证的回答；完整训练序列不超过
8192 tokens 后得到 1613 题。两种训练使用相同题目与相同选定参考回答。

`data/train/shared_core.jsonl` 保存 1848 条选定回答；`shared_core_len8192.jsonl`
保存最终 1613 条，包含原始推理与回复。数学/人格评测曾分别保存同一批输出，
发布时核对为逐字相同后只保留一份 gzip 文件。压缩不改变内容；随后做了下述隐私脱敏。

## 隐私脱敏

生成回答中的外链统一替换为 `[REDACTED_EXTERNAL_URL]`，训练 JSONL 与 RFT parquet
同步处理。移除评委重试详情、精确运行时间和延迟，ZIP/GZIP 不保存本机路径或文件时间。
公开的数据集出处和本地评测需要的回环地址保留。

原有数学指标、人格分数与失败分类均保持不变，**评分来自脱敏前的原始回答**。
涉及外链替换的回答保留原文 SHA256，并记录脱敏后的 SHA256；长度与 token 数字段仍是
历史原始序列及生成过程的记录。使用脱敏数据重新训练或重新评分，属于新的运行。
处理范围与计数见 [隐私报告](docs/PRIVACY_REPORT.json)。

最初 40000 条候选 rollout、5000 题候选文件以及 OPSD 训练时每次在线生成的轨迹
未包含在收到的资料包中，因此不在本发布目录中。提供的数据构造代码可在补齐源文件后运行。
最终训练集和完整测试 rollout 已收录，可以直接用于训练和结果复核。

整理过程中没有重新训练、重新采样或调用评测 API。详细口径见
[方法](docs/METHOD.md) 与 [复现说明](docs/REPRODUCIBILITY.md)。

## 许可与来源

尚未代作者指定代码许可证；发布到 GitHub 前需选定许可证并添加 `LICENSE`。
题目、翻译、模型权重与第三方代码分别遵守各自上游许可，代码许可证不能自动覆盖这些内容。
来源与修改记录见 [source_manifest.json](docs/source_manifest.json)。
