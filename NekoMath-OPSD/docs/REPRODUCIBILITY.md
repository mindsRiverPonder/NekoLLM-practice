# 复现与整理说明

## 版本对应

- `train_opsd_core.py` 是最终参考解泄漏修复后的训练器。
- 原实验 `opsd_v3`、`opsd_v4_s1234`、`opsd_v5_s5678` 共用该训练器；数字表示运行历史，不是三个算法版本。
- 废版 v1、v2 不收录。仅保留名称对应的说明，避免错认 seed。
- 原启动器默认写到旧目录、原 chain 脚本等待固定 PID 并混入废版比较，均已改为当前目录的可配置启动。

## 本次对代码的改动

训练器仅更换说明与路径初始化，增加 `--model`、`--system-prompt`；核心函数和优化器保持原样。
启动器保留实际实验的参数：KL chunk=512、pack_tokens=24576、buffer=3。
Python 训练器单独运行时的这三个默认值与实际启动参数不同，复现实验请使用启动器。
新 RFT 启动器直接使用发布的最终 parquet，省去每次重复过滤；如需重建：

```bash
python scripts/filter_shared_by_length.py --input data/train/shared_core.jsonl \
  --output outputs/shared_core_len8192.jsonl --max-length 8192 --model "$COLDSTART_MODEL"
python scripts/build_rft_parquet.py --shared outputs/shared_core_len8192.jsonl \
  --output outputs/rft_core.parquet
```

评测器的路径已参数化，中文读写显式使用 UTF-8。人格 runner 增加 gzip 输入支持，
请求必须显式指定站点和模型；哈希不匹配时退出。评分数值与评分模板保持原样。
修正了 API base URL 已以 `/v1` 结尾时重复拼接 `/v1` 的问题。
选定训练回答仅把服务器绝对来源路径改为逻辑文件名，保留原始 source 行号、采样 ID 和内容。

后续隐私整理又遮蔽了生成回答中的外链，并移除了评委服务诊断与精确时间。
训练 JSONL 与 parquet 同步处理，数学正文不变；保留脱敏前与脱敏后的回答哈希。
原评分针对脱敏前输入，历史长度字段也针对原始序列。详见 PRIVACY_REPORT.json。

## 外部依赖和未提供的材料

本次归档未包含训练权重、冷启动 SFT 全流程、原始 40000 候选回答、
5000 题候选输入和在线 OPSD 训练轨迹文本。发布包不虚构或重新生成这些历史数据。
`requirements-training.txt`/`requirements-evaluation.txt` 仅据归档列出核心版本，
不是完整环境锁定文件。veRL 版本/commit、部分依赖版本与原 tokenizer 文件缺失。
完整 GPU 训练和真实服务调用未在此次整理中重跑。

`persona_reward.py` 中保留可选 veRL 数学归一化；是否安装对应模块会影响部分等价判分。
现存指标基于归档的 correct/format_valid 字段，发布验证不擅自重算这些字段并覆盖原分数。

原人格 failed 模板把未通过回答称为最终答案错误，但路由实际也包含格式失败。
历史模板原样保留以复核评分；修订模板做新实验时，应另建版本并重新评测。
RFT 与 OPSD 的训练 token 统计口径不同，不能仅将两者计数相除当作严格匹配的计算预算。

## 许可准备

作者尚未指定本次代码发布许可证，因此没有擅自添加 MIT/Apache 等授权。
待选定后添加 LICENSE。数据和模型的上游授权独立于代码许可。
