# 最终实现

RFT：对 coldstart 生成并严格验证的成功回答进行标准 SFT。

Neko-OPSD：学生与固定教师从相同 coldstart 初始化，学生全参数更新，教师冻结。
学生只看人设、题目和自身前缀；教师的 system 消息额外含参考回答和指导指令。
两侧使用同一学生生成前缀，`row_prompts` 保留原有前缀一致性断言。

在回答位置 t，学生分布为 p_t，教师分布为 q_t：

$$K_t=\sum_{v\in\mathcal V}q_t(v)\log\frac{q_t(v)}{p_t(v)},\qquad
\widetilde K_t=5\tanh(K_t/5).$$

$$\mathcal L=\frac1B\sum_{i=1}^{B}\frac1{T_i}\sum_{t=1}^{T_i}\widetilde K_{i,t}.$$

学生轨迹作为已采样的固定输入；教师无梯度，只更新学生。
分块计算 lm_head/KL、token packing、梯度检查点和分片 master optimizer 保留原样。
master 权重 fp32，计算参数与 Adam moments bf16；这与标准的全 fp32 Adam moments 不同。

| 配置 | RFT | Neko-OPSD |
|---|---:|---:|
| 数据量 | 1613 | 1613 |
| epoch | 2 | 2 |
| global batch | 8 | 8 |
| 每卡 micro batch | 1 | 4 |
| lr | 1e-6 | 1e-6 |
| 完整学生序列上限 | 8192 | 10240 |
| 在线生成上限 | — | 8192 |
| 教师序列上限 | — | 20480 |
| seed | 42 | 42 / 1234 / 5678 |
| rollout buffer | — | 3 |

buffer=3 时，两卡一次生成 24 条（每卡 12 条），分三次全局 batch=8 更新；
每条仅使用一次，后两次更新使用的轨迹分别滞后 1、2 步，不做重要性加权。
教师超长序列沿用原实现的头尾裁剪；学生生成预算不由参考回答长度决定。

数学评测：T=1.0、top_p=0.95、max_tokens=16384、每题 4 次，
seed=42+index*4+sample_id。对完整 completion（包含 think）做严格标签检查。
人格评测保持原模板和路由，两位评委只参与评测，不参与训练。数值维度逐条等权平均；
错误类别不做数值平均。未通过集合包含格式失败，不等同于所有答案都算错。

本目录是原有 OPSD 的任务适配实现，不是官方代码，也不把工程适配单独声称为新算法。
