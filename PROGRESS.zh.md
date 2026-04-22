# 当前成果

> 截至 2026-04-22 · 本仓库目前的完成状态与关键数据。
> 相关文档:[README.zh.md](README.zh.md)(快速上手)·
> [OPEN_ISSUES.zh.md](OPEN_ISSUES.zh.md)(未决问题)·
> [CLAUDE.md](CLAUDE.md)(给远端 Claude 的交接文档)。

## 一、一句话总结

论文 `main.tex` 提出的**带隐含随机波动率的神经标记点过程**已经在 Mac 本地
完成**阶段 A**:模型、前向 SDE 积分器、MPP 损失、合成数据、真实中文微博数据
(Weibo-COV 2.0)、RoBERTa-wwm-ext 文本编码 —— 全部贯通,21 个单元测试全绿,
代码已推到 `github.com:trumpool/SDE` 远端。**阶段 B**(`torchsde` 伴随 +
Linux GPU)留给下一步。

## 二、已完成的模块

### 2.1 模型与数学

- **潜在动力学**([src/sde.py](src/sde.py)) —— `z, v` 联合演化的 Euler-Maruyama
  积分器,包含:
  - CIR 波动率 full-truncation 保正
  - 相关 Brownian 增量(scalar `ρ`,通过 Cholesky 注入)
  - 事件感知网格(所有事件时间作为断点,无 Euler 步跨越事件)
  - 梯形法则计算 survival 积分 `∫λ_g dt`

- **双通道强度**([src/nets.py](src/nets.py)) —— `λ_g = Softplus(a_tr·φ_tr(z) + a_vol·φ_vol(v) + b)`
  严格按论文 §2.3 实现。

- **GMM mark 解码器** —— `K` 个分量,方差受 `s(v) = 1 + v` 调制,完整支持
  `log_prob` 和 `sample`。

- **顶层 `NeuralSVMPP`**([src/model.py](src/model.py)) —— 拥有全部子网络
  + CIR 参数 `(κ, v̄, ξ, ρ)`;`κ, v̄, ξ` 经 softplus 保正,`ρ` 经 tanh 限幅;
  可选一个可学习的 `Linear(bert_dim → d_x)` 投影层(论文 §4 的 `W_p h^{[CLS]} + b_p`)。

- **MPP 损失**([src/loss.py](src/loss.py)) —— 严格按论文 eq. (2.11):
  `J = -Σ log λ_g(t_i) - β·Σ log p(x_i) + ∫λ_g dt`

### 2.2 数据管线

- **合成数据生成器**([src/synth.py](src/synth.py)) —— 手写的真值过程,用
  Bernoulli thinning 采样,用于正确性检验。

- **真实数据加载器**([src/weibo_data.py](src/weibo_data.py))
  - 解析 Weibo-COV 2.0 CSV(schema 7 个字段)
  - 按 `user_id` 分组
  - 最小长度过滤(默认 ≥ 5 事件)
  - 时间戳 ±0.5 秒抖动(打破秒级并列,论文要求严格递增)
  - 时间单位换成**天**(一月 ≈ 30 天,SDE `dt` 友好)
  - 每个用户独立的 `[t_0, T]` 窗口(含 10 分钟 head/tail padding)
  - 可选传入 BERT 缓存,自动把 768-d `[CLS]` 替换占位特征

- **RoBERTa-wwm-ext 编码器**([src/text_encoder.py](src/text_encoder.py))
  - HuggingFace 惰性加载
  - MPS 批量推理(`batch_size=32`,`max_length=256`)
  - 输出以 fp16 存盘,节省缓存体积

### 2.3 脚本与工具

| 脚本 | 作用 |
|---|---|
| [setup.sh](setup.sh) | 一键引导:venv + 依赖 + 测试 + 数据下载 + 编码 + sanity 训练 |
| [scripts/train_small.py](scripts/train_small.py) | 合成数据端到端冒烟测试 |
| [scripts/make_synth.py](scripts/make_synth.py) | 生成合成数据集并保存 |
| [scripts/encode_weibo.py](scripts/encode_weibo.py) | 一次性批量 BERT 编码 |
| [scripts/train_weibo.py](scripts/train_weibo.py) | 真实数据训练(可选 BERT 缓存) |

### 2.4 文档

| 文档 | 目的 |
|---|---|
| [README.md](README.md) / [README.zh.md](README.zh.md) | 用户导向的快速上手 |
| [OPEN_ISSUES.md](OPEN_ISSUES.md) / [OPEN_ISSUES.zh.md](OPEN_ISSUES.zh.md) | 16 条待讨论的设计决策 |
| [CLAUDE.md](CLAUDE.md) | 10 节的交接文档,供远端 Claude 启动项目 |
| **PROGRESS.zh.md**(本文件) | 成果快照 |

## 三、关键数据

### 3.1 测试情况 —— 21/21 通过

| 测试文件 | 数量 | 覆盖点 |
|---|---|---|
| `test_nets.py` | 6 | 神经网络形状、强度正性、GMM log-prob 有限、GMM 采样 |
| `test_sde.py` | 6 | 网格构造、相关 Brownian 协方差、CIR 均值回复、v 正性、事件跳跃 |
| `test_loss.py` | 2 | 端到端 loss 有限、反传触达所有子模块 |
| **`test_gradient.py`** | **2** | **autograd vs 有限差分,打在 `b_λ` 和 `raw_log_kappa`(后者穿过整条 SDE 积分链)** |
| `test_weibo_data.py` | 3 | CSV 加载、最小长度过滤、summary |
| `test_bert_integration.py` | 2 | BERT 768-d 输入端到端 + 投影层梯度非零 |

> **最关键的测试**是 `test_finite_difference_vs_autograd_on_log_kappa`:它用有限
> 差分校验 `∂L/∂(log κ)` 的 autograd 结果,相对误差 < 0.5%。这个梯度要穿过
> 整个 Euler-Maruyama 积分器、强度函数、GMM 解码器和 survival 积分,
> 任何一处数学写错都会触发测试失败。

### 3.2 数据处理结果

**2019-12.csv(Weibo-COV 2.0)**

| 项 | 数值 |
|---|---|
| 原始行数 | 47,134 |
| 文件大小 | 58 MB |
| 唯一用户 | 43,812 |
| 字段数 | 10(`_id, user_id, crawl_time, created_at, like_num, repost_num, comment_num, content, origin_weibo, geo_info`) |
| `created_at` 精度 | **秒**(不是 README 说的分钟) |
| 秒级时间并列 | 仅 1 对,抖动后全为严格递增 |
| 转发率 | **83.8%**(`is_repost` 是 mark 的重要维度) |
| 过滤后(≥ 5 事件) | **90 条序列 / 728 事件** |

**BERT 编码结果**

| 项 | 数值 |
|---|---|
| 模型 | `hfl/chinese-roberta-wwm-ext`(约 400 MB) |
| 全量编码耗时 | **1242 秒 ≈ 20.7 分钟**(Apple M4 MPS) |
| 吞吐 | 37.9 texts/s |
| 缓存大小 | **70 MB**(47,134 × 768 fp16) |
| 输出精度 | fp16 存盘,float32 参训 |

### 3.3 训练结果(sanity 验证)

**合成数据**([scripts/train_small.py](scripts/train_small.py))

```
8 个合成序列,32 个事件,50 步 Adam,CPU
初始 loss:20.15  →  最终:12.02  (-40%)
```

**真实数据 + BERT 编码**([scripts/train_weibo.py](scripts/train_weibo.py))

```
90 个用户,728 个事件,mark 从 BERT 768-d 投影到 32-d,β=0.1,CPU
[step   0]  loss/ev=+5.637   nll_time=+0.296   nll_mark=+39.662   surv=+1.375
[step  10]  loss/ev=+2.643   nll_time=+0.316   nll_mark=+15.262   surv=+0.802
[step  20]  loss/ev=+4.026   nll_time=+0.361   nll_mark=+30.445   surv=+0.620
[step  29]  loss/ev=+0.057   nll_time=+0.037   nll_mark= −7.576   surv=+0.778

30 步,213 秒 (~7 秒/步)
loss/ev: 5.64 → 0.06 (↓ 99%)
```

关键观察:
- 模型自动识别输入是 768-d(`[data] mark input dim: 768 (BERT [CLS])`),
  经 `Linear(768→32)` 投影后进 SDE
- GMM 在 BERT 特征上**急剧拟合**(`nll_mark` 从 +40 跌到 -8 —— 密度 > 1,
  对连续 mark 是合理的,也提示 728 事件 + 25K 参数过拟合风险极高)
- **`ρ` 从初始 0 漂到 -0.036**:模型自发学到 `z` 与 `v` 的 Brownian 弱负相关
- **CIR 参数几乎没动**:30 步太短,这类超参收敛一般需要 500+ 步

### 3.4 代码规模

| 统计项 | 数值 |
|---|---|
| Python 总行数 | **2063** 行(src + scripts + tests) |
| src 模块 | 8 个(含 `__init__.py`) |
| scripts | 4 个 |
| tests | 7 个文件,**21 个用例** |
| 文档 | 6 个 .md(README/OPEN_ISSUES 各中英两份 + CLAUDE.md + PROGRESS.zh.md) |

## 四、关键设计决策(带依据)

| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| 1 | 子网络结构 | 2 层 tanh MLP,hidden=64 | 论文未规定;保守选择 |
| 2 | 维度 `d_v` | `= d_z` | 论文 §2.2 明示同维 |
| 3 | `s(v)` 调制 | `1 + v` 逐元素 | 论文只说"正值",默认保证下界 1 |
| 4 | `β` 默认 | 1.0(BERT 模式下实测用 0.1) | 论文建议 1;BERT 下 mark 项碾压 |
| 5 | `ρ` 默认 | 0(learnable) | 论文允许,未给默认 |
| 6 | CIR 正性 | full-truncation | Lord/Koekkoek/van Dijk,稳健 |
| 7 | Ito vs Strato | 特定模型用 **Ito** | `z` 的扩散系数 `sqrt(v)` 不含 `z`,两者等价;CIR 惯例 |
| 8 | 事件时间精度 | 秒级 + ±0.5 秒抖动 | 原数据秒级精度,抖动打破并列 |
| 9 | 时间单位 | **天** | 月级数据 → T≈30 天,`dt=0.05` 天 ≈ 72 分钟 |
| 10 | 最小序列长度 | ≥ 5 事件 | 短序列 MPP 退化;本文档 §3.2 列出分布 |
| 11 | 转发策略 | 保留 + `is_repost` 标志 | 用户语义一致,83.8% 转发率不丢 |
| 12 | BERT max_length | 256 | 中位帖长 322 字,覆盖核心信号;比 512 快 4× |
| 13 | BERT 投影 `d_in` | 32 | 与论文 §4 一致,可学习 |

完整理由见 [OPEN_ISSUES.zh.md](OPEN_ISSUES.zh.md) 的 16 条编号条目。

## 五、Git 与远端

**远端**:`git@github.com:trumpool/SDE.git`(账户 `trumpool`,SSH,`origin/main` 已同步)

**7 个 commit 的时间线:**

| 日期 | SHA | 内容 |
|---|---|---|
| 2026-04-20 | `4ec450c` | 阶段 A 初始实现 + 合成数据 |
| 2026-04-20 | `983ad54` | Weibo 真数据管线(CSV loader + 占位 mark + 训练脚本) |
| 2026-04-20 | `3ea70ee` | RoBERTa-wwm-ext 文本编码接通 |
| 2026-04-21 | `3bf6ca9` | OPEN_ISSUES 更新(β 调参、burst 用户、MPS 慢) |
| 2026-04-21 | `959cb73` | `setup.sh` 一键引导 |
| 2026-04-21 | `f774a78` | `CLAUDE.md` 远端交接文档 |
| 2026-04-22 | `b9f671d` | README / OPEN_ISSUES 中文版 |

## 六、管线现在能做什么

一条命令从零到训好:

```bash
git clone git@github.com:trumpool/SDE.git
cd SDE
bash setup.sh                 # ≈ 25 分钟:装环境 + 测试 + 下载数据 + 编码 + 训练 sanity
```

单独各阶段可独立运行(`setup.sh` 的 flag 或直接调脚本)。详见
[README.zh.md](README.zh.md)。

## 七、目前还做不了的事

下列项目依赖后续工作(对应 [OPEN_ISSUES.zh.md](OPEN_ISSUES.zh.md) 编号):

- **长训练到收敛** —— 30 步只是 sanity。收敛与否需要 500+ 步 + train/val split + β 扫描
- **多用户 batch** —— 现在是 Python for 循环串行,对 90 序列够用,扩到 10K+ 必须 batch 化(§12)
- **`torchsde` 伴随法** —— 论文 Algorithm 1 的真正 O(1) 内存实现,留给 Linux GPU(§9, §10)
- **多月数据** —— Drive 上 2019-12 到 2020-12 共 13 个月 CSV,Drive ID 表在 [CLAUDE.md](CLAUDE.md) §9
- **RoBERTa fine-tuning** —— 目前 BERT 冻结,只训投影层;端到端微调是另一条路径
- **消融实验** —— 按论文 §2.3 的暗示,关 `a_vol` vs 关 `a_tr` 对 NLL 的对比,证明波动率通道的预测价值

## 八、迁移到 Linux 的成本

阶段 B 的 Linux 迁移是**低摩擦**的:

- `bash setup.sh` 在 Linux 上应当开箱即用(PyTorch 同样走 pip)
- 仅需把 `--device cpu` 换成 `--device cuda`
- CUDA 很可能不会遇到 Mac MPS 的 kernel launch 开销问题(§16)
- 大批量训练前仍需先解决 §12(变长序列 batch)

预计工作量:装机 30 分钟 + 验证 setup.sh 通 30 分钟 + batch 化改造 1-2 天。

---

*文档状态 · 最后更新:2026-04-22,对应 commit `b9f671d`*
