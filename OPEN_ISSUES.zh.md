# 待讨论问题 / 开放事项

> [English version](OPEN_ISSUES.md) · 中文版

标记规约:
- **[BLOCKER]** — 上真实数据前必须解决
- **[DESIGN]** — 当前有可用默认值,但值得讨论确认
- **[DEFERRED]** — 已认识到,但推迟到某个阶段再处理
- **[RESOLVED]** — 已解决,留条目作为历史记录

## 模型规范中的空白

1. **[DESIGN] 子网络结构论文未给定。**
   论文未规定 `μ_θ`、`J_φ`、`φ_tr`、`φ_vol` 及 GMM 各头的具体形式。当前默认:
   - `μ_θ(z, t)`:2 层 MLP,输入 `[z, t]`,hidden=64,tanh,输出 `d_z`
   - `J_φ(z, x)`:2 层 MLP,输入 `[z, x]`,hidden=64,tanh,输出 `d_z`
   - `φ_tr(z)`:Linear(d_z → d_h) + tanh,然后与 `a_tr` 点积
   - `φ_vol(v)`:Linear(d_v → d_h) + tanh,然后与 `a_vol` 点积
   - GMM:`K` 个分量,`π, μ, logσ²` 各自是共享 MLP 主干 + 线性头

2. **[DESIGN] 维度 `d_z`、`d_v`。**
   论文 §2.2 写的是 `z ∈ ℝ^{d_z}, v ∈ ℝ_+^{d_z}`(**同维**),但 §2.1 的一般
   形式里写 `v ∈ ℝ^{d_v}`。我们按 §2.2 实现,采用逐坐标波动率,`d_v = d_z`。

3. **[DESIGN] mark 解码器中的波动率缩放函数 `s(v)`。**
   论文只说"正值缩放函数"。我们默认 `s(v) = 1 + v`(逐元素),使高波动率
   单调增大 mark 方差,同时下界为 1。备选方案:`softplus(v)` 或 `exp(v)`。

4. **[DESIGN] 默认 `β`。** 论文建议 `β = 1`,我们暴露为超参数。

5. **[DESIGN] 默认相关系数 `ρ`。** 论文允许 `d⟨W_z,W_v⟩ = ρ dt`,未给默认值。
   我们默认 `ρ = 0`(独立)并将 `ρ` 作为标量超参数暴露。实现上通过
   Cholesky 分解把相关性注入 `[dW_z, dW_v]` 的联合增量。

## 数值问题

6. **[BLOCKER for real data] CIR 正性保持。**
   当 Feller 条件 `2κv̄ ≥ ξ²` 不满足时,`dv = κ(v̄ - v)dt + ξ·sqrt(v)·dW_v`
   的 Euler-Maruyama 步骤有可能跨入负值。我们采用 Lord/Koekkoek/van Dijk
   的 **full-truncation** 方案(在 `sqrt` 之前做 `v_+ = max(v, 0)`)。
   真实数据规模下应当和 (a) absorbing 方案、(b) 精确 CIR 步进(非中心卡方分布
   抽样)做对比。

7. **[DESIGN] Survival 积分 `∫λ_g dt`。**
   用梯形法则在离散化网格上近似。网格步长是超参数:太粗 → NLL 有偏差;
   太细 → 代价高。需要做网格敏感性检验。

8. **[DESIGN] 事件处的离散化。**
   Algorithm 1 要求在 `[t_i, t_{i+1})` 上积分。我们构造一个把所有事件时间
   作为断点的共享网格,保证没有 Euler 步会跨越一个事件。

## 伴随方法(Algorithm 1)

9. **[DEFERRED → Phase B] `torchsde` Stratonovich 伴随 + 事件跳跃。**
   阶段 A 直接用 PyTorch autograd 反传展开后的 Euler-Maruyama 积分器 ——
   这在路径意义上等价于论文的伴随方法。阶段 B 要切换到
   `torchsde.sdeint_adjoint`,在各 event-free 子区间逐段调用,并冻结 Brownian
   序列以复现路径。潜在坑点:
   - `torchsde` 期待 `sde_type='stratonovich'` 时的 `sqrt(v(t))⊙dW`,对于
     CIR 这种状态相关扩散项,Ito-Stratonovich 修正必须要么预加到漂移里,
     要么统一用 `sde_type='ito'` 保持代码 Ito 一致。

10. **[DEFERRED → Phase B] O(1) 内存反传 + 事件跳跃。**
    论文事件处伴随更新(式 3.20–3.24)需要左极限状态 `s(t_i^-)`。阶段 B 的方案:
    只在事件时间点 checkpoint 左极限状态,事件之间通过反向 SDE 重构路径 ——
    这是工程上最复杂的一步。

## 文本 mark(§4)

11. **[RESOLVED 2026-04-20] RoBERTa-wwm-ext 编码已接通。**
    `src/text_encoder.py` + `scripts/encode_weibo.py` 用
    `hfl/chinese-roberta-wwm-ext`(max_length=256,batch_size=32)把每条微博
    编码为 `[CLS]` 向量,以 fp16 存为 `data/encoded/<month>_cls.pt`。
    `NeuralSVMPP` 里带一个可学习的 `Linear(768, d_x=32)` 投影(对应论文
    §4 的 `W_p h^{[CLS]} + b_p`),在 `forward_sequence` 和 `compute_loss`
    里自动触发。Apple M4 MPS 吞吐约 48 texts/s,一个月约 16-20 分钟。

## 训练 / 数据

12. **[BLOCKER for real data] 变长序列的 batch 化。**
    目前 `train_weibo.py` 是 Python 循环逐条序列跑,没有真正的 batch。
    放大到真实规模时需要 packed sequence + loss 上的 mask。对于 2019-12
    的 90 条序列,CPU 上一步约 7 秒,仍可接受。

13. **[DESIGN] 初始条件 `s(t_0)`。**
    默认 `z(t_0) = 0`,`v(t_0) = v̄`(均值回复点)。可以改成 learnable。

14. **[DESIGN] 带 BERT 特征时 `β` 的平衡。**
    当 mark 是 32-d BERT 投影时,GMM 对数似然在 `β=1` 下会碾压 timing 项。
    在 2019-12 上我们实测用 `β=0.1` 时两项能同时下降,比较健康。开放问题:
    是把默认改成依据某种原则(例如让 mark 项逐维归一化到与 timing 同尺度),
    还是继续把 `β` 作为需要调的超参?

15. **[DESIGN] 短窗口"突发"序列。**
    2019-12 里 14/90 的用户 horizon `T < 0.1 天` —— 几分钟内发了 5+ 条帖子
    (疑似转发 cascade 或 bot)。SDE 网格仍然能处理它们
    (`n_sub = max(1, ceil(span/dt))`),但 MPP 的语义可能不合适:在 35 秒的
    突发里,潜在跳扩散几乎没时间演化。可选方案:
    (a) 加一个最小 horizon 的过滤阈值;
    (b) 用一个单独的"簇发事件"似然处理它们;
    (c) 接受现状并前进 —— 样本量小(约 15%),对主结果影响有限。

16. **[DEFERRED → Phase B] MPS 上展开积分器的性能。**
    MPS 跑带很多小算子的 Euler-Maruyama 展开时,比 CPU 慢得多;一次 90 序列
    的训练在 MPS 上挂了 10+ 分钟才被我 kill。CUDA 应当会好(kernel launch
    开销小)。如果 CUDA 依然慢,修复方式是把 batch 改成沿网格并行,
    而不是沿序列并行。留个记号,Linux 迁移时关注。
