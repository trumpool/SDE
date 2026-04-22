# 带隐含随机波动率的神经标记点过程

> [English version](README.md) · 中文版

`main.tex` 中所提出模型的参考实现 —— 一个连续时间的潜在跳跃-扩散标记点过程,具备以下结构:

- `z(t)`:由神经漂移 `μ_θ` 驱动、事件触发跳跃 `J_φ` 的潜在状态;
- `v(t)`:正值 CIR 型波动率过程,同时调制 `z` 的扩散强度和 mark 方差;
- `λ_g(t) = Softplus(a_tr·φ_tr(z) + a_vol·φ_vol(v) + b_λ)`:**双通道强度**;
- `p(x|z,v)`:**波动率调制的高斯混合解码器**。

## 项目结构

```
SDE/
  src/                     # 核心库
    nets.py                # µ_θ, J_φ, φ_tr, φ_vol, GMM 解码器
    sde.py                 # 前向积分器 (Euler-Maruyama + 事件跳跃)
    loss.py                # NLL + β·GMM + survival 积分
    model.py               # 顶层 NeuralSVMPP 模块
    synth.py               # 合成数据生成器 (Hawkes 风格 + 波动率突发)
    weibo_data.py          # Weibo-COV 2.0 CSV → 按用户分组的事件序列
    text_encoder.py        # RoBERTa-wwm-ext [CLS] 编码器
    utils.py               # 随机种子、设备选择
  tests/                   # 正确性检查(pytest,共 21 个)
    test_nets.py
    test_sde.py            # CIR 均值回复、正性
    test_loss.py
    test_gradient.py       # autograd vs 有限差分(最关键的测试)
    test_weibo_data.py
    test_bert_integration.py
  scripts/
    train_small.py         # 合成数据端到端 sanity run
    make_synth.py          # 生成并保存合成数据集
    encode_weibo.py        # 离线把 Weibo CSV 编码为 [CLS] 缓存
    train_weibo.py         # 真实 Weibo 数据训练(可选 BERT 缓存)
  data/                    # (gitignore) 合成/真实数据
  OPEN_ISSUES.md           # 待讨论或延后的设计问题(中文版见 OPEN_ISSUES.zh.md)
  CLAUDE.md                # 给下一位接手 Claude 的详细交接文档
  setup.sh                 # 一键引导脚本
  main.tex                 # 论文正文
```

## 快速开始

从一次全新的 clone 开始,一条命令直到训练完一次:

```bash
bash setup.sh                 # venv + 依赖 + 测试 + 下载 + BERT 编码 + sanity 训练
bash setup.sh --quick         # 装完依赖和测试就停(跳过 20 分钟的编码)
bash setup.sh --no-train      # 只编码不训练
bash setup.sh --device mps    # 训练跑在 Apple GPU 上(注意:在展开积分器上通常
                              #   比 CPU 还慢,见 OPEN_ISSUES §16)
```

每一步都 idempotent —— 已经产出的 artifact 会被跳过。

### 手动用法(`setup.sh` 装好 env 之后)

```bash
source .venv/bin/activate
python -m pytest tests/ -v

# 合成数据 sanity(无需下载):
python scripts/train_small.py

# 真实数据端到端(下载 + 编码 + 训练):
gdown 1dakfZtBG0itJTHc3_544t2sPHplTpqW_ -O data/raw/2019-12.csv
python scripts/encode_weibo.py --csv data/raw/2019-12.csv
python scripts/train_weibo.py --bert-cache data/encoded/2019-12_cls.pt \
    --max-seqs 90 --steps 50 --beta 0.1
```

### 数据集

真实数据实验使用 **Weibo-COV 2.0**(Hu 等人,NLP4COVID@EMNLP 2020):
每月一个 CSV 文件的新冠疫情期间中文微博帖子。字段 schema、时间戳抖动策略、
mark 特征提取与按用户分序列的逻辑,参见 `src/weibo_data.py`。
`data/` 整个目录已 gitignore —— 通过 `setup.sh`(自动拉 2019-12)或用
`gdown` 配合 `CLAUDE.md` 表中的 Drive ID 下载其他月份。

## 实现阶段

| 阶段 | 反向传播方式 | 运行位置 | 目标 |
|------|------------|---------|------|
| A | PyTorch autograd(展开积分器) | Mac CPU | 正确性 + 小批量合成数据 |
| B | `torchsde.sdeint_adjoint`(Stratonovich 伴随) | Linux GPU | 论文 Algorithm 1,O(1) 内存,大批量 |

阶段 A 检查整条轨迹,通过 autograd 反传 —— 这在数学上等价于论文命题 3.1
的伴随方法(且在短视界内数值更稳)。阶段 B 是真正落地 `torchsde.sdeint_adjoint`
的地方,需要小心处理 Brownian 路径可复现性和事件跳跃的手工拼接。

详见 [OPEN_ISSUES.md](OPEN_ISSUES.md)(中文:[OPEN_ISSUES.zh.md](OPEN_ISSUES.zh.md))。

## 当前状态(2026-04-22)

- ✅ **21/21 测试通过**,包括最关键的 autograd vs 有限差分(穿过整条 SDE 积分链)
- ✅ **BERT 真实数据管线跑通**:2019-12 的 47K 条推文完成编码,28 秒训练 30 步 loss/ev 5.6 → 0.06
- ⏳ Phase B(`torchsde` 伴随)等迁移到 Linux GPU
- ⏳ 长训练 + validation split + β 扫描

Git 远端:`git@github.com:trumpool/SDE.git`(分支 `main`)
