# Neural Marked Point Process with Latent Stochastic Volatility

> English · [中文版](README.zh.md)

Reference implementation of the model described in `main.tex` — a continuous-time
latent jump-diffusion marked point process where

- `z(t)` is a latent state driven by a neural drift `μ_θ` and event-triggered jumps `J_φ`,
- `v(t)` is a positive CIR-type volatility that modulates both diffusion and mark variance,
- `λ_g(t) = Softplus(a_tr·φ_tr(z) + a_vol·φ_vol(v) + b_λ)` is the dual-channel intensity,
- `p(x|z,v)` is a volatility-modulated Gaussian-mixture decoder.

## Project layout

```
SDE/
  src/                     # core library
    nets.py                # µ_θ, J_φ, φ_tr, φ_vol, GMM decoder
    sde.py                 # forward integrator (Euler-Maruyama w/ event jumps)
    intensity.py           # dual-channel ground intensity
    loss.py                # NLL + β·GMM + survival integral
    model.py               # top-level NeuralSVMPP module
    synth.py               # synthetic data generator (Hawkes-like, volatility bursts)
    utils.py               # seeding, device, logging helpers
  tests/                   # correctness checks
    test_nets.py
    test_sde.py            # CIR mean reversion, positivity
    test_intensity.py
    test_loss.py
    test_gradient.py       # autograd vs finite-diff on scalar loss
  scripts/
    train_small.py         # end-to-end sanity run on tiny synthetic batch
    make_synth.py          # generate + save synthetic dataset
  data/                    # (gitignored) synthetic / real data
  OPEN_ISSUES.md           # things that need discussion or later work
  main.tex                 # paper source
```

## Quickstart

One command from a fresh clone takes you all the way to a trained model:

```bash
bash setup.sh                 # venv + deps + tests + download + BERT encode + training sanity
bash setup.sh --quick         # stop after tests + deps (skip the 20-min encoding)
bash setup.sh --no-train      # encode, but skip the training run
bash setup.sh --device mps    # use Apple GPU for the training run (note: usually slower
                              #   than CPU for the unrolled integrator; see OPEN_ISSUES §16)
```

Every step is idempotent — re-running skips steps whose artifact already exists.

### Manual usage (after ``setup.sh`` has installed the env once)

```bash
source .venv/bin/activate
python -m pytest tests/ -v

# Synthetic-data sanity run (no download needed):
python scripts/train_small.py

# Real-data end-to-end (download + encode + train):
gdown 1dakfZtBG0itJTHc3_544t2sPHplTpqW_ -O data/raw/2019-12.csv
python scripts/encode_weibo.py --csv data/raw/2019-12.csv
python scripts/train_weibo.py --bert-cache data/encoded/2019-12_cls.pt \
    --max-seqs 90 --steps 50 --beta 0.1
```

### Dataset

Real-data experiments use **Weibo-COV 2.0** (Hu et al., NLP4COVID@EMNLP 2020):
monthly CSV files of COVID-era Chinese Weibo posts. See
`src/weibo_data.py` for schema, jitter strategy, mark feature extraction, and
per-user sequence construction. `data/` is gitignored — reproduce via
`setup.sh` (which pulls 2019-12) or by calling `gdown` with the IDs listed in
that module's docstring for other months.

## Phases

| Phase | Backend          | Where     | Goal                                                 |
|-------|------------------|-----------|------------------------------------------------------|
| A     | PyTorch autograd | Mac (CPU) | Correctness, small synthetic batch                   |
| B     | `torchsde` adjoint (Stratonovich)      | Linux GPU | Algorithm 1 with O(1)-memory adjoint, large batches |

Phase A checkpoints the entire trajectory and back-propagates via autograd.
Mathematically this is equivalent to the adjoint method of Proposition 1 (and
more numerically stable on short horizons); Phase B is where we wire in
`torchsde.sdeint_adjoint` and pay attention to Brownian-path reproducibility.

See [OPEN_ISSUES.md](OPEN_ISSUES.md) for design decisions that still need discussion.
