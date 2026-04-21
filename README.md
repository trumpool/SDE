# Neural Marked Point Process with Latent Stochastic Volatility

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

```bash
# 1. create venv & install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. run unit tests
python -m pytest tests/ -v

# 3. tiny end-to-end training run (synthetic)
python scripts/train_small.py

# 4. real-data end-to-end run (needs one Weibo-COV 2.0 monthly CSV)
mkdir -p data/raw
gdown 1dakfZtBG0itJTHc3_544t2sPHplTpqW_ -O data/raw/2019-12.csv
python scripts/train_weibo.py --max-seqs 30 --steps 50
```

### Dataset

Real-data experiments use **Weibo-COV 2.0** (Hu et al., NLP4COVID@EMNLP 2020):
monthly CSV files of COVID-era Chinese Weibo posts. See
`src/weibo_data.py` for schema, jitter strategy, mark feature extraction, and
per-user sequence construction. `data/` is gitignored — reproduce via
`gdown` using the IDs listed in that module's docstring.

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
