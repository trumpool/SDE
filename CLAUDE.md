# Project handoff — for the next Claude

> Audience: a Claude Code instance picking up this repo, probably on a Linux
> GPU box. If you are that Claude, read this file top-to-bottom before
> touching code. If you are a human: most of this is still accurate but the
> tone is aimed at Claude.

---

## 1. What this project is

The paper [main.tex](main.tex) proposes a **Neural Marked Point Process with
Latent Stochastic Volatility** — a continuous-time latent jump-diffusion MPP
where:

- `z(t)` is a neural latent state,
- `v(t)` is a **positive CIR-type volatility** that modulates both the
  diffusion of `z` and the variance of the mark distribution,
- observed events cause **instantaneous jumps in `z`** (but not `v`),
- the conditional intensity is **dual-channel**:
  `λ_g = Softplus(a_tr·φ_tr(z) + a_vol·φ_vol(v) + b)`,
- marks are decoded by a **volatility-modulated Gaussian mixture**.

The training algorithm is the forward-backward **adjoint method** laid out in
Proposition 3.1 + Algorithm 1 of the paper (pathwise Stratonovich adjoint,
event-time jump updates). The appendix contains the rigorous proof.

### The user's goals, in order

1. **Phase A (DONE, this is where we are):** PyTorch autograd through an
   unrolled Euler-Maruyama integrator. Mathematically equivalent to the
   paper's adjoint on the realized path; easier to debug. Runs on Mac CPU.
2. **Phase B (TODO):** `torchsde.sdeint_adjoint` implementing Algorithm 1
   literally with O(1)-memory pathwise adjoint. Linux GPU target.
3. **Empirical study (IN PROGRESS):** Weibo-COV 2.0 — Chinese COVID tweets
   as a marked point process. Per-user timelines are sequences; tweet text
   is the mark, encoded by RoBERTa-wwm-ext per paper §4.

---

## 2. First five minutes: orient yourself

```bash
# You just cloned. Bring the env up:
bash setup.sh --quick           # venv + deps + tests only (~30 s)

# Then, if you want to verify end-to-end with real data:
bash setup.sh                   # adds 60 MB download + 20 min BERT encode + 10-step train
```

After `setup.sh --quick` you can run everything manually from the venv:

```bash
source .venv/bin/activate
python -m pytest tests/ -v      # should be 21/21 green
python scripts/train_small.py   # synthetic-only, no downloads, ~10 s
```

### Repo map

```
SDE/
├── main.tex                    # THE PAPER. Read §2 + §3 + Algorithm 1 first.
├── setup.sh                    # one-shot bootstrap (idempotent)
├── CLAUDE.md                   # this file
├── README.md                   # user-facing quickstart
├── OPEN_ISSUES.md              # ← READ AFTER THIS FILE. Design decisions
│                               #   with "why", numbered §1–§16.
├── requirements.txt
├── .gitignore                  # ignores .venv/ and data/ entirely
│
├── src/                        # library, pure torch modules
│   ├── nets.py                 # DriftNet, JumpNet, DualChannelIntensity,
│   │                           #   GMMMarkDecoder — all neural pieces
│   ├── sde.py                  # Euler-Maruyama integrator, CIR full-truncation,
│   │                           #   correlated Brownians, event-aware time grid,
│   │                           #   trapezoidal survival integral
│   ├── model.py                # NeuralSVMPP — owns all nets + CIR params
│   │                           #   (κ, v̄, ξ softplus-parameterized; ρ tanh-bounded;
│   │                           #   optional learnable Linear(bert_dim → d_x))
│   ├── loss.py                 # MPP NLL = −Σ log λ_g(t_i) − β·Σ log p(x_i) + ∫λ dt
│   ├── synth.py                # hand-crafted ground-truth generator (thinning)
│   ├── weibo_data.py           # Weibo-COV 2.0 CSV → per-user sequences with
│   │                           #   ±0.5s timestamp jitter + optional BERT cache
│   ├── text_encoder.py         # WeiboTextEncoder — HuggingFace RoBERTa-wwm-ext
│   └── utils.py                # seeding, device pick
│
├── scripts/
│   ├── make_synth.py           # save a synthetic dataset
│   ├── train_small.py          # train on synthetic — the smoke test
│   ├── encode_weibo.py         # offline BERT [CLS] → cache .pt
│   └── train_weibo.py          # train on real Weibo data (optionally with BERT cache)
│
├── tests/                      # pytest; CSV-dependent tests skipif missing
│   ├── test_nets.py
│   ├── test_sde.py             # CIR mean reversion, correlation, grid
│   ├── test_loss.py
│   ├── test_gradient.py        # autograd vs finite-difference (critical)
│   ├── test_weibo_data.py
│   └── test_bert_integration.py
│
└── data/                       # gitignored, reproduce via setup.sh or scripts
    ├── raw/                    # 2019-12.csv (60 MB) after download
    └── encoded/                # 2019-12_cls.pt (70 MB) after encoding
```

---

## 3. What's already working (do not re-derive)

### 3.1 Math is right

The single most important test is
[test_finite_difference_vs_autograd_on_log_kappa](tests/test_gradient.py#L48):
the gradient of the total loss w.r.t. the raw log-κ parameter matches finite
difference to better than 0.5% relative error. This parameter's gradient
flows through **every** SDE step, so passing this test means the forward
Euler-Maruyama + trapezoidal survival + GMM decoder + autograd backward
chain is all self-consistent.

Do not modify the integrator or the loss without re-running this test.

### 3.2 Weibo-COV 2.0 empirical run

We have trained on one month of Chinese Weibo posts, encoded to 768-d
[CLS] via `hfl/chinese-roberta-wwm-ext`, projected through a learnable
`Linear(768, 32)` (paper's `W_p h^{[CLS]} + b_p`, §4) into the MPP model.

Concrete numbers from 2019-12:
- **Raw CSV:** 47,134 tweets / 43,812 unique users
- **After `min_length=5` filter:** 90 sequences / 728 events
- **Encoding:** 20.7 min on Apple M4 MPS, `batch_size=32`, `max_length=256`
- **Training sanity (30 steps, CPU, β=0.1):** loss/ev 5.64 → 0.06

### 3.3 Git state

Remote: `git@github.com:trumpool/SDE.git`, branch `main`, up-to-date with
`origin/main`. Last five commits:

```
959cb73 One-shot bootstrap: setup.sh
3bf6ca9 OPEN_ISSUES: log findings from the BERT sanity run
3ea70ee Wire RoBERTa-wwm-ext text marks (paper §4)
983ad54 Real-data pipeline for Weibo-COV 2.0
4ec450c Initial implementation of neural MPP with latent stochastic volatility
```

---

## 4. Design decisions that are NOT in the paper

These are documented with rationale in [OPEN_ISSUES.md](OPEN_ISSUES.md) —
read that file next. Highlights:

| # | Decision | Default | Why |
|---|---|---|---|
| 1 | Sub-net architectures | 2-layer tanh MLPs, hidden=64 | paper silent |
| 3 | `s(v)` scaling in mark decoder | `1 + v` elementwise | positive, ≥1, monotone |
| 4 | `β` in the loss | 1.0 default, 0.1 used with BERT marks | GMM on 32-d dominates timing if β=1 |
| 5 | `ρ` correlation of `W_z, W_v` | 0 (learnable) | paper allows, no default given |
| 6 | CIR positivity | full truncation (Lord/Koekkoek/van Dijk) | avoids NaN from `sqrt(v<0)` |
| — | Ito vs Stratonovich | Ito for the specific model | Euler-Maruyama + CIR convention |

---

## 5. Known issues / gotchas

1. **MPS is slower than CPU** for the unrolled integrator on Mac. Many tiny
   per-step ops hit MPS kernel-launch overhead. CUDA should not have this
   problem. If you see a training run hung at 0 steps for minutes, kill it
   and retry with `--device cpu`.

2. **Python stdout buffering in background tasks:** `python foo.py > log.txt`
   will not flush until process exits. If you need live progress, use
   `python -u foo.py > log.txt` **or** write logs through `print(..., flush=True)`.
   Do NOT pipe to `tee` when the target directory may not exist — a failed
   `tee` silently breaks the pipeline with SIGPIPE.

3. **`torchsde` + event jumps needs careful engineering.** The paper's
   adjoint formula for event times (eq 3.20–3.24) needs the **left-limit**
   state `s(t_i^-)`. You'll have to call `sdeint_adjoint` piecewise on
   event-free intervals and apply the event map by hand. Brownian-path
   determinism is the #1 pitfall — checkpoint RNG state and replay it.

4. **Burst users:** ~15% of filtered Weibo sequences have horizon
   `T < 0.1 day` (users who posted 5+ tweets within minutes — bots or
   repost cascades). See OPEN_ISSUES §15.

5. **Variable-length batching is NOT implemented.** Current loop is a
   Python for-loop over sequences per training step. For small data this
   is fine; for scale, needs packed sequences with masking. OPEN_ISSUES §12.

6. **28K-param model on 728 events ⇒ overfits fast.** The BERT sanity run
   reached `nll_mark ≈ −7.6` (density > 1) in 30 steps. Before reading
   further into "convergence" you need a train/val split; the current
   script does not have one.

---

## 6. What to do next (prioritized; the user may redirect)

### P0 — make Phase A results trustworthy (~1 day)

1. **Add train/val split to `train_weibo.py`.** Currently no validation.
   Hold out say 20% of sequences; compute val NLL per step; log both. Be
   careful: "per-event" normalization is what lets cross-sequence comparison
   be meaningful.
2. **β sweep** ∈ {0.01, 0.1, 0.5, 1.0}. Plot both nll_time/ev and
   nll_mark/ev across β. The right β is the one where the two terms have
   comparable magnitudes AND val NLL is best.
3. **Longer training** — 500+ steps with gradient-norm clip already on.
   Confirm CIR params (κ, v̄, ξ, ρ) actually move. They barely moved in
   30 steps.

### P1 — more data (~1 day)

4. **Download more months.** The Drive folder
   `1xozXd2cKw0pPvWgppvGbaIhpQzS8o-bp` has 2019-12 → 2020-12 as monthly
   CSVs plus `user.csv`. IDs are hard-coded in
   [scripts/encode_weibo.py docstring-adjacent usage](scripts/encode_weibo.py)
   or can be re-listed via `gdown --folder --no-download`. COVID peak is
   2020-02/03.
5. **Multi-month loader.** Extend `weibo_data.py` to accept a list of CSV
   paths + matching cache paths, concatenating sequences.

### P2 — Linux move + Phase B (~1 week)

6. **Linux bring-up.** `bash setup.sh --quick` should work out of the box
   (we tested on Mac; Linux is lower-friction for torch). If CUDA is
   available `--device cuda` in `train_weibo.py` is the only switch.
7. **Phase B — `torchsde.sdeint_adjoint`.** This is the paper's Algorithm 1
   properly implemented, not our autograd-through-unrolled shortcut. Core
   work:
   - Express the SDE as a `torchsde.SDEIto` (or `SDEStratonovich`) class
     with `f` (drift) and `g` (diffusion) methods.
   - Integrate between events with `sdeint_adjoint(..., adjoint_method="milstein")`.
   - Apply the event map manually between calls.
   - Freeze Brownian noise seed for reverse-mode correctness.
   - Verify the adjoint-method gradient matches our autograd gradient on a
     small synthetic example — this is the "do I trust Phase B" test and it
     MUST be a test, not an ad-hoc script.
8. **Variable-length batching.** Pack sequences with masking on events and
   on the survival-integral grid.

### P3 — methodological exploration (flexible)

9. **Ablations** (paper §2.3 motivates this directly): zero-out `a_vol`
   (no volatility channel) vs zero-out `a_tr` (no trend channel). NLL
   difference quantifies the predictive value of volatility.
10. **Diagnostics / inspection script.** Given a trained model, plot:
    - learned κ, v̄, ξ, ρ over training
    - fitted intensity λ(t) vs event points for a few held-out sequences
    - mark reconstruction quality (decode `[CLS]` back and check cosine
      similarity with raw `[CLS]`)

---

## 7. Things NOT to do

- **Don't `rm -rf data/`.** It's gitignored but holds real downloaded data
  + encoded cache. If you need to clear, delete specific subdirs.
- **Don't edit `main.tex`.** That's the paper, authored by four people.
  If your work changes an assumption stated in the paper, record it in
  `OPEN_ISSUES.md` and ask the user.
- **Don't silence CIR-positivity warnings.** If a CIR step goes negative
  at scale, something is wrong with `(κ, v̄, ξ)` — investigate, don't mask.
- **Don't rewrite the commit author.** The current author
  (`Xiaoyang Wan <wan3@seas.upenn.edu>`) is one of the paper co-authors.
  The user's email in Claude's context is different; do not assume they
  are the same person.
- **Don't push changes to `main` without running `pytest tests/`.**
  Especially `test_gradient.py` — it's slow-ish (2–4 s) but catches math
  regressions that are otherwise invisible.

---

## 8. Collaborating with the user

- **Language:** the user writes primarily in Chinese. Reply in Chinese
  unless they switch.
- **Style:** they like an environment/state check and a short design
  proposal BEFORE any non-trivial coding. Once they say "可以" or "写",
  proceed without further check-ins unless you hit a real blocker.
- **Memory:** there is an auto memory system at
  `~/.claude/projects/-Users-jeffvan-Desktop-neural-SDE/memory/` on the
  Mac where this repo was started. On a fresh machine you'll start without
  it — this file plus `OPEN_ISSUES.md` is meant to be the substitute.

---

## 9. Quick-reference command cheatsheet

```bash
# env
bash setup.sh --quick           # full deps + tests
source .venv/bin/activate

# synthetic sanity
python scripts/train_small.py
python scripts/make_synth.py

# real data — full loop
bash setup.sh                   # downloads + encodes + 10-step train
python scripts/train_weibo.py \
    --bert-cache data/encoded/2019-12_cls.pt \
    --max-seqs 90 --steps 100 --beta 0.1 --device cpu

# download another month
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=<ID>', 'data/raw/<NAME>.csv', quiet=False)"

# encode another month
python scripts/encode_weibo.py --csv data/raw/<NAME>.csv

# tests
python -m pytest tests/ -v
python -m pytest tests/test_gradient.py -v   # math-correctness suite

# git
git status
git log --oneline
git push                        # upstream already tracked
```

### Drive folder IDs for the monthly CSVs

Parent folder: `1xozXd2cKw0pPvWgppvGbaIhpQzS8o-bp`

| File | Drive ID |
|---|---|
| 2019-12.csv | `1dakfZtBG0itJTHc3_544t2sPHplTpqW_` |
| 2020-01.csv | `1dBkDMm7dQmup57mkXsxUpWr2WAP4Diiu` |
| 2020-02.csv | `1_UCmSb5YvhK9442mbJkX597CUJHok3P-` |
| 2020-03.csv | `13kzCaSmxEV-l9iyYA6CyQdoAE4OTMsIm` |
| 2020-04.csv | `1KcRBa9trYsBL4Lahsrzy83hSp0ui4sKq` |
| 2020-05.csv | `1nO8r4VbA3yLMVimJAMv_BaD_lopFrG0f` |
| 2020-06.csv | `1PHoOWq8rjZUWX7wWQimjH6FQPNiBek4E` |
| 2020-07.csv | `13YHT90exPJjVm-qzhY4NsQKelHdKGCD4` |
| 2020-08.csv | `1iXNRU3d_wXQPN-Jw8a_eU8NnPMZsMyDU` |
| 2020-09.csv | `1buduaHrO0UI-Y8NR_CMKA-hh9TgOCjFl` |
| 2020-10.csv | `1h3kUbg9c4VMtc4T6tcbzTu7EqiJi_gpe` |
| 2020-11.csv | `16F7iROP5uvSGhZG9SnkrLFUQiGu3wmsR` |
| 2020-12.csv | `1RfzEfrWjBCETN5k-GJdl8UqCy3ThxHvY` |
| user.csv    | `1IIOHBxc9yVbhttesAeKflkA1ce4ZUGyT` |

---

## 10. If something breaks, start here

- `pytest tests/ -v` — if red, the code is inconsistent with its own tests.
- `python scripts/train_small.py` — if this fails, the synthetic pipeline
  itself broke; BERT / Weibo plumbing is downstream.
- `python -c "import torch; print(torch.__version__)"` — if torch missing,
  the venv activation failed.
- For a BERT-specific failure, check
  `~/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext/` — if
  partial, delete and re-download.
- Git push refused? Check `git remote -v` matches
  `git@github.com:trumpool/SDE.git`; ensure SSH key is on GitHub account
  `trumpool`.

---

*Last updated: 2026-04-21, commit `959cb73` (plus this doc once committed).*
