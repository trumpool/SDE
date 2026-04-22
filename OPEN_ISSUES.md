# Open Issues / Things to Discuss

> English · [中文版](OPEN_ISSUES.zh.md)

Items marked **[BLOCKER]** must be resolved before real data; **[DESIGN]** has a
working default but deserves discussion.

## Model specification gaps

1. **[DESIGN] Architectures not specified in paper.**
   The paper does not prescribe the functional form of `μ_θ`, `J_φ`, `φ_tr`,
   `φ_vol`, or the GMM heads. Current defaults:
   - `μ_θ(z, t)`: 2-layer MLP, input `[z, t]`, hidden=64, tanh, output `d_z`
   - `J_φ(z, x)`: 2-layer MLP, input `[z, x]`, hidden=64, tanh, output `d_z`
   - `φ_tr(z)`: Linear(d_z → d_h) + tanh (then dotted with `a_tr`)
   - `φ_vol(v)`: Linear(d_v → d_h) + tanh (then dotted with `a_vol`)
   - GMM: `K` components; `π, μ, logσ²` are each a shared MLP trunk + linear head

2. **[DESIGN] Dimensions `d_z`, `d_v`.**
   Paper has `z ∈ ℝ^{d_z}`, `v ∈ ℝ_+^{d_z}` in §2.2 (same dimension!), but the
   general formulation in §2.1 says `v ∈ ℝ^{d_v}`. We implement per-coordinate
   volatility with `d_v = d_z` to match §2.2.

3. **[DESIGN] Volatility scaling `s(v)` in the mark decoder.**
   Paper just says "positive scaling function". We default to `s(v) = 1 + v`
   elementwise, so that high volatility increases mark variance monotonically and
   remains bounded below by 1. Alternative: `s(v) = softplus(v)` or `s(v) = exp(v)`.

4. **[DESIGN] Default `β`.** Paper suggests `β = 1`. We expose it as a hyperparam.

5. **[DESIGN] Default correlation `ρ`.** Paper allows `d⟨W_z,W_v⟩ = ρ dt`, no
   default given. We default `ρ = 0` (independent) and expose `ρ` as a scalar
   hyperparameter. Implemented via Cholesky on the joint `[dW_z, dW_v]` increments.

## Numerical issues

6. **[BLOCKER for real data] CIR positivity.**
   Euler-Maruyama on `dv = κ(v̄ - v)dt + ξ·sqrt(v)·dW_v` can go negative when the
   Feller condition `2κv̄ ≥ ξ²` is violated. We use **full truncation**
   (`v_+ = max(v, 0)` before the `sqrt`) per Lord/Koekkoek/van Dijk. For real data
   we should compare against (a) the absorbing scheme and (b) an exact CIR
   stepper (non-central chi-square).

7. **[DESIGN] Survival integral `∫λ_g dt`.**
   Trapezoidal rule on the discretization grid. Sampling rate is a hyperparam;
   too coarse → biased NLL, too fine → cost. Should validate grid sensitivity.

8. **[DESIGN] Event-free vs. event-inclusive discretization.**
   Algorithm 1 says "integrate on `[t_i, t_{i+1})`". We build a shared grid that
   includes the event times as breakpoints so no step straddles an event.

## Adjoint method (Algorithm 1)

9. **[DEFERRED → Phase B] `torchsde` Stratonovich adjoint + event jumps.**
   Phase A uses plain PyTorch autograd through the unrolled integrator. This is
   mathematically equivalent to the adjoint on the realized path. Phase B swaps
   in `torchsde.sdeint_adjoint` with piecewise calls over event-free intervals
   and a frozen Brownian sequence. Potential gotchas:
   - `torchsde` expects `noise_type` and `sde_type='stratonovich'` — we pass
     `sqrt(v(t))⊙dW` correctly in Stratonovich by noting that for CIR with
     state-dependent diffusion the Ito-Stratonovich correction must be baked
     into the drift (or use `sde_type='ito'` and keep the code Ito-consistent).

10. **[DEFERRED → Phase B] O(1)-memory reverse pass with event jumps.**
    The paper's adjoint jump formula (eq 3.20–3.24) needs the left-limit state
    `s(t_i^-)` at each event. In Phase B we'll checkpoint only the left-limit
    states at event times and reconstruct between events via reverse SDE — this
    is the non-trivial engineering piece.

## Text-valued marks (§4)

11. **[RESOLVED 2026-04-20] RoBERTa-wwm-ext encoding wired.**
    `src/text_encoder.py` + `scripts/encode_weibo.py` encode posts with
    `hfl/chinese-roberta-wwm-ext` at `max_length=256`, `batch_size=32`,
    storing `[CLS]` as fp16 in `data/encoded/<month>_cls.pt`. `NeuralSVMPP`
    owns a learnable `Linear(768, d_x=32)` projector per paper §4, applied
    automatically inside `forward_sequence` (and in `compute_loss` when the
    decoder scores marks). Throughput on Apple M4 MPS: ~48 texts/s
    (~16 min for one monthly CSV).

## Training / data

12. **[BLOCKER for real data] Batching of variable-length sequences.**
    Current implementation handles one sequence at a time (train_weibo.py loops
    serially over sequences per step). Real data at scale will need packed
    sequences with a mask in the loss. For 2019-12 with 90 sequences this is
    fine on CPU (~7 s/step).

13. **[DESIGN] Initial conditions `s(t_0)`.**
    We default `z(t_0) = 0`, `v(t_0) = v̄` (mean-revert point). Could learn them.

14. **[DESIGN] ``β`` balancing timing vs. marks with BERT features.**
    With 32-d BERT-projected marks the GMM log-likelihood dominates the
    timing term at ``β=1``. Empirically on 2019-12 we used ``β=0.1`` and got
    a healthy decrease in both terms. Open question: should we fix a
    principled default (e.g. per-dimension normalization so the mark term is
    comparable in scale to the timing term) or keep ``β`` as a tuning knob?

15. **[DESIGN] Short-window "burst" sequences.**
    14/90 sequences in 2019-12 have horizon ``T < 0.1 day`` — users who
    posted 5+ tweets within minutes (likely repost cascades or bots). The
    SDE grid still handles them (``n_sub = max(1, ceil(span/dt))``) but the
    MPP semantics may not fit: within a 35-second burst, the latent
    jump-diffusion barely evolves. Options: (a) minimum horizon filter,
    (b) separate clustered-event likelihood, (c) accept and move on — the
    sample is small (~15%).

16. **[DEFERRED → Phase B] MPS unrolled-integrator performance.**
    MPS was much slower than CPU for the Euler-Maruyama rollout with many
    tiny ops; a 90-seq run hung for 10+ minutes before I killed it. CUDA
    should behave better (fewer kernel-launch pessimizations), but if not,
    the fix is to batch across sequences along the grid. Log this for the
    Linux move.
