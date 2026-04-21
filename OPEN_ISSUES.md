# Open Issues / Things to Discuss

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
    Current implementation handles one sequence at a time (or a same-length batch
    via padding). Real data will need packed sequences with a mask in the loss.

13. **[DESIGN] Initial conditions `s(t_0)`.**
    We default `z(t_0) = 0`, `v(t_0) = v̄` (mean-revert point). Could learn them.
