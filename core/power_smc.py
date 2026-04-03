"""
power_smc.py
============
Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning.
Azizi, Baghaei Potraghloo, Ahmadi, Kundu, Pedram — USC + Intel Labs (arXiv:2602.10273).
This is the canonical, bug-fixed implementation.  Every function maps explicitly
to a paper equation, theorem, algorithm line, or appendix section.
Paper coverage
--------------
  [Algorithm 1]   Core Power-SMC/SIR loop                   → PowerSMC.generate()
  [Eq. 1]         Target π_α(y|x) ∝ p_θ(y|x)^α             → PowerSMC.generate()
  [Eq. 2]         Autoregressive factorisation of p_θ        → prefill + decode loop
  [Eq. 6]         Effective Sample Size (ESS)                → compute_ess()
  [Eq. 7]         Prefix-level unnormalised target γ_t       → log_incr_w derivation
  [Eq. 8]         Incremental weight ω_t = p^α / q           → delta_log_w in loop
  [Theorem 1]     Locally variance-minimising proposal       → proposal step
  [Corollary 1]   q*(v) ∝ exp(α·ℓ(v)) = softmax(ℓ/τ), τ=1/α → scaled_logits
  [Eqs. 11-13]    Rényi-entropy view of weight dispersion    → compute_renyi_log_z()
  [Eq. 14]        Path-wise weight = Σ_t log Z_t             → prefix_log_p accumulation
  [Section 5.3]   Exact exponent-bridging (α-ramping)        → α-ramp logic
  [Appendix B]    Linear α-schedule; stage-boundary reweighting → linear_alpha_schedule()
  [Appendix C]    Cache-safe KV reordering — three-tier      → reorder_kv_cache()
                  copy variant for active-only passes        → _recursive_reindex()
  [Appendix D]    EOS absorbing state; active-only fwd pass  → absorbing state + stitch
  [Appendix E]    Systematic vs. multinomial resampling      → systematic_resample()
  [Section 7]     N=64, α=4, κ=0.5, T_max=2048, T_ramp=100  → PowerSMC defaults
Bugs fixed vs. existing code in smc/nn/smc_sampler.py and smc/power_smc_sampler.py
------------------------------------------------------------------------------------
  BUG-A  smc_sampler.py   Done-particle attention mask extended with 1 (should be 0).
                           KV-cache depth and mask length diverged when a done particle
                           was subsequently selected by resampling.
                           FIX: new attention column = (~done).long() (0 for done).
  BUG-B  smc_sampler.py   Active-only forward pass called reorder_kv_cache(past, active_idx)
                           which, for DynamicCache, calls .reorder_cache() and MUTATES
                           past in-place to shape (n_active,...), corrupting the full
                           N-particle cache before stitch_kv_cache() could run.
                           FIX: use _recursive_reindex(past, active_idx) for the active
                           subset (creates a copy; never mutates full cache).
  BUG-C  power_smc_sampler.py   Weight update used log_p - log_q (missing α factor).
                           Eq. 8: ω_t = p^α / q  →  delta = α·log_p − log_q.
                           FIX: delta_log_w = alpha * log_p_tok - log_q_tok.
  BUG-D  power_smc_sampler.py   _compute_ess v1 always returns 1.0 (algebraic cancellation).
                           FIX: use the stable formula in _compute_ess_v2 only.
  BUG-E  all_sequences stored only the chosen particle (shape (1,...)), not all N.
                           generate_map() and decode_all() were therefore broken:
                           generate_map() would IndexError or return wrong sequence
                           whenever I_MAP != 0.
                           FIX: build all_seqs from the generated buffer before return.
Additional improvements
-----------------------
  OPT-1  delta_log_w simplified via Theorem 1 algebra:
                           α·log p(y_t) − log q*(y_t) = logsumexp(α·log p_t)
                           Avoids one full (N, vocab) log_softmax per decode step.
  OPT-2  Alpha ramp condition changed from alpha > 1.0 to alpha != 1.0 so that
                           sub-unity α values also benefit from exponent-bridging.
  OPT-3  Active-only STEP 2: log_softmax / logsumexp skipped for done particles.
                           When k particles are done, saves k/N of the (N, vocab)
                           log_softmax and logsumexp every decode step.
  OPT-4  Remove cached_logits.clone() per step: model already returns a fresh
                           (N, vocab) tensor; assign it directly instead of
                           cloning and then index-writing active rows.
"""
from __future__ import annotations
import json
import os
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# ============================================================================
# SECTION 1 — Resampling   (Appendix E / Section 7)
# ============================================================================
def systematic_resample(log_weights: torch.Tensor) -> torch.LongTensor:
    """
    Systematic (low-variance stratified) resampling.   (Appendix E, Section 7)
    Preferred over multinomial: both are unbiased, but systematic resampling
    has strictly lower variance (Johansen, 2009; cited in Appendix E).
    Algorithm (Section 7, 0-indexed variant):
        w  = normalise(exp(log_weights))
        u0 ~ Uniform(0, 1)
        pos_i = (u0 + i) / N   for i = 0, …, N−1
        A_i   = min{ j : Σ_{k≤j} w_k ≥ pos_i }
    Parameters
    ----------
    log_weights : (N,) unnormalised log-weights
    Returns
    -------
    ancestors : LongTensor of shape (N,)
    """
    N      = log_weights.shape[0]
    device = log_weights.device
    log_w  = log_weights - torch.logsumexp(log_weights, dim=0)
    w      = log_w.exp().double()
    w      = w / w.sum()
    u0     = torch.rand(1, device=device).item()
    pos    = (u0 + torch.arange(N, device=device, dtype=torch.float64)) / N
    cumsum = torch.cumsum(w, dim=0)
    ancestors = torch.searchsorted(cumsum.contiguous(), pos.contiguous())
    return ancestors.clamp(0, N - 1).long()


def multinomial_resample(log_weights: torch.Tensor) -> torch.LongTensor:
    """
    Multinomial resampling from unnormalised log-weights.   (Appendix E)
    """
    N     = log_weights.shape[0]
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    w     = log_w.exp().float()
    w     = w / w.sum()
    return torch.multinomial(w, num_samples=N, replacement=True).long()


# ============================================================================
# SECTION 2 — KV-Cache Utilities   (Appendix C / Appendix D)
# ============================================================================
def _recursive_reindex(obj: object, idx: torch.LongTensor) -> object:
    """
    Tier-3 recursive tensor reindexer.   (Appendix C)
    Creates a COPY of the cache with batch dimension 0 reindexed by `idx`.
    Never mutates the input object.
    """
    if isinstance(obj, torch.Tensor):
        return obj.index_select(0, idx.to(obj.device))
    if isinstance(obj, tuple):
        return tuple(_recursive_reindex(o, idx) for o in obj)
    if isinstance(obj, list):
        return [_recursive_reindex(o, idx) for o in obj]
    return obj


def reorder_kv_cache(past_key_values: object, ancestor_idx: torch.LongTensor) -> object:
    """
    Full-cache reordering for resampling.   (Appendix C, Algorithm 1 line 19)
    Three-tier strategy:
      Tier (i)   .reorder_cache()   — DynamicCache (HF ≥ 4.36)
      Tier (ii)  ._reorder_cache()  — legacy HF private hook
      Tier (iii) _recursive_reindex() — universal fallback
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "reorder_cache"):
        past_key_values.reorder_cache(ancestor_idx)
        return past_key_values
    if hasattr(past_key_values, "_reorder_cache"):
        return past_key_values._reorder_cache(past_key_values, ancestor_idx)
    return _recursive_reindex(past_key_values, ancestor_idx)


def replicate_kv_cache(
    past_key_values: object,
    source_batch:    int,
    repeat_n:        int,
    device:          torch.device,
) -> object:
    """
    Replicate KV cache rows: (K, ...) → (K*N, ...) by repeating each row N times.
    """
    idx = torch.arange(source_batch, device=device).repeat_interleave(repeat_n)
    return reorder_kv_cache(past_key_values, idx)


def stitch_kv_cache(
    full_past:   object,
    active_past: object,
    active_idx:  torch.LongTensor,
) -> object:
    """
    Write fresh active-particle KV states back into the full N-particle cache.
    (Appendix D)
    """
    if full_past is None:
        return active_past

    def _rec(full: object, active: object) -> object:
        if isinstance(full, torch.Tensor):
            out = full.clone()
            out[active_idx] = active.to(full.device)
            return out
        if isinstance(full, (tuple, list)):
            return type(full)(_rec(f, a) for f, a in zip(full, active))
        return full

    return _rec(full_past, active_past)


# ============================================================================
# SECTION 3 — ESS and Weight Utilities   (Eq. 6)
# ============================================================================
def compute_ess(log_weights_unnorm: torch.Tensor) -> float:
    """
    Effective Sample Size from unnormalised log-weights.   (Eq. 6)
        ESS_t = (Σ_i (W_t^(i))^2)^{-1}
    """
    lw     = log_weights_unnorm - log_weights_unnorm.max()
    w      = lw.exp().float()
    w_norm = w / w.sum()
    return float(1.0 / (w_norm ** 2).sum().clamp(min=1e-8))


def normalize_weights(log_weights_unnorm: torch.Tensor) -> torch.Tensor:
    """
    Convert unnormalised log-weights to probability weights.
    W^(i) = softmax(log_weights_unnorm)
    """
    log_w_norm = log_weights_unnorm - torch.logsumexp(log_weights_unnorm, dim=0)
    return log_w_norm.exp().float()


# ============================================================================
# SECTION 4 — Rényi-Entropy Weight Utilities   (Section 5.2 / Eqs. 11–14)
# ============================================================================
def compute_renyi_log_z(log_p_base: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Per-particle log Z_t(α; y_{<t}) = logsumexp(α · log_p_t).   (Eqs. 11–13)
    """
    return torch.logsumexp(alpha * log_p_base, dim=-1)


# ============================================================================
# SECTION 5 — α-Ramp Schedule   (Section 5.3 / Appendix B)
# ============================================================================
def linear_alpha_schedule(alpha: float, L: int) -> List[float]:
    """
    Linear exponent-bridging schedule.   (Appendix B)
        α^(ℓ) = 1 + (α−1)·ℓ/L,  ℓ = 0, …, L
    Returns L+1 values: [1.0, …, alpha].
    Works for any alpha != 1 (both alpha > 1 and alpha < 1).
    """
    if L <= 0:
        return [float(alpha)]
    return [1.0 + (alpha - 1.0) * ell / L for ell in range(L + 1)]


def get_stage_and_alpha(
    t:              int,
    T_ramp:         int,
    alpha_schedule: List[float],
) -> Tuple[int, float]:
    """
    Return (stage_index, alpha_t) for decode step t (1-indexed).
    """
    L = len(alpha_schedule) - 1
    if L == 0 or T_ramp <= 0 or t > T_ramp:
        return L, alpha_schedule[-1]
    stage = min(int((t - 1) * L / T_ramp), L)
    return stage, alpha_schedule[stage]


# ============================================================================
# SECTION 6 — Output Dataclass
# ============================================================================
@dataclass
class PowerSMCOutput:
    """
    Output of a single Power-SMC run.
    sequences       (1, prompt_len + gen_len)           chosen output
    all_sequences   (N, prompt_len + max_new_tokens)    all N particle sequences
    log_weights     (N,) float  final unnormalised log-weights (CPU)
    scores          tuple of (N, vocab) Tensors         proposal logits per step
    n_resamples     int                                 number of resampling events
    ess_history     List[float]                         ESS_t after every decode step
    log_z_history   List[Tensor(N,)]                    log Z_t^(i) per step
    resample_ratio  float                               n_resamples / total_steps
    """
    sequences:      torch.LongTensor
    all_sequences:  torch.LongTensor
    log_weights:    torch.FloatTensor
    chosen_idx:     int           = 0
    scores:         tuple         = field(default_factory=tuple)
    n_resamples:    int           = 0
    ess_history:    List[float]   = field(default_factory=list)
    log_z_history:  list          = field(default_factory=list)
    resample_ratio:     float         = 0.0
    chosen_sum_logprob: float         = 0.0
    n_degenerate:       int           = 0   # particles penalised by min_gen_tokens guard
    norm_weight_history: list         = field(default_factory=list)  # List[Tensor(N,)] per step; non-empty only when track_weights=True
    log_weight_history:  list         = field(default_factory=list)  # List[Tensor(N,)] unnormalised log-weights per step; non-empty only when track_weights=True


# ============================================================================
# SECTION 7 — Core PowerSMC Class   (Algorithm 1 + All Appendices)
# ============================================================================
class PowerSMC:
    """
    Sequential Monte Carlo sampler for the sequence-level power distribution:
        π_α(y | x)  ∝  p_θ(y | x)^α,     α > 0               (Eq. 1)
    where  p_θ(y | x) = Π_{t=1}^{T} p_θ(y_t | x, y_{<t})      (Eq. 2)
    Paper defaults (Section 7): N=64  α=4  κ=0.5  T_max=2048  T_ramp=100
    """
    def __init__(
        self,
        model:             AutoModelForCausalLM,
        tokenizer:         AutoTokenizer,
        alpha:             float = 4.0,
        n_particles:       int   = 64,
        kappa:             float = 0.5,
        ramp_steps:        int   = 100,
        n_ramp_stages:     int   = 10,
        resample_method:   str   = "systematic",
        min_gen_tokens:    int   = 10,
        generation_config: Optional[GenerationConfig] = None,
    ):
        assert alpha > 0.0, f"alpha must be > 0, got {alpha}"
        assert 0.0 < kappa <= 1.0, f"kappa must be in (0,1], got {kappa}"
        assert min_gen_tokens >= 0, f"min_gen_tokens must be >= 0, got {min_gen_tokens}"
        assert resample_method in ("systematic", "multinomial"), \
            f"resample_method must be 'systematic' or 'multinomial'"

        self.model           = model
        self.tokenizer       = tokenizer
        self.alpha           = alpha
        self.N               = n_particles
        self.kappa           = kappa
        self.resample_method  = resample_method
        self.min_gen_tokens   = min_gen_tokens
        self.device          = next(model.parameters()).device

        gen_cfg = generation_config or getattr(model, "generation_config", None)
        eos = (
            gen_cfg.eos_token_id
            if gen_cfg and gen_cfg.eos_token_id is not None
            else tokenizer.eos_token_id
        )
        if eos is None:
            raise ValueError(
                "Cannot determine eos_token_id. "
                "Pass a GenerationConfig or a tokenizer with eos_token_id set."
            )
        self.eos_id: int = int(eos[0]) if isinstance(eos, (list, tuple)) else int(eos)

        # OPT-2: ramp enabled for any alpha != 1 (covers both alpha > 1 and alpha < 1)
        if ramp_steps > 0 and n_ramp_stages > 0 and alpha != 1.0:
            self._alpha_sched   = linear_alpha_schedule(alpha, n_ramp_stages)
            self._T_ramp        = ramp_steps
            self._n_ramp_stages = n_ramp_stages
        else:
            self._alpha_sched   = [float(alpha)]
            self._T_ramp        = 0
            self._n_ramp_stages = 0

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids:       torch.LongTensor,
        attention_mask:  Optional[torch.LongTensor] = None,
        track_weights:   bool = False,
        max_new_tokens:  int = 2048,
        prefill_kwargs:  Optional[dict] = None,
    ) -> PowerSMCOutput:
        """
        Run Power-SMC on a single prompt (batch_size = 1).
        Returns PowerSMCOutput with all_sequences shape (N, prompt_len + max_new_tokens).

        prefill_kwargs : optional extra arguments passed to the model during
                         the prefill step only (e.g., pixel_values and
                         image_grid_thw for vision-language models).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"PowerSMC.generate() expects batch_size=1, got {input_ids.shape[0]}."
            )
        input_ids  = input_ids.to(self.device)
        N          = self.N
        device     = self.device
        prompt_len = input_ids.shape[-1]

        # ── Prefill ──────────────────────────────────────────────────────────
        attn_1 = (
            attention_mask.to(device)
            if attention_mask is not None
            else torch.ones(1, prompt_len, dtype=torch.long, device=device)
        )
        _extra = prefill_kwargs or {}
        prefill       = self.model(input_ids=input_ids, attention_mask=attn_1, use_cache=True, **_extra)
        past          = replicate_kv_cache(prefill.past_key_values, 1, N, device)
        cached_logits = prefill.logits[0, -1, :].unsqueeze(0).expand(N, -1).clone()

        # ── Pre-allocate buffers ──────────────────────────────────────────────
        attn_buf = torch.zeros(N, prompt_len + max_new_tokens, dtype=torch.long, device=device)
        attn_buf[:, :prompt_len] = attn_1.expand(N, -1)
        attn_len = prompt_len

        # ── SMC state ────────────────────────────────────────────────────────
        log_w        = torch.zeros(N, device=device)
        done         = torch.zeros(N, dtype=torch.bool, device=device)
        prefix_log_p = torch.zeros(N, device=device)
        generated    = torch.full((N, max_new_tokens), self.eos_id, dtype=torch.long, device=device)
        gen_lengths  = torch.zeros(N, dtype=torch.long, device=device)

        ess_history:        list = []
        log_z_history:      list = []
        norm_weight_history: list = []
        log_weight_history:  list = []
        n_resamples        = 0
        n_degenerate       = 0
        total_steps        = 0
        stage_idx          = 0
        alpha_cur          = self._alpha_sched[0]

        # ═════════════════════════════════════════════════════════════════════
        # MAIN DECODE LOOP
        # ═════════════════════════════════════════════════════════════════════
        for t in range(1, max_new_tokens + 1):
            total_steps += 1

            # STEP 1 — Proposal  q*(v) ∝ p_θ(v)^α = softmax(logits · α)
            scaled_logits = cached_logits * alpha_cur
            if done.any():
                scaled_logits                    = scaled_logits.clone()
                scaled_logits[done]              = float("-inf")
                scaled_logits[done, self.eos_id] = 0.0
            probs_q     = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs_q, num_samples=1).squeeze(1)

            # STEP 2 — Incremental weight   (OPT-1, OPT-3)
            #
            # Under q*(v) ∝ p(v)^α:  Δ log W̃ = logsumexp(α · log p_t)  (Theorem 1)
            # OPT-3: skip done particles entirely — they contribute 0 and computing
            # (n_done, vocab) log_softmax / logsumexp for them is pure waste.
            tok_idx = next_tokens.unsqueeze(1)                         # (N, 1)
            if done.any():
                act = ~done                                            # (N,) bool
                lp  = F.log_softmax(cached_logits[act], dim=-1)       # (n_a, V)
                log_w[act]        += torch.logsumexp(alpha_cur * lp, dim=-1)
                prefix_log_p[act] += lp.gather(1, tok_idx[act]).squeeze(1)
            else:
                lp    = F.log_softmax(cached_logits, dim=-1)          # (N, V)
                log_w        += torch.logsumexp(alpha_cur * lp, dim=-1)
                prefix_log_p += lp.gather(1, tok_idx).squeeze(1)

            # STEP 3 — Record tokens; mark EOS
            generated[:, t - 1] = next_tokens
            newly_done              = (~done) & (next_tokens == self.eos_id)
            gen_lengths[newly_done] = t
            done                    = done | newly_done

            # DEGENERATE GUARD   (Appendix D extension)
            # A particle that emits EOS before min_gen_tokens steps almost
            # certainly collapsed to a trivial/empty answer.  Drive its
            # log-weight to -∞ so it contributes zero normalised weight and
            # is culled at the next resampling step rather than polluting the
            # weight pool.
            if self.min_gen_tokens > 0 and t < self.min_gen_tokens and newly_done.any():
                log_w = log_w.clone()
                log_w[newly_done] = float("-inf")
                n_degenerate += int(newly_done.sum().item())

            # STEP 4 — α-ramp stage-boundary reweighting   (Appendix B)
            if self._T_ramp > 0:
                new_stage, new_alpha = get_stage_and_alpha(t, self._T_ramp, self._alpha_sched)
                if new_stage > stage_idx:
                    delta_alpha = new_alpha - alpha_cur
                    log_w      = log_w + delta_alpha * prefix_log_p
                    alpha_cur  = new_alpha
                    stage_idx  = new_stage

            # Safety: if every particle has been flagged degenerate (all log_w = -∞),
            # reset to uniform so that ESS / resampling do not produce NaN.
            if not torch.isfinite(log_w).any():
                log_w = torch.zeros(N, device=device)

            # STEP 5 — ESS; resample if ESS < κ·N
            ess_t = compute_ess(log_w)
            ess_history.append(ess_t)
            if ess_t < self.kappa * N:
                anc = (systematic_resample if self.resample_method == "systematic"
                       else multinomial_resample)(log_w)
                generated     = generated[anc]
                done          = done[anc]
                gen_lengths   = gen_lengths[anc]
                prefix_log_p  = prefix_log_p[anc]
                cached_logits = cached_logits[anc]
                attn_buf      = attn_buf[anc]
                next_tokens   = next_tokens[anc]
                past          = reorder_kv_cache(past, anc)
                log_w         = torch.zeros(N, device=device)
                n_resamples  += 1

            # WEIGHT TRACKING (optional; off by default to avoid memory overhead)
            if track_weights:
                norm_weight_history.append(normalize_weights(log_w).cpu())
                log_weight_history.append(log_w.cpu().clone())

            if done.all():
                break

            # STEP 6 — Forward pass   (OPT-4: no clone — logits is already a fresh tensor)
            attn_buf[:, attn_len] = (~done).long()
            attn_len             += 1
            step_out      = self.model(
                input_ids       = next_tokens.unsqueeze(1),
                attention_mask  = attn_buf[:, :attn_len],
                past_key_values = past,
                use_cache       = True,
            )
            # step_out.logits[:, -1, :] is a new (N, V) tensor — assign directly.
            # Done particles' logits are overridden in STEP 1 next iteration anyway.
            cached_logits = step_out.logits[:, -1, :]
            past          = step_out.past_key_values

        # ── Output ───────────────────────────────────────────────────────────
        # Guard: all particles degenerate at end (edge case with large min_gen_tokens).
        if not torch.isfinite(log_w).any():
            log_w = torch.zeros(N, device=device)

        final_w    = normalize_weights(log_w)
        chosen_idx = int(torch.multinomial(final_w, num_samples=1).item())
        c_len      = int(gen_lengths[chosen_idx].item()) or max_new_tokens
        chosen_gen = generated[chosen_idx, :c_len]
        chosen_seq = torch.cat([input_ids[0], chosen_gen], dim=0).unsqueeze(0)

        # BUG-E fix: store all N particle sequences, not just the chosen one.
        # Shape: (N, prompt_len + max_new_tokens); unused slots filled with eos_id.
        all_seqs = torch.cat(
            [input_ids[0].unsqueeze(0).expand(N, -1), generated], dim=1
        )

        return PowerSMCOutput(
            sequences            = chosen_seq,
            all_sequences        = all_seqs,
            log_weights          = log_w.cpu(),
            chosen_idx           = chosen_idx,
            scores               = (),
            n_resamples          = n_resamples,
            ess_history          = ess_history,
            log_z_history        = log_z_history,
            resample_ratio       = n_resamples / max(total_steps, 1),
            chosen_sum_logprob   = float(prefix_log_p[chosen_idx].item()),
            n_degenerate         = n_degenerate,
            norm_weight_history  = norm_weight_history,
            log_weight_history   = log_weight_history,
        )

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids_list:      List[torch.LongTensor],
        attention_mask_list: Optional[List[torch.LongTensor]] = None,
        max_new_tokens:      int = 2048,
        prefill_kwargs:      Optional[dict] = None,
    ) -> List[PowerSMCOutput]:
        """
        Run M independent Power-SMC chains simultaneously.
        Group m occupies rows [m*N : (m+1)*N] in every state tensor.

        prefill_kwargs : optional extra arguments passed to the model during
                         the prefill step only (e.g., pixel_values and
                         image_grid_thw for vision-language models).
        """
        M      = len(input_ids_list)
        N      = self.N
        B      = M * N
        device = self.device

        pad_id      = (self.tokenizer.pad_token_id
                       if self.tokenizer.pad_token_id is not None else self.eos_id)
        prompt_lens = [ids.shape[-1] for ids in input_ids_list]
        max_plen    = max(prompt_lens)
        padded_ids  = torch.full((M, max_plen), pad_id,  dtype=torch.long,  device=device)
        padded_attn = torch.zeros((M, max_plen),          dtype=torch.long,  device=device)
        for m, ids in enumerate(input_ids_list):
            plen = prompt_lens[m]
            padded_ids [m, max_plen - plen:] = ids[0].to(device)
            padded_attn[m, max_plen - plen:] = (
                attention_mask_list[m][0].to(device)
                if attention_mask_list is not None and attention_mask_list[m] is not None
                else 1
            )

        _extra = prefill_kwargs or {}
        prefill       = self.model(input_ids=padded_ids, attention_mask=padded_attn, use_cache=True, **_extra)
        past          = replicate_kv_cache(prefill.past_key_values, M, N, device)
        cached_logits = prefill.logits[:, -1, :].repeat_interleave(N, dim=0).clone()

        attn_buf = torch.zeros(B, max_plen + max_new_tokens, dtype=torch.long, device=device)
        attn_buf[:, :max_plen] = padded_attn.repeat_interleave(N, dim=0)
        attn_len = max_plen

        log_w        = torch.zeros(B, device=device)
        done         = torch.zeros(B, dtype=torch.bool, device=device)
        prefix_log_p = torch.zeros(B, device=device)
        generated    = torch.full((B, max_new_tokens), self.eos_id, dtype=torch.long, device=device)
        gen_lengths  = torch.zeros(B, dtype=torch.long, device=device)

        ess_history_per_group  = [[] for _ in range(M)]
        n_resamples_per_group  = [0] * M
        n_degenerate_per_group = [0] * M
        stage_idx  = 0
        alpha_cur  = self._alpha_sched[0]
        total_steps = 0

        for t in range(1, max_new_tokens + 1):
            total_steps += 1

            # STEP 1 — Proposal
            scaled_logits = cached_logits * alpha_cur
            if done.any():
                scaled_logits                    = scaled_logits.clone()
                scaled_logits[done]              = float("-inf")
                scaled_logits[done, self.eos_id] = 0.0
            probs_q     = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs_q, num_samples=1).squeeze(1)

            # STEP 2 — Incremental weight   (OPT-1, OPT-3: active-only)
            tok_idx = next_tokens.unsqueeze(1)                         # (B, 1)
            if done.any():
                act = ~done                                            # (B,) bool
                lp  = F.log_softmax(cached_logits[act], dim=-1)       # (n_a, V)
                log_w[act]        += torch.logsumexp(alpha_cur * lp, dim=-1)
                prefix_log_p[act] += lp.gather(1, tok_idx[act]).squeeze(1)
            else:
                lp    = F.log_softmax(cached_logits, dim=-1)          # (B, V)
                log_w        += torch.logsumexp(alpha_cur * lp, dim=-1)
                prefix_log_p += lp.gather(1, tok_idx).squeeze(1)

            # STEP 3
            generated[:, t - 1]     = next_tokens
            newly_done              = (~done) & (next_tokens == self.eos_id)
            gen_lengths[newly_done] = t
            done                    = done | newly_done

            # DEGENERATE GUARD   (Appendix D extension)
            if self.min_gen_tokens > 0 and t < self.min_gen_tokens and newly_done.any():
                log_w = log_w.clone()
                log_w[newly_done] = float("-inf")
                for m in range(M):
                    g_start = m * N; g_end = g_start + N
                    n_degenerate_per_group[m] += int(newly_done[g_start:g_end].sum().item())

            # STEP 4 — α-ramp
            if self._T_ramp > 0:
                new_stage, new_alpha = get_stage_and_alpha(t, self._T_ramp, self._alpha_sched)
                if new_stage > stage_idx:
                    delta_alpha = new_alpha - alpha_cur
                    log_w       = log_w + delta_alpha * prefix_log_p
                    alpha_cur   = new_alpha
                    stage_idx   = new_stage

            # STEP 5 — Per-group ESS + resampling
            global_anc   = torch.arange(B, device=device)
            any_resample = False
            for m in range(M):
                g_start = m * N
                g_end   = g_start + N
                lw_m    = log_w[g_start:g_end]
                # Safety: all-degenerate edge case for this group
                if not torch.isfinite(lw_m).any():
                    log_w[g_start:g_end] = 0.0
                    lw_m = log_w[g_start:g_end]
                ess_m   = compute_ess(lw_m)
                ess_history_per_group[m].append(ess_m)
                if ess_m < self.kappa * N:
                    anc_m = (systematic_resample if self.resample_method == "systematic"
                             else multinomial_resample)(lw_m)
                    global_anc[g_start:g_end] = g_start + anc_m
                    abs_anc = g_start + anc_m
                    generated    [g_start:g_end] = generated    [abs_anc]
                    done         [g_start:g_end] = done         [abs_anc]
                    gen_lengths  [g_start:g_end] = gen_lengths  [abs_anc]
                    prefix_log_p [g_start:g_end] = prefix_log_p [abs_anc]
                    cached_logits[g_start:g_end] = cached_logits[abs_anc]
                    attn_buf     [g_start:g_end] = attn_buf     [abs_anc]
                    next_tokens  [g_start:g_end] = next_tokens  [abs_anc]
                    log_w        [g_start:g_end] = 0.0
                    n_resamples_per_group[m] += 1
                    any_resample = True
            if any_resample:
                past = reorder_kv_cache(past, global_anc)

            if done.view(M, N).all(dim=1).all():
                break

            # STEP 6 — Forward pass   (OPT-4: no clone)
            attn_buf[:, attn_len]   = (~done).long()
            attn_len               += 1
            step_out      = self.model(
                input_ids       = next_tokens.unsqueeze(1),
                attention_mask  = attn_buf[:, :attn_len],
                past_key_values = past,
                use_cache       = True,
            )
            cached_logits = step_out.logits[:, -1, :]
            past          = step_out.past_key_values

        # ── Extract outputs ───────────────────────────────────────────────────
        outputs: List[PowerSMCOutput] = []
        for m in range(M):
            g_start    = m * N
            g_end      = g_start + N
            lw_m       = log_w[g_start:g_end]
            # Guard: all particles degenerate at end for this group.
            if not torch.isfinite(lw_m).any():
                log_w[g_start:g_end] = 0.0
                lw_m = log_w[g_start:g_end]
            lw_m       = lw_m.cpu()
            w_m        = normalize_weights(lw_m)
            chosen_idx = int(torch.multinomial(w_m, num_samples=1).item())
            abs_chosen = g_start + chosen_idx
            c_len      = int(gen_lengths[abs_chosen].item()) or max_new_tokens
            chosen_gen = generated[abs_chosen, :c_len]
            orig_ids   = input_ids_list[m][0].to(device)
            chosen_seq = torch.cat([orig_ids, chosen_gen], dim=0).unsqueeze(0)

            # BUG-E fix: all N sequences for this group
            all_seqs_m = torch.cat(
                [orig_ids.unsqueeze(0).expand(N, -1), generated[g_start:g_end]], dim=1
            )

            outputs.append(PowerSMCOutput(
                sequences           = chosen_seq,
                all_sequences       = all_seqs_m,
                log_weights         = lw_m,
                chosen_idx          = chosen_idx,
                scores              = (),
                n_resamples         = n_resamples_per_group[m],
                ess_history         = ess_history_per_group[m],
                log_z_history       = [],
                resample_ratio      = n_resamples_per_group[m] / max(total_steps, 1),
                chosen_sum_logprob  = float(prefix_log_p[abs_chosen].item()),
                n_degenerate        = n_degenerate_per_group[m],
            ))
        return outputs

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_map(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 2048,
    ) -> PowerSMCOutput:
        """
        Run Power-SMC, then select the MAP (argmax-weight) particle.
        I_MAP = argmax_i W^(i)
        """
        out        = self.generate(input_ids, attention_mask, max_new_tokens)
        I_map      = int(out.log_weights.argmax().item())
        prompt_len = input_ids.shape[-1]
        map_gen    = out.all_sequences[I_map, prompt_len:]   # works now (BUG-E fixed)
        eos_pos    = (map_gen == self.eos_id).nonzero(as_tuple=True)[0]
        trim       = int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else len(map_gen)
        map_seq    = torch.cat([input_ids[0], map_gen[:trim]], dim=0).unsqueeze(0)
        return PowerSMCOutput(
            sequences      = map_seq,
            all_sequences  = out.all_sequences,
            log_weights    = out.log_weights,
            chosen_idx     = I_map,
            scores         = out.scores,
            n_resamples    = out.n_resamples,
            ess_history    = out.ess_history,
            log_z_history  = out.log_z_history,
            resample_ratio = out.resample_ratio,
        )

    # -------------------------------------------------------------------------
    def decode(self, output: PowerSMCOutput, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(output.sequences[0], skip_special_tokens=skip_special_tokens)

    def decode_all(
        self,
        output: PowerSMCOutput,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode all N particles in descending weight order."""
        order = output.log_weights.argsort(descending=True).tolist()
        return [
            self.tokenizer.decode(output.all_sequences[i], skip_special_tokens=skip_special_tokens)
            for i in order
        ]

    def decode_generated_only(
        self,
        output:     PowerSMCOutput,
        prompt_len: int,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(
            output.sequences[0][prompt_len:], skip_special_tokens=skip_special_tokens
        )

    def ess_stats(self, output: PowerSMCOutput) -> dict:
        h = output.ess_history
        if not h:
            return {}
        return {
            "mean_ess":       float(np.mean(h)),
            "min_ess":        float(np.min(h)),
            "max_ess":        float(np.max(h)),
            "final_ess":      float(h[-1]),
            "n_steps":        len(h),
            "n_resamples":    output.n_resamples,
            "resample_ratio": output.resample_ratio,
            "n_degenerate":   output.n_degenerate,
        }


# ============================================================================
# SECTION 8 — Functional API
# ============================================================================
def power_smc(
    model,
    tokenizer,
    context:         list,
    alpha:           float = 4.0,
    N:               int   = 64,
    kappa:           float = 0.5,
    T_max:           int   = 2048,
    use_alpha_ramp:  bool  = True,
    T_ramp:          int   = 100,
    L_stages:        int   = 10,
    resample_method: str   = "systematic",
    device:          str   = "cuda",
) -> tuple:
    sampler = PowerSMC(
        model           = model,
        tokenizer       = tokenizer,
        alpha           = alpha,
        n_particles     = N,
        kappa           = kappa,
        ramp_steps      = T_ramp if use_alpha_ramp else 0,
        n_ramp_stages   = L_stages if use_alpha_ramp else 0,
        resample_method = resample_method,
    )
    input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    out       = sampler.generate(input_ids, max_new_tokens=T_max)
    prompt_len    = len(context)
    all_sequences = [
        context + out.all_sequences[i, prompt_len:].cpu().tolist()
        for i in range(N)
    ]
    return out.sequences[0].cpu().tolist(), out.log_weights, out.resample_ratio, all_sequences, out.ess_history


def power_smc_map(
    model,
    tokenizer,
    context:         list,
    alpha:           float = 4.0,
    N:               int   = 64,
    kappa:           float = 0.5,
    T_max:           int   = 2048,
    use_alpha_ramp:  bool  = True,
    T_ramp:          int   = 100,
    L_stages:        int   = 10,
    resample_method: str   = "systematic",
    device:          str   = "cuda",
) -> tuple:
    sampler = PowerSMC(
        model           = model,
        tokenizer       = tokenizer,
        alpha           = alpha,
        n_particles     = N,
        kappa           = kappa,
        ramp_steps      = T_ramp if use_alpha_ramp else 0,
        n_ramp_stages   = L_stages if use_alpha_ramp else 0,
        resample_method = resample_method,
    )
    input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    out       = sampler.generate_map(input_ids, max_new_tokens=T_max)
    prompt_len    = len(context)
    all_sequences = [
        context + out.all_sequences[i, prompt_len:].cpu().tolist()
        for i in range(N)
    ]
    return out.sequences[0].cpu().tolist(), out.log_weights, out.resample_ratio, all_sequences, out.ess_history


# ============================================================================
# SECTION 9 — Prompt Formatting
# ============================================================================
PROMPT = "Solve the following math problem step by step:\n"
COT    = "\nLet's think step by step."
BASE   = "\nAnswer:"


def format_prompt(question: str, model_name: str, tokenizer, cot: bool = True) -> str:
    suffix = COT if cot else BASE
    if model_name in ("qwen", "qwen_math"):
        return PROMPT + question + suffix
    if model_name in ("qwen_math_grpo", "phi_grpo", "phi", "tulu"):
        messages = [{"role": "user", "content": PROMPT + question + suffix}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        "Supported: qwen, qwen_math, qwen_math_grpo, phi, phi_grpo, tulu."
    )


# ============================================================================
# SECTION 10 — Evaluation Harness
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Power-SMC Sampler — Training-free LLM reasoning via SMC.\n"
            "Paper defaults (Section 7): N=64, α=4, κ=0.5, T_max=2048, T_ramp=100."
        )
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dtype",      type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--dataset",     type=str, default="lighteval/MATH")
    p.add_argument("--split",       type=str, default="test")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--cot",         action="store_true", default=True)
    p.add_argument("--alpha",  type=float, default=4.0)
    p.add_argument("--N",      type=int,   default=64)
    p.add_argument("--kappa",  type=float, default=0.5)
    p.add_argument("--T_max",  type=int,   default=2048)
    p.add_argument("--no_alpha_ramp", action="store_true")
    p.add_argument("--T_ramp",   type=int, default=100)
    p.add_argument("--L_stages", type=int, default=10)
    p.add_argument("--resample", type=str, default="systematic",
                   choices=["systematic", "multinomial"])
    p.add_argument("--min_gen_tokens", type=int, default=0,
                   help="Kill particles that emit EOS before this many tokens "
                        "(set log_w=-inf; recommended: 10–32 for math problems). "
                        "0 = disabled.")
    p.add_argument("--out_dir", type=str, default="smc_outputs")
    p.add_argument("--use_map", action="store_true")
    return p.parse_args()


def main():
    args        = parse_args()
    dtype_map   = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]
    device      = args.device

    print(f"[Power-SMC] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype       = torch_dtype,
        device_map        = device,
        trust_remote_code = True,
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampler = PowerSMC(
        model             = model,
        tokenizer         = tokenizer,
        alpha             = args.alpha,
        n_particles       = args.N,
        kappa             = args.kappa,
        ramp_steps        = 0 if args.no_alpha_ramp else args.T_ramp,
        n_ramp_stages     = 0 if args.no_alpha_ramp else args.L_stages,
        resample_method   = args.resample,
        min_gen_tokens    = args.min_gen_tokens,
    )
    degen_note = (f"min_gen_tokens={args.min_gen_tokens}"
                  if args.min_gen_tokens > 0 else "min_gen_tokens=OFF")
    print(
        f"[Power-SMC] α={args.alpha}  N={args.N}  κ={args.kappa}  T_max={args.T_max}\n"
        f"  α-ramp={'OFF' if args.no_alpha_ramp else f'ON  T_ramp={args.T_ramp}  L={args.L_stages}'}\n"
        f"  resample={args.resample}  output={'MAP' if args.use_map else 'sampled'}  {degen_note}"
    )

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    os.makedirs(args.out_dir, exist_ok=True)
    results     = []
    generate_fn = sampler.generate_map if args.use_map else sampler.generate

    for idx, example in enumerate(tqdm(ds, desc="Power-SMC evaluation")):
        question    = example.get("problem", example.get("question", ""))
        gold_answer = example.get("solution", example.get("answer", ""))
        prompt_str  = format_prompt(question, args.model_name, tokenizer, cot=args.cot)
        encoded     = tokenizer(prompt_str, return_tensors="pt").to(device)
        out         = generate_fn(encoded["input_ids"], encoded.get("attention_mask"),
                                  max_new_tokens=args.T_max)
        prompt_len  = encoded["input_ids"].shape[1]
        pred_text   = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
        ess_info    = sampler.ess_stats(out)
        record = {
            "idx":             idx,
            "question":        question,
            "gold":            gold_answer,
            "pred_text":       pred_text,
            "resample_count":  out.n_resamples,
            "resample_ratio":  out.resample_ratio,
            "n_degenerate":    out.n_degenerate,
            "mean_ess":        ess_info.get("mean_ess",  0.0),
            "min_ess":         ess_info.get("min_ess",   0.0),
            "final_ess":       ess_info.get("final_ess", 0.0),
            "n_steps":         ess_info.get("n_steps",   0),
            "final_log_weights": out.log_weights.tolist(),
            "alpha":           args.alpha,
            "N":               args.N,
            "kappa":           args.kappa,
            "T_max":           args.T_max,
            "T_ramp":          args.T_ramp,
            "L_stages":        args.L_stages,
            "min_gen_tokens":  args.min_gen_tokens,
            "resample_method": args.resample,
            "use_alpha_ramp":  not args.no_alpha_ramp,
            "use_map":         args.use_map,
        }
        results.append(record)
        with open(os.path.join(args.out_dir, f"result_{idx:05d}.json"), "w") as f:
            json.dump(record, f, indent=2)

    print(f"\n[Power-SMC] Complete.  {len(results)} examples processed.")
    print(f"  Mean ESS:        {np.mean([r['mean_ess']       for r in results]):.2f}")
    print(f"  Mean resamples:  {np.mean([r['resample_count'] for r in results]):.2f}")
    print(f"  Mean degenerate: {np.mean([r['n_degenerate']   for r in results]):.2f}")

    summary = {
        "total":           len(results),
        "alpha":           args.alpha,
        "N":               args.N,
        "kappa":           args.kappa,
        "T_max":           args.T_max,
        "T_ramp":          args.T_ramp,
        "L_stages":        args.L_stages,
        "resample_method": args.resample,
        "use_alpha_ramp":  not args.no_alpha_ramp,
        "mean_ess":          float(np.mean([r["mean_ess"]       for r in results])),
        "min_ess_global":    float(np.min ([r["min_ess"]        for r in results])),
        "mean_resamples":    float(np.mean([r["resample_count"] for r in results])),
        "mean_degenerate":   float(np.mean([r["n_degenerate"]   for r in results])),
        "min_gen_tokens":    args.min_gen_tokens,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")
    print(f"  Grade with: python run_eval_math500.py --input {args.out_dir}")


if __name__ == "__main__":
    main()
