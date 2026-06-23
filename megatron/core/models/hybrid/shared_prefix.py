# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime context for shared-prefix ("tree") packing through a HybridStack.

A GRPO group of G completions shares one prompt P. Instead of running the model on the G
duplicated ``[P + C_i]`` sequences (re-scanning / re-attending P every time), the stack runs a
SINGLE forward over the packed sequence ``[P, C_1, ..., C_G]`` (P stored once):

  * ATTENTION layers honor the *tree mask* (each token attends causally to the prefix and to its
    own completion branch, never to a sibling branch) with position-aware RoPE (each completion
    continues from the prefix's positions). Mask + RoPE express the sharing -- no module surgery.
  * MAMBA (and other recurrent) layers cannot express branch isolation with a mask -- a scan is
    sequential -- so they fork INTERNALLY: scan P once, then scan each completion C_i from P's
    captured conv + SSM end-state (``MambaMixer.fork_segment``), and write the per-segment outputs
    back into the packed positions.

``SharedPrefixContext`` carries the packed layout (prefix length + per-completion lengths) so a
stateful layer can slice the packed activation into its P / C_i segments. It is threaded through
``forward`` like ``inference_context`` and is a pure-Python holder (no model imports) so it can be
shared between ``megatron/core/ssm`` (MambaLayer) and ``megatron/core/models/hybrid`` (HybridStack)
without an import cycle.
"""

import logging
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAVE_FLEX_ATTENTION = True
except ImportError:  # torch < 2.5
    create_block_mask = None
    flex_attention = None
    HAVE_FLEX_ATTENTION = False

# Lazily torch.compile'd flex_attention. Compilation is keyed on tensor shapes, so distinct packed
# lengths recompile (and cache) -- acceptable since bin shapes recur within a run. The compiled
# kernel skips fully-masked (sibling-branch) blocks, which is the whole speedup over the dense
# [T,T]-mask SDPA path (measured ~11x faster, ~5x faster than the un-shared baseline @ Lp5247/G16).
_COMPILED_FLEX = None


def _get_compiled_flex():
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention)
    return _COMPILED_FLEX


# --- Lightweight shared-prefix attention profiling -------------------------------------------
# Attributes the SP attention cost into mask-build / flex-kernel and exposes a *recompile proxy*
# (count of DISTINCT packed lengths -- torch.compile is shape-keyed, so each new length is a fresh
# compile). Call counts + unique-shape tracking are free (no sync) and always on. Per-call CUDA-
# event timing forces a sync, so it is gated behind NRL_SP_PROFILE=1. ``maybe_log_sp_profile``
# emits a summary line every ``NRL_SP_PROFILE_EVERY`` flex calls (default 200).
_SP_PROFILE_TIMING = os.environ.get("NRL_SP_PROFILE", "0") not in ("0", "", "false", "False")
_SP_PROFILE_EVERY = int(os.environ.get("NRL_SP_PROFILE_EVERY", "200"))
_SP_PROFILE = {
    "flex_calls": 0,
    "mask_builds": 0,
    "flex_shapes": set(),  # distinct packed lengths -> recompile proxy
    "mask_ms": 0.0,        # only populated when _SP_PROFILE_TIMING
    "flex_ms": 0.0,
}


def sp_profile_summary() -> dict:
    """Snapshot of shared-prefix attention counters (safe to call any time)."""
    p = _SP_PROFILE
    calls = p["flex_calls"]
    n_shapes = len(p["flex_shapes"])
    return {
        "shared_prefix/flex_calls": calls,
        "shared_prefix/mask_builds": p["mask_builds"],
        "shared_prefix/unique_flex_shapes": n_shapes,
        # ~1.0 means almost every call is a new shape (recompile-bound); ~0 means shapes recur.
        "shared_prefix/recompile_proxy": (n_shapes / calls) if calls else 0.0,
        "shared_prefix/mask_ms_total": p["mask_ms"],
        "shared_prefix/flex_ms_total": p["flex_ms"],
        "shared_prefix/timing_enabled": _SP_PROFILE_TIMING,
    }


def maybe_log_sp_profile() -> None:
    if _SP_PROFILE["flex_calls"] % _SP_PROFILE_EVERY == 0 and _SP_PROFILE["flex_calls"] > 0:
        logger.info("[shared_prefix profile] %s", sp_profile_summary())


def build_tree_segment_ids(
    prefix_len: int, completion_lens: List[int], device, padded_len: Optional[int] = None
) -> torch.Tensor:
    """``[padded_len or total_len]`` long: 0 for prefix tokens, ``i+1`` for completion ``i``'s
    tokens, and 0 for any trailing pad positions ``[total_len:padded_len]``.

    Trailing pad positions (used to make the packed length divisible by the tensor-parallel size
    for sequence parallelism) keep segment 0: under the causal rule no real token ever attends them
    (they sit after every real token), and pad queries attend only the prefix -- their outputs are
    discarded (never in ``comp_positions``).
    """
    total = int(prefix_len) + sum(int(x) for x in completion_lens)
    n = int(padded_len) if padded_len is not None else total
    seg = torch.zeros(n, dtype=torch.long, device=device)
    cursor = int(prefix_len)
    for i, lc in enumerate(completion_lens):
        seg[cursor : cursor + int(lc)] = i + 1
        cursor += int(lc)
    return seg


def build_tree_block_mask(
    prefix_len: int, completion_lens: List[int], device, padded_len: Optional[int] = None
):
    """FlexAttention ``BlockMask`` for the shared-prefix tree: a query attends a key iff the key is
    causally before it AND (the key is in the prefix OR in the same completion branch). Sibling
    branches never attend to each other. Returns ``None`` if FlexAttention is unavailable.

    ``padded_len`` extends the mask to a tensor-parallel-divisible packed length (for sequence
    parallelism); the trailing pad positions are causally harmless (see ``build_tree_segment_ids``).

    This is the kernel realization of ``shared_prefix_packing.dense_tree_mask`` -- built here from
    just (prefix_len, completion_lens) so ``megatron.core`` needs no ``megatron.rl`` import.
    """
    if not HAVE_FLEX_ATTENTION:
        return None
    seg = build_tree_segment_ids(prefix_len, completion_lens, device, padded_len=padded_len)
    total = int(seg.numel())

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        k_is_prefix = seg[kv_idx] == 0
        same_branch = seg[kv_idx] == seg[q_idx]
        return causal & (k_is_prefix | same_branch)

    _SP_PROFILE["mask_builds"] += 1
    if _SP_PROFILE_TIMING and device is not None and str(device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bm = create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=device)
        end.record()
        torch.cuda.synchronize()
        _SP_PROFILE["mask_ms"] += start.elapsed_time(end)
        return bm
    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=device)


def flex_tree_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, block_mask, scale=None
) -> torch.Tensor:
    """Run the tree-masked attention via (compiled) FlexAttention.

    ``query/key/value`` are in Megatron core-attention layout ``[sq, b, n_heads, head_dim]``
    (key/value may have fewer heads for GQA). Returns the context in ``[sq, b, n_heads*head_dim]``,
    matching what ``core_attention`` returns to ``linear_proj``.
    """
    q = query.permute(1, 2, 0, 3)  # [b, np, sq, hn]
    k = key.permute(1, 2, 0, 3)  # [b, ng, sk, hn]
    v = value.permute(1, 2, 0, 3)
    enable_gqa = q.shape[1] != k.shape[1]
    sq, b = query.shape[0], query.shape[1]

    _SP_PROFILE["flex_calls"] += 1
    _SP_PROFILE["flex_shapes"].add(int(sq))  # distinct packed lengths -> recompile proxy
    if _SP_PROFILE_TIMING and query.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
        end.record()
        torch.cuda.synchronize()
        _SP_PROFILE["flex_ms"] += start.elapsed_time(end)
    else:
        out = _get_compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa, scale=scale)
    maybe_log_sp_profile()
    return out.permute(2, 0, 1, 3).reshape(sq, b, -1).contiguous()  # [sq, b, np*hn]


# --- Backend selection: FlexAttention (default) vs flash-composed -----------------------------
# FlexAttention's *forward* is at parity with flash, but its generated-Triton *backward* is
# ~4x slower than flash's hand-tuned CUDA backward at equal FLOPs (worse with GQA), making the
# shared-prefix training step backward-bound. ``flash_composed`` rebuilds the same tree attention
# from flash kernels + an online-softmax merge (see ``flash_composed_tree_attention``), reaching
# ~flash-class fwd AND bwd. Switch with NRL_SP_ATTENTION_BACKEND=flash_composed (default: flex).
# Dense attention path only; the hybrid/Mamba path always uses flex (no ``_sp_layout`` set).
_SP_ATTENTION_BACKEND = os.environ.get("NRL_SP_ATTENTION_BACKEND", "flex").lower()
_SP_FALLBACK_WARNED = False


def sp_attention_backend() -> str:
    """Selected shared-prefix attention backend: ``"flex"`` or ``"flash_composed"``."""
    return _SP_ATTENTION_BACKEND


def flash_composed_tree_attention(query, key, value, prefix_len, completion_lens, scale=None):
    """Tree attention rebuilt from flash kernels (no FlexAttention BlockMask).

    A completion token attends {all prefix} ∪ {its own branch, causally}. That union splits into
    two flash-doable attentions whose softmaxes are merged by log-sum-exp (LSE):
      (a) completions -> prefix     : NON-causal flash (prefix wholly precedes the completions)
      (b) completions -> own branch : block-diagonal causal flash_varlen over [C_1..C_G]
    Prefix rows come from a causal flash over [P]. Trailing pad positions (TP-divisibility) get
    zero outputs -- they are never read downstream (not in ``comp_positions``). Flash handles GQA
    natively for fwd AND bwd, so this is ~flash-class where FlexAttention's backward is not.

    ``query`` is ``[sq, b, np, hn]`` and ``key``/``value`` ``[sq, b, ng, hn]`` (core-attention
    layout); ``b`` must be 1 (a single packed sequence). Returns ``[sq, b, np*hn]`` to match
    ``flex_tree_attention`` / the static ``core_attention`` output. Same default scale
    (1/sqrt(hn)) as the flex path when ``scale is None``.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    P = int(prefix_len)
    Cs = [int(c) for c in completion_lens]
    ncomp = sum(Cs)
    total = P + ncomp
    assert sq >= total, f"packed length {sq} < Lp+sum(Lc) {total}"

    q = query[:, 0]
    k = key[:, 0]
    v = value[:, 0]  # [sq, n, hn]
    qf, kf, vf = q[:P], k[:P], v[:P]
    qc, kc, vc = q[P:total], k[P:total], v[P:total]

    # (a) completions attend the full prefix (non-causal cross-attention).
    oa, lse_a, _ = flash_attn_func(
        qc.unsqueeze(0), kf.unsqueeze(0), vf.unsqueeze(0),
        softmax_scale=scale, causal=False, return_attn_probs=True,
    )
    oa = oa.squeeze(0)        # [ncomp, np, hn]
    lse_a = lse_a.squeeze(0)  # [np, ncomp]

    # (b) completions attend their own branch, causally (block-diagonal varlen).
    cu = [0]
    for c in Cs:
        cu.append(cu[-1] + c)
    cu = torch.tensor(cu, dtype=torch.int32, device=q.device)
    max_c = max(Cs)
    ob, lse_b, _ = flash_attn_varlen_func(
        qc, kc, vc, cu, cu, max_c, max_c,
        softmax_scale=scale, causal=True, return_attn_probs=True,
    )                          # ob [ncomp, np, hn], lse_b [np, ncomp]

    # online-softmax (LSE) merge -> exact softmax over the union of attended keys.
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    denom = wa + wb
    wa = (wa / denom).transpose(0, 1).unsqueeze(-1)  # [ncomp, np, 1]
    wb = (wb / denom).transpose(0, 1).unsqueeze(-1)
    # LSE (hence wa/wb) is fp32; recast the merged output back to the input dtype so the packed
    # output matches the flex path and downstream Linear layers (bf16) do not see an fp32 input.
    o_comp = (oa * wa + ob * wb).to(query.dtype)       # [ncomp, np, hn]

    # prefix rows: causal self-attention over [P].
    cu_p = torch.tensor([0, P], dtype=torch.int32, device=q.device)
    o_pre = flash_attn_varlen_func(
        qf, kf, vf, cu_p, cu_p, P, P, softmax_scale=scale, causal=True,
    )                          # [P, np, hn]

    out = torch.cat([o_pre, o_comp], dim=0)            # [total, np, hn]
    if sq > total:  # trailing pad positions; outputs discarded downstream.
        out = torch.cat([out, out.new_zeros(sq - total, np_, hn)], dim=0)
    return out.reshape(sq, 1, np_ * hn).contiguous()   # [sq, b, np*hn]


def flash_composed_forest_attention(query, key, value, forest, scale=None):
    """Forest (multi-group / Case-1) shared-prefix attention.

    ``forest`` is a list of ``(token_offset, prefix_len, completion_lens)``, one per group packed
    into this bin. Groups are block-diagonal (a token never attends another group), so the forest
    forward is just the validated single-group :func:`flash_composed_tree_attention` applied to
    each group's contiguous slice, written back into the packed output. Trailing pad positions
    (beyond the last group) are zero. (A single multi-segment ``cu_seqlens`` call would fuse the
    groups into one kernel launch -- a later perf optimization; this is the correctness-first form.

    Generalizes the depth-1 forest now; arbitrary-depth trees (Case 2) extend the per-group call,
    not this loop.)
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "shared-prefix packing uses a single packed sequence (b == 1)"
    out = query.new_zeros(sq, np_ * hn)
    for off, prefix_len, completion_lens in forest:
        total = int(prefix_len) + sum(int(c) for c in completion_lens)
        o = flash_composed_tree_attention(
            query[off : off + total], key[off : off + total], value[off : off + total],
            prefix_len, completion_lens, scale=scale,
        )  # [total, 1, np*hn]
        out[off : off + total] = o.reshape(total, np_ * hn)
    return out.reshape(sq, 1, np_ * hn).contiguous()


def run_shared_prefix_attention(query, key, value, *, block_mask, layout=None, forest=None, scale=None):
    """Dispatch shared-prefix attention to the selected backend.

    ``forest`` (a list of per-group ``(offset, prefix_len, completion_lens)``) selects the
    multi-group Case-1 path; ``layout`` (a single ``(prefix_len, completion_lens)``) is the
    one-group path. Both require ``flash_composed`` + a layout; the hybrid path supplies only
    ``block_mask`` (no layout/forest) and always takes the flex path (warned once).
    """
    global _SP_FALLBACK_WARNED
    if sp_attention_backend() == "flash_composed":
        if forest is not None:
            return flash_composed_forest_attention(query, key, value, forest, scale=scale)
        if layout is not None:
            return flash_composed_tree_attention(query, key, value, layout[0], layout[1], scale=scale)
        if not _SP_FALLBACK_WARNED:
            logger.warning(
                "NRL_SP_ATTENTION_BACKEND=flash_composed but no shared-prefix layout was "
                "provided (e.g. hybrid/Mamba path); falling back to FlexAttention."
            )
            _SP_FALLBACK_WARNED = True
    return flex_tree_attention(query, key, value, block_mask, scale=scale)


@dataclass
class SharedPrefixParams:
    """Model-forward input describing one packed shared-prefix group ``[P, C_1, ..., C_G]``.

    Analogous to ``PackedSeqParams`` for THD packing: the RL/data layer builds it (lengths + the
    tree mask + prefix-continued position_ids from ``shared_prefix_packing``), threads it into
    ``HybridModel.forward``, which routes to ``HybridStack.forward_shared_prefix`` (instead of the
    dense decoder) and makes RoPE position-aware from ``position_ids``.
    """

    prefix_len: int
    completion_lens: List[int]
    # tree attention mask (Megatron convention, True == masked): prefix causal + each completion
    # attends the prefix and its own branch only. None is allowed for Mamba/MLP-only stacks.
    attention_mask: Optional[torch.Tensor] = None
    # prefix-continued positions [packed_len] (P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1, trailing
    # pad -> 0) for position-aware RoPE; None falls back to packed-index RoPE.
    position_ids: Optional[torch.Tensor] = None
    # Full packed length fed to the model (>= total_len, padded to a tensor-parallel multiple for
    # sequence parallelism). The tree BlockMask is built over this length. None == total_len (no
    # padding). Needed explicitly because under sequence parallelism the decoder sees a
    # sequence-sharded activation (length packed_len // TP), not the full packed sequence.
    packed_len: Optional[int] = None

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.completion_lens)


class SharedPrefixContext:
    """Packed shared-prefix layout for a single group ``[P, C_1, ..., C_G]``.

    A layer that consumes this slices the packed activation ``(total_len, 1, D)`` into the prefix
    ``[0:Lp]`` and each completion ``[start_i:start_i+Lc_i]``; gradients flow back through the
    forked state into the prefix, so the shared prompt receives the summed gradient of all its
    completions (loss/gradient-preserving).
    """

    def __init__(self, prefix_len: int, completion_lens: List[int]) -> None:
        self.prefix_len = int(prefix_len)
        self.completion_lens = [int(x) for x in completion_lens]
        assert self.prefix_len >= 1, "shared prefix must be non-empty"

    @property
    def num_completions(self) -> int:
        return len(self.completion_lens)

    @property
    def total_len(self) -> int:
        return self.prefix_len + sum(self.completion_lens)

    def segments(self) -> Iterator[Tuple[int, int, bool]]:
        """Yield ``(start, length, is_prefix)`` for the prefix then each completion, in packed
        order."""
        yield 0, self.prefix_len, True
        cursor = self.prefix_len
        for lc in self.completion_lens:
            yield cursor, lc, False
            cursor += lc
