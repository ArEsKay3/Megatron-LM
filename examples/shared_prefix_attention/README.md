# Shared-prefix (tree) attention kernel benchmark

Standalone harness to measure — and drive down — the per-token cost of the fused
shared-prefix tree-attention kernel
(`megatron/core/models/hybrid/shared_prefix.py::flash_composed_forest_attention_fused`).

Depends only on `megatron.core.models.hybrid.shared_prefix` + `flash_attn` + `torch`
(FlexAttention is used as the exactness oracle). No NeMo-RL dependency.

## Run

```bash
# GPU node, inside the container (flash_attn is bf16-only):
python examples/shared_prefix_attention/bench_tree_kernel.py
python examples/shared_prefix_attention/bench_tree_kernel.py --shapes branched_mc --iters 16
```

## Example output (GB300, bf16, iters=8)

```
# GB300, bf16, iters=8    (ovh=per_tok_ovh, net=net_speedup, Δflex=maxdiff_flex)
       shape  depth  tree_tok  base_tok   dup  fused_ms  bd_ms    ovh    net  Δflex
 balanced_d4      4     16384     49152  3.00     19.89  13.38  4.46x  0.67x   4e-3
 branched_mc      6     24512     32608  1.33     24.71  10.70  3.07x  0.43x   4e-3
```

(The tool also prints `nodes` and `rows` per shape; trimmed here for width.)

Reading it: the fused tree kernel is exact (`maxdiff_flex` ~4e-3, bf16 rounding) but currently
**slower on attention** than block-diagonal flash — `net_speedup` 0.43x on the branched-MC shape
(≈2.3x slower), because the ~3.1x per-token kernel penalty outweighs the 1.33x token reduction.
The optimization goal is to drive `per_tok_ovh` down (toward the 2.0x guard target, ideally ~1x);
once `per_tok_ovh < dup`, `net_speedup` crosses 1.0 and the tree wins on attention. (Numbers move
with GPU/flash version and shape; regenerate before drawing conclusions.)

## What it measures

For each tree SHAPE, on identical work (the same leaf trajectories):

- **FUSED** — `flash_composed_forest_attention_fused` over the *deduplicated* tree (prompt/shared
  spine stored once).
- **BLOCK-DIAG** — plain causal `flash_attn_varlen` over the *same* trajectories fully expanded
  (no sharing; prompt re-encoded per leaf). This is what nemo-rl uses without shared-prefix packing.

Each shape is reported for **two phases** — `fwd+bwd` (training) and `fwd-only` (logprob):

| metric | role | meaning |
|---|---|---|
| `net` | **VERDICT** | `bd_ms / fused_ms` — total wall-time for the SAME trajectories, tree vs block-diag. >1 ⇒ tree wins. Reported per phase. |
| `FLOP ceiling` | **theoretical best** | `bd_pairs / tree_pairs` — net speedup if the tree kernel ran at flash's achieved TFLOP/s. The tree mask has *fewer* (q,k) pairs (shared prefix attended once), so this — not `dup` — is the real headroom / fastest we could go. |
| achieved `TFLOP/s` + `eff%` | **kernel target** | fused vs block-diag achieved TFLOP/s (FLOPs the mask implies ÷ time). `eff% = fused/bd`; the gap to 100% is the kernel work left. `net = FLOP_ceiling × eff`. |
| `per_tok_ovh` | per-token view | per-token time ratio (size-independent). Kept because `--sweep` uses it to prove linear scaling; **secondary** to the FLOP view (the tree does different FLOPs per token). |
| `dup` | context | `base_tokens / tree_tokens` — token reduction from sharing. |
| `maxdiff_flex` | correctness | max abs diff vs FlexAttention tree-mask reference (bf16 ⇒ ~1e-2). MUST stay small. |

**Why FLOP/s, not per-token.** Per-token normalization silently penalizes the tree for doing
*fewer* attention FLOPs (the shared prefix is attended once, not re-encoded per leaf). The attention
**mask**, not the token count, sets the real math. So the honest measures are the **FLOP ceiling**
(best possible net, from the sparser mask) and **achieved TFLOP/s** (how close the kernel runs to
flash's rate): `net = FLOP_ceiling × (fused_TFLOP/s ÷ bd_TFLOP/s)`. Block-diag flash runs near peak,
so `bd_TFLOP/s` is the bar; `eff%` is the kernel gap to close.

**Why two phases.** Logprob is a single **forward**; training is **fwd+bwd**. The tree overhead lives
in attention, and the backward adds large *non-attention* weight-gradient GEMMs that dilute it — so
the SP penalty is **worse for fwd-only (logprob)** than fwd+bwd (training). The bench reports both, so
the optimizer watches the **forward** number — the phase SP hurts most.

## Why per-token normalization is valid (`--sweep`)

The two variants process *different* total token counts (tree = deduplicated, block-diag = expanded),
so we compare them **per token**. That is only fair if both are in the linear regime — i.e. time
scales with token count, so per-token cost is flat. Verify it by replicating a shape into a K-tree
forest (more trajectories, *same* per-tree shape — growing token count without lengthening any
sequence):

```bash
python examples/shared_prefix_attention/bench_tree_kernel.py --shapes branched_mc --sweep 1,2,4,8
```

Observed (GB300) — per-token cost is flat (drifts *down* slightly as launch overhead amortizes), and
the overhead ratio is bin-size independent:

```
      shape    K   tree_tok   base_tok  fused_ns/tok  bd_ns/tok  per_tok_ovh
branched_mc    1      24512      32608        1010.1      328.2        3.08x
branched_mc    2      49024      65216         974.4      321.5        3.03x
branched_mc    4      98048     130432         957.2      318.3        3.01x
branched_mc    8     196096     260864         931.4      316.8        2.94x
```

So the single-shape overhead is representative (if anything slightly pessimistic at small size), and
no iso-bin / equal-length mode is needed — normalizing by tokens is sound.

## Shapes

- `balanced_d4` — balanced tree (root 1024, 512-token nodes, branch factor 2, depth 4). Stress shape.
- `branched_mc` — the branched-MC RL shape: a prompt root, a seed spine of 6 segments (a chain),
  4 siblings forking at the root, and branches forking off seed segments. Deep but with long
  segments — representative of the live GRPO workload (dup ≈ 1.3–1.45).

Any optimization MUST keep `maxdiff_flex` small (exactness) — see
`tests/.../test_shared_prefix_attention_parity.py` in the consuming repo for the parity gate.
