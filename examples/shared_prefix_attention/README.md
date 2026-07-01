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

## What it measures

For each tree SHAPE, on identical work (the same leaf trajectories):

- **FUSED** — `flash_composed_forest_attention_fused` over the *deduplicated* tree (prompt/shared
  spine stored once).
- **BLOCK-DIAG** — plain causal `flash_attn_varlen` over the *same* trajectories fully expanded
  (no sharing; prompt re-encoded per leaf). This is what nemo-rl uses without shared-prefix packing.

Reported (fwd+bwd, best-of-N):

| column | meaning |
|---|---|
| `dup` | `base_tokens / tree_tokens` — token reduction from prefix sharing |
| `per_tok_ovh` | `(fused_ms/tree_tok) / (bd_ms/base_tok)` — the **kernel penalty** to drive toward 1.0 |
| `net_speedup` | `bd_ms / fused_ms` == `dup / per_tok_ovh` — the bottom line (>1 ⇒ tree wins on attention) |
| `max|Δflex|` | exactness gate: fused vs FlexAttention tree mask (bf16 ⇒ ~1e-2). MUST stay small. |

**Throughput model:** shared-prefix packing wins on attention iff `dup > per_tok_overhead`.
The fused kernel does *fewer* attention FLOPs than the expanded baseline, so the overhead is the
fp32 online-softmax merge + per-depth cross passes + index gathers/scatters — **not** real compute.
That is the headroom: driving `per_tok_ovh` toward 1.0 (target ~2.0 first) directly raises `net`.

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

Any optimization MUST keep `max|Δflex|` small (exactness) — see
`tests/.../test_shared_prefix_attention_parity.py` in the consuming repo for the parity gate.
