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

## Shapes

- `balanced_d4` — balanced tree (root 1024, 512-token nodes, branch factor 2, depth 4). Stress shape.
- `branched_mc` — the branched-MC RL shape: a prompt root, a seed spine of 6 segments (a chain),
  4 siblings forking at the root, and branches forking off seed segments. Deep but with long
  segments — representative of the live GRPO workload (dup ≈ 1.3–1.45).

Any optimization MUST keep `max|Δflex|` small (exactness) — see
`tests/.../test_shared_prefix_attention_parity.py` in the consuming repo for the parity gate.
