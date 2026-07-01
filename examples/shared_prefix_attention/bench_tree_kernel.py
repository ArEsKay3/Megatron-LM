# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained benchmark for the fused shared-prefix (tree) attention kernel.

Goal: give a faithful, standalone harness to drive down the per-token cost of
``flash_composed_forest_attention_fused`` (in
``megatron/core/models/hybrid/shared_prefix.py``). Depends only on that module +
``flash_attn`` + ``torch`` (FlexAttention for the exactness oracle) -- no NeMo-RL.

For each tree SHAPE we compare, on identical work (the same leaf trajectories):
  * FUSED tree    -- flash_composed_forest_attention_fused over the DEDUPLICATED tree
                     (prompt/shared spine stored once): ``tree_tokens`` tokens.
  * BLOCK-DIAG    -- plain causal flash_attn_varlen over the SAME trajectories fully
                     expanded (no sharing; prompt re-encoded per leaf): ``base_tokens``.

Reported per shape (fwd+bwd, best-of-N):
  dup            = base_tokens / tree_tokens              (token reduction from sharing)
  per_tok_overhead = (fused_ms/tree_tokens) / (bd_ms/base_tokens)   (kernel penalty)
  net_speedup    = bd_ms / fused_ms  == dup / per_tok_overhead      (the bottom line)
  exact          = fused matches FlexAttention tree mask (correctness gate)

Throughput model: shared-prefix wins on attention iff ``dup > per_tok_overhead``.
The fused kernel does FEWER attention FLOPs than the expanded baseline, so the
overhead is merge/gather/launch, not real compute -- that's the headroom to close.

Run on a GPU node (flash_attn is bf16-only):
  python examples/shared_prefix_attention/bench_tree_kernel.py
  python examples/shared_prefix_attention/bench_tree_kernel.py --iters 10 --shapes branched_mc
"""

from __future__ import annotations

import argparse

import torch

# GQA dims (Qwen3-4B-like): 32 query heads over 8 kv heads, head_dim 128.
NP, NG, HN = 32, 8, 128


# --------------------------------------------------------------------------------------
# Tree shapes -> (node_start, node_len, node_parent, total_tokens, baseline_row_lens)
# baseline_row_lens = per trained-trajectory root->leaf path length (the no-sharing bin).
# Node arrays MUST be DFS-preorder (the fused kernel asserts it).
# --------------------------------------------------------------------------------------
def _finish(parent, length, leaf_rows):
    start, acc = [], 0
    for ln in length:
        start.append(acc)
        acc += ln
    return start, length, parent, acc, leaf_rows


def balanced_tree(P=1024, L=512, B=2, D=4):
    """Balanced tree: root len P, every other node len L, B children, depth D (the existing shape)."""
    parent, length = [], []
    path_len = {}  # node -> root..node token length

    def dfs(p, d, plen):
        i = len(parent)
        parent.append(p)
        ln = P if d == 0 else L
        length.append(ln)
        path_len[i] = plen + ln
        if d < D:
            for _ in range(B):
                dfs(i, d + 1, path_len[i])

    dfs(-1, 0, 0)
    is_parent = set(parent)
    leaf_rows = [path_len[i] for i in range(len(parent)) if i not in is_parent]
    return _finish(parent, length, leaf_rows)


def branched_mc_tree(P=512, seg=400, n_boundaries=6, sib_len=2900, n_sib=4,
                     branch_len=2500, branch_boundaries=(1, 2, 3, 4)):
    """Branched-MC shape: prompt root -> a seed spine of ``n_boundaries`` segments (a chain),
    with ``n_sib`` siblings forking at the root and one branch forking off the seed segment at
    each boundary in ``branch_boundaries``. Deep (spine depth ~n_boundaries), which is what makes
    the fused kernel's per-depth cross passes expensive -- the regime our live runs hit.

    Trained trajectories (the block-diag baseline rows): the seed (prompt+full spine), each sibling
    (prompt+sib_len), and each branch (prompt + spine-up-to-its-boundary + branch_len).
    """
    parent, length = [], []

    def add(p, ln):
        i = len(parent)
        parent.append(p)
        length.append(ln)
        return i

    branch_set = set(branch_boundaries)
    root = add(-1, P)               # prompt
    for _ in range(n_sib):          # siblings fork at the root (emitted before the spine subtree)
        add(root, sib_len)

    # Seed spine as a chain, emitted in DFS PREORDER: each segment, then its branch (a leaf), then
    # recurse into the deeper spine -- so every segment's subtree is the contiguous run after it
    # (required by the fused kernel). ``k`` is the 1-indexed boundary.
    def build_spine(parent_node, k):
        s = add(parent_node, seg)
        if k in branch_set:
            add(s, branch_len)      # branch continuation off this boundary
        if k < n_boundaries:
            build_spine(s, k + 1)

    build_spine(root, 1)

    # baseline row lengths (root->leaf path per trained trajectory)
    seed_path = P + n_boundaries * seg
    rows = [seed_path]                             # the seed trajectory
    rows += [P + sib_len] * n_sib                  # siblings
    rows += [P + b * seg + branch_len for b in branch_boundaries]  # branches
    return _finish(parent, length, rows)


SHAPES = {
    "balanced_d4": lambda: balanced_tree(),
    "branched_mc": lambda: branched_mc_tree(),
}


def replicate(shape, K):
    """Replicate a shape K times into a K-tree forest (more trajectories, SAME per-tree shape).

    Grows total tokens by adding trajectories -- NOT by lengthening sequences -- which is the
    packing regime (attention time is linear in token count at fixed per-sequence length). Used by
    --sweep to verify per-token cost is flat, so the overhead ratio is bin-size independent.
    """
    node_start, node_len, node_parent, _total, base_rows = shape
    n = len(node_len)
    nl, par = [], []
    for r in range(K):
        for i in range(n):
            nl.append(node_len[i])
            par.append(-1 if node_parent[i] == -1 else node_parent[i] + r * n)
    return _finish([p for p in par], nl, base_rows * K)


# --------------------------------------------------------------------------------------
# Timing + measurement
# --------------------------------------------------------------------------------------
def _best_fwd_bwd_ms(fn, q, k, v, iters, warmup):
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        fn(q, k, v).float().sum().backward()
    best = float("inf")
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(q, k, v).float().sum().backward()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def _flex_exact(sp, q, k, v, node_start, node_len, node_parent, total, dev):
    """max|fused - FlexAttention(tree mask)| over all rows (correctness gate)."""
    import torch.nn.attention.flex_attention as flex

    seg = []
    for i, ln in enumerate(node_len):
        seg += [i] * ln
    n = len(node_parent)
    anc = torch.zeros(n, n, dtype=torch.bool, device=dev)
    for a in range(n):
        x = a
        while x != -1:
            anc[a, x] = True
            x = node_parent[x]
    seg_t = torch.tensor(seg, device=dev)

    def mask_mod(b, h, qi, ki):
        return anc[seg_t[qi], seg_t[ki]] & (ki <= qi)

    bm = flex.create_block_mask(mask_mod, B=1, H=None, Q_LEN=total, KV_LEN=total, device=dev)
    o_fused = sp.flash_composed_forest_attention_fused(q, k, v, node_start, node_len, node_parent).squeeze(1)
    o_flex = sp.flex_tree_attention(q, k, v, bm).squeeze(1)
    return (o_fused.float() - o_flex.float()).abs().max().item()


def measure(sp, name, builder, iters, warmup, dev, dt, check_exact=True):
    from flash_attn import flash_attn_varlen_func

    node_start, node_len, node_parent, total, base_rows = builder()
    depth_max = 0
    for i in range(len(node_parent)):
        d, p = 0, node_parent[i]
        while p != -1:
            d += 1
            p = node_parent[p]
        depth_max = max(depth_max, d)

    torch.manual_seed(0)
    # exactness on non-grad tensors (the FlexAttention oracle is O(total^2) memory, so skip it for
    # very large replicated shapes -- exactness is shape-topology-dependent, not size-dependent).
    maxdiff = float("nan")
    if check_exact:
        q0 = torch.randn(total, 1, NP, HN, device=dev, dtype=dt)
        k0 = torch.randn(total, 1, NG, HN, device=dev, dtype=dt)
        v0 = torch.randn(total, 1, NG, HN, device=dev, dtype=dt)
        maxdiff = _flex_exact(sp, q0, k0, v0, node_start, node_len, node_parent, total, dev)

    # FUSED tree over the deduplicated tree
    def mkg(h):
        return torch.randn(total, 1, h, HN, device=dev, dtype=dt, requires_grad=True)

    qf, kf, vf = mkg(NP), mkg(NG), mkg(NG)
    t_fused = _best_fwd_bwd_ms(
        lambda a, b, c: sp.flash_composed_forest_attention_fused(a, b, c, node_start, node_len, node_parent),
        qf, kf, vf, iters, warmup,
    )

    # BLOCK-DIAG baseline: causal flash_varlen over the same leaves fully expanded (no sharing)
    base_tokens = sum(base_rows)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(base_rows), 0).tolist()),
                      dtype=torch.int32, device=dev)
    maxlen = max(base_rows)
    qb = torch.randn(base_tokens, NP, HN, device=dev, dtype=dt, requires_grad=True)
    kb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt, requires_grad=True)
    vb = torch.randn(base_tokens, NG, HN, device=dev, dtype=dt, requires_grad=True)
    t_bd = _best_fwd_bwd_ms(
        lambda a, b, c: flash_attn_varlen_func(a, b, c, cu, cu, maxlen, maxlen, causal=True),
        qb, kb, vb, iters, warmup,
    )

    dup = base_tokens / total
    per_tok_overhead = (t_fused / total) / (t_bd / base_tokens)
    net = t_bd / t_fused
    return dict(name=name, nodes=len(node_len), rows=len(base_rows), depth=depth_max,
                tree_tokens=total, base_tokens=base_tokens, dup=dup,
                fused_ms=t_fused, bd_ms=t_bd, overhead=per_tok_overhead, net=net, maxdiff=maxdiff)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="balanced_d4,branched_mc",
                    help="comma list of: " + ",".join(SHAPES))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--sweep", default="",
                    help="comma list of replication factors K (e.g. 1,2,4,8): replicate each shape "
                         "into a K-tree forest and report per-token cost, to verify it is flat "
                         "(=> the overhead ratio is bin-size independent; no iso-bin mode needed)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: run on a GPU node.")
        return 1
    try:
        import flash_attn  # noqa: F401
    except Exception:
        print("ERROR: flash_attn required.")
        return 1
    from megatron.core.models.hybrid import shared_prefix as sp

    if not sp.HAVE_FLEX_ATTENTION:
        print("ERROR: FlexAttention unavailable (need torch >= 2.5).")
        return 1

    dev = torch.device("cuda")
    dt = torch.bfloat16
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for nm in shapes:
        if nm not in SHAPES:
            print(f"  (unknown shape {nm}; known: {list(SHAPES)})")
            return 1
    print(f"# device={torch.cuda.get_device_name()} dtype=bf16 heads={NP} kv_heads={NG} head_dim={HN} "
          f"iters={args.iters}")

    if args.sweep:
        # Scaling check: replicate each shape into a K-tree forest (more trajectories, fixed per-tree
        # shape) and report per-token cost. Flat per-token across K => attention time is linear in
        # total tokens for BOTH paths => the overhead ratio is bin-size independent.
        Ks = [int(x) for x in args.sweep.split(",") if x.strip()]
        print("# SCALING SWEEP: replicate KxK (fixed per-tree shape). Flat per-token => linear regime.\n")
        hdr = ["shape", "K", "tree_tok", "base_tok", "fused_ns/tok", "bd_ns/tok", "per_tok_ovh"]
        print("  ".join(f"{c:>13}" for c in hdr))
        for nm in shapes:
            base = SHAPES[nm]()
            for K in Ks:
                r = measure(sp, nm, (lambda b=base, k=K: replicate(b, k)), args.iters, args.warmup,
                            dev, dt, check_exact=(K == 1))
                fpt = r["fused_ms"] / r["tree_tokens"] * 1e6
                bpt = r["bd_ms"] / r["base_tokens"] * 1e6
                vals = [nm, K, r["tree_tokens"], r["base_tokens"],
                        f"{fpt:.1f}", f"{bpt:.1f}", f"{r['overhead']:.2f}x"]
                print("  ".join(f"{str(v):>13}" for v in vals))
        print("\n# per-token flat across K => single-shape overhead is representative (bin-size independent).")
        return 0

    hdr = ["shape", "nodes", "rows", "depth", "tree_tok", "base_tok", "dup",
           "fused_ms", "bd_ms", "per_tok_ovh", "net_speedup", "maxdiff_flex"]
    print("  ".join(f"{c:>12}" for c in hdr))
    for nm in shapes:
        r = measure(sp, nm, SHAPES[nm], args.iters, args.warmup, dev, dt)
        vals = [r["name"], r["nodes"], r["rows"], r["depth"], r["tree_tokens"], r["base_tokens"],
                f"{r['dup']:.2f}", f"{r['fused_ms']:.2f}", f"{r['bd_ms']:.2f}",
                f"{r['overhead']:.2f}x", f"{r['net']:.2f}x", f"{r['maxdiff']:.1e}"]
        print("  ".join(f"{str(v):>12}" for v in vals))
    print("\n# net_speedup = bd_ms/fused_ms = dup/per_tok_ovh.  >1 = tree wins on attention.")
    print("# per_tok_ovh is the kernel penalty to drive toward 1.0 (merge+gather+per-depth launches).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
