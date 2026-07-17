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

"""Shape-agnostic shared-prefix bin layout (:class:`TreeBinLayout`) and its length-side operations.

This is the token-free, role-free layer of the Strategy-B shared-prefix packer: it describes ONE
packed bin (a forest of >=1 trees) purely in terms of node spans and lengths, and knows nothing about
how the forest was constructed (branched-MC roles, GRPO star groups, ...). It lives next to
:class:`megatron.rl.tree_layout.PackedTreeLayout` so BOTH the nemo-rl path (via
``nemo_rl.experience.mc_tree_packing``, which builds a forest from branched-MC roles) and Megatron
native RL can share it.

  * :class:`TreeBinLayout` -- the layout object that rides as ``ProcessedMicrobatch.shared_prefix_layout``;
  * :func:`_emit_tree_bin_layout` -- build one from flat per-node records (the sole emitter);
  * :func:`build_star_tree_bin_layout` -- depth-1 GRPO group(s) as a star (seedless, role-free);
  * :func:`concat_tree_bin_layouts` / :func:`fully_trained_rows` / :func:`drop_row_from_tree_bin_layout`
    -- the co-pack + seed-agnostic DP-balance-peel operations.

The role-aware branched-MC forest builder (``build_tree_bin_layout(s)`` + ``_build_spec_forest``) stays
in ``nemo_rl.experience.mc_tree_packing`` and feeds :func:`_emit_tree_bin_layout` here.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.rl.tree_layout import PackedTreeLayout


@dataclass
class TreeBinLayout:
    """Token-free shared-prefix layout for ONE packed bin (a forest of >=1 trees).

    The Strategy-B worker-side counterpart to :class:`PackedTree`: built LENGTH-side by the
    sequence packer (``nemo_rl/data/packing/advanced.py``) from each row's
    ``(row_role, prompt_len, real_len, boundary_pos)`` -- no token ids needed at pack time, exactly
    like depth-1 ``build_shared_prefix_layout(Lp, Lcs)``. It is what rides as
    ``ProcessedMicrobatch.shared_prefix_layout`` and feeds the three worker seams:

      * forward (``_build_shared_prefix_params``; dispatched by ``hasattr(layout, "node_parent")``):
        ``node_start`` / ``node_len`` / ``node_parent`` (+ ``position_ids`` override);
      * token gather (``_pack_shared_prefix_for_megatron``): per-node ``(source_row, source_offset)``
        copies each node's span out of the bin's ``[G, S]`` rows into the deduped ``[1, T]``;
      * logprob scatter (``from_parallel_logits_to_logprobs_shared_prefix``; ``scatter_row`` branch):
        ``comp_positions`` / ``prev_positions`` / ``scatter_row`` / ``scatter_col`` / ``n_rows``.

    All row indices are BIN-LOCAL (numbered by position in the bin's ``[G, S]`` tensor), because the
    loss runs per microbatch on this bin's rows. Node ordering mirrors :func:`tree_rollout_to_packed_tree`
    (DFS preorder) so the packed layout matches the CPU oracle node-for-node; the parity test asserts it.
    """

    total_len: int
    node_start: list[int]
    node_len: list[int]
    node_parent: list[int]
    position_ids: torch.Tensor       # [total_len] prefix-continued RoPE
    source_row: list[int]            # [num_nodes] bin row each node's tokens are copied from
    source_offset: list[int]         # [num_nodes] offset of that span within the source row
    comp_positions: torch.Tensor     # [n_comp] packed index of each trained token
    prev_positions: torch.Tensor     # [n_comp] packed index whose logit predicts it (fan-out)
    scatter_row: torch.Tensor        # [n_comp] bin-local trajectory row
    scatter_col: torch.Tensor        # [n_comp] next-token column in the [n_rows, S-1] view
    n_rows: int                      # bin rows (== the bin's [G, S] row count)
    # The per-row metadata (bin order) this layout was built from. Pack-time only (unused on the
    # worker); read by the DP-balance peel to re-source masked context (per-row prompt_len).
    row_metas: list[dict[str, Any]]
    # [num_nodes] bin-local owner row of each node's TRAINED tokens, or -1 for a pure-context node
    # (a masked ancestor trained in another bin). This is the node-level (seed-agnostic) handle the
    # DP-balance peel operates on -- it needs no role/seed knowledge and works for whole trees,
    # split sub-bins, and co-packed forests alike. Pack-time only (unused on the worker).
    node_owner: list[int]


def _row_trained_totals(flat: list[dict[str, Any]]) -> dict[int, int]:
    """Per-row WHOLE-TREE trained-token count from the flattened node forest: ``{owner_row: sum of
    trained node lengths}``. Seed-agnostic. Used to stamp ``_trained_len`` so a SPLIT sub-bin's row
    metas still carry each row's total training (see :func:`fully_trained_rows`)."""
    total: dict[int, int] = {}
    for rec in flat:
        if rec["trained"]:
            total[int(rec["owner_row"])] = total.get(int(rec["owner_row"]), 0) + int(rec["length"])
    return total


def _emit_tree_bin_layout(
    flat: list[dict[str, Any]],
    *,
    n_rows: int,
    row_metas: list[dict[str, Any]],
    device: str,
) -> TreeBinLayout:
    """Emit a :class:`TreeBinLayout` from flat per-node records in topological order.

    Each record: ``length`` / ``parent`` (local idx) / ``trained`` / ``source_row`` /
    ``source_offset`` / ``scatter_row`` / ``base_col``. Generic -- shared by the single-bin path
    and the splitter.
    """
    node_len = [int(r["length"]) for r in flat]
    node_parent = [int(r["parent"]) for r in flat]
    node_start: list[int] = []
    cur = 0
    for L in node_len:
        node_start.append(cur)
        cur += L
    layout = PackedTreeLayout(node_start=node_start, node_len=node_len, node_parent=node_parent)
    prev_index = layout.prev_token_index()

    comp_positions: list[int] = []
    prev_positions: list[int] = []
    scatter_row: list[int] = []
    scatter_col: list[int] = []
    for i, r in enumerate(flat):
        if not r["trained"]:
            continue
        s = node_start[i]
        base = int(r["base_col"])
        sr = int(r["scatter_row"])
        for j in range(node_len[i]):
            comp_positions.append(s + j)
            prev_positions.append(int(prev_index[s + j]))
            scatter_row.append(sr)
            scatter_col.append(base + j - 1)

    def _t(xs: list[int]) -> torch.Tensor:
        return torch.tensor(xs, dtype=torch.long, device=device)

    # Per-node owner row for the peel: the trained span's row, or -1 for a pure-context node.
    node_owner = [int(r["scatter_row"]) if r["trained"] else -1 for r in flat]

    return TreeBinLayout(
        total_len=int(layout.total_len),
        node_start=node_start,
        node_len=node_len,
        node_parent=node_parent,
        position_ids=_t(layout.position_ids()),
        source_row=[int(r["source_row"]) for r in flat],
        source_offset=[int(r["source_offset"]) for r in flat],
        comp_positions=_t(comp_positions),
        prev_positions=_t(prev_positions),
        scatter_row=_t(scatter_row),
        scatter_col=_t(scatter_col),
        n_rows=n_rows,
        row_metas=row_metas,
        node_owner=node_owner,
    )


def build_star_tree_bin_layout(rows: list[dict[str, Any]], device: str = "cpu") -> TreeBinLayout:
    """Depth-1 GRPO group(s) as a star :class:`TreeBinLayout` -- one layout object for depth-1 and
    arbitrary-depth alike. Each ``group_id`` becomes a root (the shared prompt, context) plus one
    trained child per completion row: the seedless analogue of the role-aware forest builder (a plain
    GRPO group has no seed spine, so it cannot go through the seed-requiring role adapter).

    ``rows``: bin-local list of dicts with ``group_id`` / ``prompt_len`` / ``real_len``. A completion
    with no fresh tokens (``real_len <= prompt_len``) is skipped. The resulting node arrays are
    DFS-preorder (root then its children, per group), and the depth-1 logprob semantics fall out of
    the generic ``scatter_row`` fan-out identically to the old ``SharedPrefixLayout`` path.
    """
    by_group: dict[Any, list[int]] = {}
    for k, r in enumerate(rows):
        by_group.setdefault(r["group_id"], []).append(k)

    flat: list[dict[str, Any]] = []
    for ridxs in by_group.values():
        Lp = int(rows[ridxs[0]]["prompt_len"])
        root_idx = len(flat)
        # root = shared prompt (context, never trained); source its tokens from the group's first row.
        flat.append(
            {"length": Lp, "parent": -1, "trained": False,
             "source_row": ridxs[0], "source_offset": 0, "scatter_row": -1, "base_col": -1}
        )
        for lr in ridxs:
            comp = int(rows[lr]["real_len"]) - Lp
            if comp <= 0:
                continue
            flat.append(
                {"length": comp, "parent": root_idx, "trained": True,
                 "source_row": lr, "source_offset": Lp, "scatter_row": lr, "base_col": Lp}
            )
    metas = [{**dict(r), "_trained_len": max(0, int(r["real_len"]) - int(r["prompt_len"]))} for r in rows]
    return _emit_tree_bin_layout(flat, n_rows=len(rows), row_metas=metas, device=device)


def concat_tree_bin_layouts(layouts: list["TreeBinLayout"], device: str = "cpu") -> "TreeBinLayout":
    """Concatenate several TreeBinLayouts into ONE physical bin (a forest, block-diagonal between
    them -- distinct trees / sub-bins share nothing). Lets the packer co-pack under-filled sub-bins
    up to capacity (recovers fill / fewer microbatches) without changing any per-unit dedup.

    Offsets are mechanical: packed indices (node_start via node_len, comp/prev_positions) shift by
    the running token count; node_parent by the running node count; source_row/scatter_row by the
    running bin-row count; source_offset and scatter_col are within-row and unchanged. The caller
    must concatenate the units' row-position lists in the SAME order.
    """
    node_len: list[int] = []
    node_parent: list[int] = []
    source_row: list[int] = []
    source_offset: list[int] = []
    comp: list[int] = []
    prev: list[int] = []
    srow: list[int] = []
    scol: list[int] = []
    node_owner: list[int] = []
    pos_parts: list[torch.Tensor] = []
    row_metas: list[dict[str, Any]] = []
    tok_off = 0
    node_off = 0
    row_off = 0
    for lay in layouts:
        node_len.extend(int(x) for x in lay.node_len)
        node_parent.extend(-1 if int(p) == -1 else int(p) + node_off for p in lay.node_parent)
        source_row.extend(int(r) + row_off for r in lay.source_row)
        source_offset.extend(int(o) for o in lay.source_offset)
        comp.extend(int(c) + tok_off for c in lay.comp_positions.tolist())
        prev.extend((int(p) + tok_off) if int(p) >= 0 else -1 for p in lay.prev_positions.tolist())
        srow.extend(int(r) + row_off for r in lay.scatter_row.tolist())
        scol.extend(int(c) for c in lay.scatter_col.tolist())
        node_owner.extend((int(o) + row_off) if int(o) >= 0 else -1 for o in lay.node_owner)
        pos_parts.append(lay.position_ids.to(device))
        row_metas.extend(lay.row_metas)
        tok_off += int(lay.total_len)
        node_off += len(lay.node_len)
        row_off += int(lay.n_rows)

    node_start: list[int] = []
    cur = 0
    for L in node_len:
        node_start.append(cur)
        cur += L

    def _t(xs: list[int]) -> torch.Tensor:
        return torch.tensor(xs, dtype=torch.long, device=device)

    return TreeBinLayout(
        total_len=tok_off,
        node_start=node_start,
        node_len=node_len,
        node_parent=node_parent,
        position_ids=torch.cat(pos_parts) if pos_parts else _t([]),
        source_row=source_row,
        source_offset=source_offset,
        comp_positions=_t(comp),
        prev_positions=_t(prev),
        scatter_row=_t(srow),
        scatter_col=_t(scol),
        n_rows=row_off,
        row_metas=row_metas,
        node_owner=node_owner,
    )


def fully_trained_rows(layout: "TreeBinLayout") -> set[int]:
    """Bin-local rows whose ENTIRE completion is trained within this bin.

    A row is safe to peel to its own block-diagonal bin only if it is fully trained here: otherwise
    the block-diag bin (which trains the row's whole loss-masked completion) would re-train segments
    this row owns in OTHER sub-bins -> double counting. A whole-tree bin has every row fully trained;
    a split sub-bin whose seed spine was carved across microbatches has partially-trained rows, which
    this excludes.

    Seed-agnostic: compares this bin's trained tokens per row (summed over ``node_owner``) against the
    row's WHOLE-TREE trained-token count stamped at build time (``row_metas[r]["_trained_len"]``). We
    must NOT use ``real_len - prompt_len`` -- that overcounts a branch, which trains only its fresh
    continuation (``real_len - prompt_len - boundary_pos``), not the shared seed prefix.
    """
    trained_tok = [0] * int(layout.n_rows)
    for i, owner in enumerate(layout.node_owner):
        if owner >= 0:
            trained_tok[owner] += int(layout.node_len[i])
    out: set[int] = set()
    for r in range(int(layout.n_rows)):
        total = int(layout.row_metas[r].get("_trained_len", -1))
        if total > 0 and trained_tok[r] == total:
            out.add(r)
    return out


def drop_row_from_tree_bin_layout(
    layout: "TreeBinLayout", drop_row: int, device: str = "cpu"
) -> Optional["TreeBinLayout"]:
    """Return a new :class:`TreeBinLayout` for all rows except ``drop_row``, or ``None`` if fewer
    than two rows remain (caller makes the remainder a block-diagonal bin).

    Seed-agnostic node-level peel: keeps every node trained by a surviving row plus the ancestor
    nodes those need as context (a ``drop_row``-owned node that is still an ancestor of a survivor is
    RETAINED as context -- its training moves to ``drop_row``'s new block-diag bin), re-sources any
    context span from a surviving row whose prefix covers it, renumbers rows/nodes, and re-emits via
    :func:`_emit_tree_bin_layout`. Works uniformly for whole trees, split sub-bins, and co-packed
    forests -- no ``_build_spec_forest`` / role / seed dependency. ``drop_row`` MUST be fully trained
    in this bin (see :func:`fully_trained_rows`), else its block-diag bin would double-train.
    """
    N = len(layout.node_len)
    par = [int(x) for x in layout.node_parent]
    nl = [int(x) for x in layout.node_len]
    pos_of = [int(x) for x in layout.source_offset]  # per-node trajectory offset (== base col)
    owner = [int(x) for x in layout.node_owner]

    kept_rows = [r for r in range(int(layout.n_rows)) if r != drop_row]
    if len(kept_rows) < 2:
        return None

    # Keep nodes trained by a surviving row, then their ancestors (context they attend).
    keep = [owner[i] >= 0 and owner[i] != drop_row for i in range(N)]
    for i in range(N):
        if keep[i]:
            p = par[i]
            while p != -1 and not keep[p]:
                keep[p] = True
                p = par[p]
    kept_nodes = [i for i in range(N) if keep[i]]
    if not kept_nodes:
        return None

    row_map = {r: k for k, r in enumerate(kept_rows)}
    node_map = {old: new for new, old in enumerate(kept_nodes)}
    ctx_len = [int(layout.row_metas[r]["prompt_len"]) for r in range(int(layout.n_rows))]

    flat: list[dict[str, Any]] = []
    for old in kept_nodes:
        pos = pos_of[old]
        length = nl[old]
        p = par[old]
        p_local = node_map[p] if p != -1 else -1
        if owner[old] >= 0 and owner[old] != drop_row:
            sr = row_map[owner[old]]
            flat.append(
                {"length": length, "parent": p_local, "trained": True,
                 "source_row": sr, "source_offset": pos, "scatter_row": sr, "base_col": pos}
            )
        else:
            # context: source from any surviving row whose prefix covers [pos, pos+length)
            need = pos + length
            src = next((row_map[r] for r in kept_rows if ctx_len[r] >= need), 0)
            flat.append(
                {"length": length, "parent": p_local, "trained": False,
                 "source_row": src, "source_offset": pos, "scatter_row": -1, "base_col": -1}
            )

    return _emit_tree_bin_layout(
        flat,
        n_rows=len(kept_rows),
        row_metas=[dict(layout.row_metas[r]) for r in kept_rows],
        device=device,
    )
