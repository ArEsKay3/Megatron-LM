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

"""Forest-of-trees layout for shared-prefix / tree packing.

A packed bin is described as a **forest**: a list of nodes, each a contiguous token
span in the packed sequence with a parent (or ``-1`` for a root). An edge means "child
continues from parent" (shared prefix). This is the single descriptor behind every
packing shape:

  * plain trajectory        -> one root node
  * GRPO group (today's SP)  -> depth-1 tree: root prompt + G leaf completions
  * multiple groups per bin  -> forest of depth-1 trees   (Case 1)
  * tree of generations      -> arbitrary-depth trees      (Case 2)

It generalizes :class:`SharedPrefixLayout` (its depth-1 form). Lives in ``megatron.rl``
so it is shared by BOTH Megatron's own RL loop and downstream consumers (e.g. NeMo-RL),
which import it -- the packing/layout logic has one home. Intentionally pure Python (no
torch / megatron-core import) so it is unit-testable standalone; consumers materialize
tensors from the plain arrays here.

Invariants (validated in ``__post_init__``):
  * a parent's span precedes its children's spans (topological / DFS order);
  * node spans are contiguous and cover ``[0, total_len)`` with no gaps or overlaps;
  * exactly the roots have ``parent == -1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class PackedTreeLayout:
    """Forest layout for one packed bin.

    Args:
        node_start: per-node start offset into the packed sequence.
        node_len: per-node token count.
        node_parent: per-node parent index (into these arrays), or ``-1`` for a root.
    """

    node_start: List[int]
    node_len: List[int]
    node_parent: List[int]
    total_len: int = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.node_start)
        if not (len(self.node_len) == n == len(self.node_parent)):
            raise ValueError("node_start / node_len / node_parent length mismatch")
        if n == 0:
            raise ValueError("PackedTreeLayout needs at least one node")
        self.total_len = sum(int(x) for x in self.node_len)

        cursor = 0
        for i in range(n):
            if int(self.node_start[i]) != cursor:
                raise ValueError(
                    f"node {i} start={self.node_start[i]} != expected {cursor} "
                    "(spans must be contiguous in array order)"
                )
            if int(self.node_len[i]) <= 0:
                raise ValueError(f"node {i} has non-positive length {self.node_len[i]}")
            cursor += int(self.node_len[i])

        for i in range(n):
            p = int(self.node_parent[i])
            if p == -1:
                continue
            if not (0 <= p < n):
                raise ValueError(f"node {i} parent {p} out of range")
            if p >= i:
                raise ValueError(
                    f"node {i} parent {p} does not precede it (forest must be "
                    "topologically ordered so parents come first)"
                )

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_shared_prefix(cls, prefix_len: int, completion_lens: Sequence[int]) -> "PackedTreeLayout":
        """Depth-1 tree: root prompt (``prefix_len``) + one leaf per completion.

        The GRPO-group / today's-``SharedPrefixLayout`` special case.
        """
        prefix_len = int(prefix_len)
        comp = [int(c) for c in completion_lens]
        starts = [0]
        lens = [prefix_len]
        parents = [-1]
        cur = prefix_len
        for c in comp:
            starts.append(cur)
            lens.append(c)
            parents.append(0)
            cur += c
        return cls(node_start=starts, node_len=lens, node_parent=parents)

    @classmethod
    def concat(cls, layouts: Sequence["PackedTreeLayout"]) -> "PackedTreeLayout":
        """Combine several trees into one forest bin (Case 1: multiple groups per bin).

        Node spans are shifted by the running offset and parent indices rebased; the
        trees stay disjoint (different roots, no cross-tree edges).
        """
        starts: List[int] = []
        lens: List[int] = []
        parents: List[int] = []
        node_off = 0
        tok_off = 0
        for lay in layouts:
            for i in range(len(lay.node_start)):
                starts.append(int(lay.node_start[i]) + tok_off)
                lens.append(int(lay.node_len[i]))
                p = int(lay.node_parent[i])
                parents.append(-1 if p == -1 else p + node_off)
            node_off += len(lay.node_start)
            tok_off += lay.total_len
        return cls(node_start=starts, node_len=lens, node_parent=parents)

    # ------------------------------------------------------------------ derived views
    @property
    def num_nodes(self) -> int:
        return len(self.node_start)

    def roots(self) -> List[int]:
        return [i for i, p in enumerate(self.node_parent) if int(p) == -1]

    def depth(self) -> int:
        """Max edges from any node to its root (depth-1 star -> 1; forest of stars -> 1)."""
        d_max = 0
        for i in range(self.num_nodes):
            d, p = 0, int(self.node_parent[i])
            while p != -1:
                d += 1
                p = int(self.node_parent[p])
            d_max = max(d_max, d)
        return d_max

    def segment_ids(self) -> List[int]:
        """``[total_len]`` mapping each token to its node id."""
        seg = [0] * self.total_len
        for i in range(self.num_nodes):
            s = int(self.node_start[i])
            for t in range(s, s + int(self.node_len[i])):
                seg[t] = i
        return seg

    def ancestors(self, node: int) -> List[int]:
        """Node ids on the root->node path, inclusive of ``node``, root-first."""
        chain = []
        i = int(node)
        while i != -1:
            chain.append(i)
            i = int(self.node_parent[i])
        chain.reverse()
        return chain

    def is_ancestor(self, anc: int, node: int) -> bool:
        """True iff ``anc`` is ``node`` or one of its ancestors (key-attendable by node)."""
        i = int(node)
        while i != -1:
            if i == int(anc):
                return True
            i = int(self.node_parent[i])
        return False

    def node_pos_offset(self) -> List[int]:
        """Per-node RoPE position base: a child continues from where its parent ended.

        Roots start at 0; ``offset[child] = offset[parent] + len[parent]``. (Depth-1:
        every completion continues from the shared prompt.)
        """
        off = [0] * self.num_nodes
        for i in range(self.num_nodes):  # parents precede children, so single pass works
            p = int(self.node_parent[i])
            off[i] = 0 if p == -1 else off[p] + int(self.node_len[p])
        return off

    def position_ids(self) -> List[int]:
        """``[total_len]`` prefix-continued RoPE positions for the packed sequence."""
        off = self.node_pos_offset()
        pos = [0] * self.total_len
        for i in range(self.num_nodes):
            s = int(self.node_start[i])
            for j in range(int(self.node_len[i])):
                pos[s + j] = off[i] + j
        return pos

    def prev_token_index(self) -> List[int]:
        """``[total_len]`` fan-out map: index whose next-token logit predicts token ``t``.

        Within a node, ``prev[t] = t - 1``. A node's FIRST token is predicted from its
        PARENT's LAST token (the fan-out). The very first token of a root has no
        predecessor (``-1``). Depth-1 reduces to "each completion's first token is
        scored from the prompt's last logit."
        """
        prev = [-1] * self.total_len
        for i in range(self.num_nodes):
            s = int(self.node_start[i])
            ln = int(self.node_len[i])
            p = int(self.node_parent[i])
            if p == -1:
                prev[s] = -1  # root's first token: no predecessor
            else:
                prev[s] = int(self.node_start[p]) + int(self.node_len[p]) - 1
            for j in range(1, ln):
                prev[s + j] = s + j - 1
        return prev

    def leaf_nodes(self) -> List[int]:
        """Nodes with no children -- the completions whose tokens carry loss in GRPO."""
        has_child = [False] * self.num_nodes
        for p in self.node_parent:
            if int(p) != -1:
                has_child[int(p)] = True
        return [i for i in range(self.num_nodes) if not has_child[i]]
