# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Determinism coverage for the local parity additions on this audit branch.

This branch predates the repository-wide kernel manifest. Keep these entries
when merging into that registry. Tests cover the BF16 forward profile on GB300;
vendored backward/other-hardware implementations are not supported by this mode.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelEntry:
    name: str
    sources: tuple[str, ...]
    tests: tuple[str, ...] = ()
    kind: str = ''
    training_path: bool = False
    notes: str = ''
    exempt_reason: str = ''


KERNELS = (
    KernelEntry(
        name='parity_cuda_ops',
        sources=(
            'megatron/core/inference/parity_kernels/ops.py',
            'megatron/core/inference/parity_kernels/csrc/bindings.cpp',
            'megatron/core/inference/parity_kernels/csrc/core/batch_invariant.hpp',
            'megatron/core/inference/parity_kernels/csrc/cuda_compat.h',
            'megatron/core/inference/parity_kernels/csrc/custom_all_reduce.cuh',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/core/math.hpp',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/cub_helpers.h',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/custom_all_reduce.cu',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/dispatch_utils.h',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/layernorm_kernels.cu',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/moe/grouped_topk_kernels.cu',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/moe/moeTopKFuncs.cuh',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/moe/moe_ops.h',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/ops.h',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/quantization/vectorization.cuh',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/quantization/vectorization_utils.cuh',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/torch_utils.h',
            'megatron/core/inference/parity_kernels/csrc/libtorch_stable/type_convert.cuh',
        ),
        tests=("tests/unit_tests/determinism/kernels/test_local_parity_kernels.py",),
        kind='cuda-ext',
        notes='Norm, MoE routing/alignment/sum and IPC all-reduce: eager/graph byte comparisons with changing inputs.',
    ),
    KernelEntry(
        name='parity_moe',
        sources=(
            'megatron/core/inference/parity_kernels/moe.py',
            'megatron/core/inference/parity_kernels/moe_gemm.py',
        ),
        tests=("tests/unit_tests/determinism/kernels/test_local_parity_kernels.py",),
        kind='triton',
        notes='BF16 ReLU-squared forward, including small-token direct assignment and aligned expert batches.',
    ),
    KernelEntry(
        name='parity_collectives',
        sources=(
            'megatron/core/inference/parity_kernels/collectives.py',
            'megatron/core/inference/parity_kernels/symm_mem.py',
            'megatron/core/inference/parity_kernels/collective_sizes.py',
        ),
        tests=("tests/unit_tests/determinism/kernels/test_local_parity_kernels.py",),
        kind='dispatch',
        notes='Four-rank NVLink custom, symmetric-memory and NCCL paths. Run with production NCCL environment.',
    ),
    KernelEntry(
        name='parity_fa4',
        sources=(
            'megatron/core/inference/parity_kernels/cute/__init__.py',
            'megatron/core/inference/parity_kernels/cute/ampere_helpers.py',
            'megatron/core/inference/parity_kernels/cute/barrier.py',
            'megatron/core/inference/parity_kernels/cute/bench_utils.py',
            'megatron/core/inference/parity_kernels/cute/benchmark.py',
            'megatron/core/inference/parity_kernels/cute/benchmark_flash_attention_fp8.py',
            'megatron/core/inference/parity_kernels/cute/blackwell_helpers.py',
            'megatron/core/inference/parity_kernels/cute/block_info.py',
            'megatron/core/inference/parity_kernels/cute/block_sparse_utils.py',
            'megatron/core/inference/parity_kernels/cute/block_sparsity.py',
            'megatron/core/inference/parity_kernels/cute/cache_utils.py',
            'megatron/core/inference/parity_kernels/cute/compute_block_sparsity.py',
            'megatron/core/inference/parity_kernels/cute/copy_utils.py',
            'megatron/core/inference/parity_kernels/cute/cute_dsl_ptxas.py',
            'megatron/core/inference/parity_kernels/cute/cute_dsl_utils.py',
            'megatron/core/inference/parity_kernels/cute/fa_logging.py',
            'megatron/core/inference/parity_kernels/cute/fast_math.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd_postprocess.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd_preprocess.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd_sm100.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd_sm120.py',
            'megatron/core/inference/parity_kernels/cute/flash_bwd_sm90.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd_combine.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd_mla_sm100.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd_sm100.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd_sm120.py',
            'megatron/core/inference/parity_kernels/cute/flash_fwd_sm90.py',
            'megatron/core/inference/parity_kernels/cute/hopper_helpers.py',
            'megatron/core/inference/parity_kernels/cute/interface.py',
            'megatron/core/inference/parity_kernels/cute/mask.py',
            'megatron/core/inference/parity_kernels/cute/mma_sm100_desc.py',
            'megatron/core/inference/parity_kernels/cute/named_barrier.py',
            'megatron/core/inference/parity_kernels/cute/pack_gqa.py',
            'megatron/core/inference/parity_kernels/cute/paged_kv.py',
            'megatron/core/inference/parity_kernels/cute/pipeline.py',
            'megatron/core/inference/parity_kernels/cute/seqlen_info.py',
            'megatron/core/inference/parity_kernels/cute/sm100_hd256_2cta_fmha_backward.py',
            'megatron/core/inference/parity_kernels/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py',
            'megatron/core/inference/parity_kernels/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py',
            'megatron/core/inference/parity_kernels/cute/sm100_hd256_2cta_fmha_forward.py',
            'megatron/core/inference/parity_kernels/cute/sm90_config_search.py',
            'megatron/core/inference/parity_kernels/cute/softmax.py',
            'megatron/core/inference/parity_kernels/cute/testing.py',
            'megatron/core/inference/parity_kernels/cute/tile_scheduler.py',
            'megatron/core/inference/parity_kernels/cute/topk_gather_kv.py',
            'megatron/core/inference/parity_kernels/cute/utils.py',
        ),
        tests=("tests/unit_tests/determinism/kernels/test_local_parity_kernels.py",),
        kind='external-lib',
        notes='Forward BF16 paged-KV attention only; replay varies sequence length across page boundaries up to 196480. Backward and other dtypes/hardware helpers are retained for upstream interface compatibility and are outside this profile.',
    ),
)
