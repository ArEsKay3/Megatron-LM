# Experimental vLLM numerical parity branch

The reference is **vLLM**, including its finite-precision rounding. Agreement
with the training forward pass, or greater arithmetic accuracy, is not an
acceptance criterion for this branch. End-to-end parity is **not yet achieved**.

## Base and reference

- Base: Robert Kirby's `880de0fce84321c04a81a052722f1b909014bac6`.
- The separately committed prefill generated-logprob guard preserves the
  working-tree change present in the audited runtime.
- Reference installation: vLLM `0.25.1`, source revision `752a3a504`, PyTorch
  `2.11.0+cu130`, Triton `3.6.0`, NVIDIA GB300, BF16 unquantized Nemotron-H.
- Initial model profile: hidden size 2,688; TP=4; 16 local Mamba heads;
  128-token scan chunks; 128 routed experts, top-6, ReLU squared, scale 2.5.
- Primary target: production compilation and graph execution. Eager execution
  is a separate diagnostic target. Existing prefill captures are eager; passing
  their replay does not validate a compiled whole-model forward.
- Existing engine layouts differ: MINF uses EP=4 and sequence parallelism;
  vLLM uses EP=1 with TP-sharded experts. Resolving that difference is part of
  the work, not an assumption of equivalence.

Record source hashes, weight mapping, dtype/stride, kernel launch configuration,
padding, batch composition, cache state, and chunk boundaries with each test.
Pin the reference installation and execution profile. **Match vLLM's actual
Triton autotuning policy**, including candidate configurations and order, keys,
launch defaults, and cache behavior. Do not pin a production kernel to a measured
winner unless vLLM itself does so. Hyperparameter pinning is allowed only in
separately labeled diagnostic controls. Independent native autotuning can still
select different winners; that remains a parity issue to investigate.

## Acceptance criteria

1. Compare matching logical tokens at each operation boundary, first with
   identical inputs and weights, then in the actual complete forward pass.
2. Require **byte equality**, with dtype and shape equality, for the tested
   activations, recurrent/KV state, logits, and returned logprobs. Report numeric
   error for diagnosis; do not turn a small tolerance into an exactness claim.
3. Validate prefill, partial-chunk continuation, cached decode, and production
   graph/compilation paths. Include short and long histories, mixed lengths,
   and batching controls. Name the profile covered by each result.
4. For stochastic sampling, match filtering, probabilities, random variates,
   and RNG consumption. Start with shared recorded random draws, then verify
   seeded per-request RNG state through scheduling. An equal seed alone or
   equal sampled tokens does not prove parity.
5. Keep raw logprobs and sampling-distribution logprobs distinct. Match actual
   backend dispatch, temperature, top-k/top-p ordering, penalties, masks,
   tie handling, vocabulary padding, stop behavior, and RNG precision.

These are forward/sampling tests. They do not establish the cause of an RL
training or evaluation discrepancy.

## Initial implementation

The initial inference prefill changes implement the verified product-rounding
mechanism and align native launch/autotuning policies with the reference:

- `causal_conv1d_varlen.py` rounds each convolution product to the input dtype
  before FP32 accumulation, matching the pinned vLLM kernel. The previous
  implementation retained FP32 products.
- The convolution uses the reference's fixed 8-token by 256-channel tile,
  two pipeline stages, and default four warps. vLLM does not autotune this
  prefill convolution.
- The five corresponding forward-scan autotuners use vLLM's complete candidate
  lists in the same order, with the same keys and default options. The lists
  have 6 cumsum, 14 chunk-state, 23 chunk-scan, 6 state-passing, and 9 BMM
  configurations. Megatron-specific deterministic candidate filtering is
  removed from these five decorators. There is **no fixed cumsum tile**.
  The auxiliary `_chunk_state_varlen_kernel`, which has no corresponding
  reference decorator and is outside this tested path, is unchanged.

These changes affect this branch's inference defaults. The training kernel is
not changed. Cached single-token convolution and state update need separate
validation; a successful prefill test must not be extrapolated to decode.

`examples/inference/parity/replay_mamba_prefill.py` verifies the installed vLLM
entrypoint source hashes against the supplied capture metadata and reproduces
historical vLLM calls using their recorded configurations. That is a diagnostic
check of capture fidelity. The harness restores the native configuration lists
and caches before comparing engines with identical convolution/scan inputs.
It asserts runtime equality of the five candidate lists and reports the native
winning configurations alongside source hashes and byte comparisons.

`--diagnostic-pin-cumsum` optionally adds a control that temporarily gives MINF
the configuration selected by native vLLM. The harness restores MINF's native
configuration and cache afterward. This control is reported separately and
**cannot make a native-parity failure pass**. The command fails if historical
replay or any native comparison differs.

Run in a CUDA environment exposing both the branch and the pinned vLLM package:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/inference/parity/replay_mamba_prefill.py \
  --minf /path/to/minf-capture-run \
  --vllm /path/to/vllm-capture-run \
  --out-dir /path/to/results
```

The four workers replay independent TP shards; this command does not validate
distributed model collectives. Capture data and private model weights are not
part of the branch.

## Initial validation (2026-09-25)

GB300 job 4013022 completed on four independent shards. Each comparison uses
identical inputs, weights, and incoming state. Five related histories cover
127/128/129-token prefixes, a 5,808-token control, and an 18,964-token history,
including incoming states at their actual engine prefill chunk boundaries.

| Check | Result |
|---|---|
| Historical vLLM call replay, recorded configurations | 56 / 56 calls exact |
| Convolution, reference's native fixed launch | 52 / 52 outputs byte-identical |
| Scan, native autotuning in both engines | 52 / 52 outputs and boundary states byte-identical |
| Separately labeled cumsum-pinning diagnostic | 52 / 52 outputs and states byte-identical |

Both native autotuners selected cumsum `BLOCK_SIZE_H=2` in this run. That choice
was not pinned for the native comparisons. The complete five candidate lists
are checked at runtime against vLLM; their source decorators also match by AST.
This is one hardware/runtime/tuning run, not a guarantee of equal winners in
future processes. The earlier fixed-tile job 4012983 was cancelled and provides
no validation result.

The check stops before gate/normalization and uses common intermediate inputs.
It does not establish complete Mamba-layer, full-model, decode, graph, or
sampling parity. All saved results use zero differing bytes as the criterion;
no numeric tolerance or token-only shortcut was used.

## Grouped gated normalization (2026-09-25)

Inference with gate-before-norm now follows the reference's Torch operation
order and compiles the gate, group reduction, cast, and weight multiplication
together. The weight and group size remain module attributes, so their compiler
specialization matches vLLM's. Passing them as independent dynamic arguments
produced a few differing bytes in the first diagnostic and was rejected.
Training and gate-after-norm retain their existing path.

GB300 job 4013222 passed 96 short-input/shape-control cases in eager and compiled
mode. Job 4013575 then tested the actual `ExtendedRMSNorm.forward` integration
on all 52 captured inputs, including full long-context chunks and original
strides: every compiled output was byte-identical to the independently compiled
installed vLLM method. Eager reference comparisons also passed. The singleton
shape controls are not actual cached-decode validation.

`examples/inference/parity/replay_mamba_norm.py` accepts `--minf`, `--vllm`, and
`--out-dir` for full-tensor capture runs. It preserves input strides, records
reference source hashes and runtime versions, and fails on any differing byte.
Whole-model compilation can change fusion boundaries; its validation remains
separate and pending.

## Remaining work

| Stage | Required proof |
|---|---|
| Embedding, input normalization, projections | Full-token, identical-input equality with matching GEMM shapes and padding |
| Mamba gate/RMSNorm | Actual compiled reference, gate/rounding placement, weights, group reduction |
| Mamba output projection and residual | Partial GEMM, TP reduction, residual-add rounding separately |
| Mamba decode and cache | Same history/state, chunk position, cache precision, graph execution |
| MoE routing | Weight mapping, FP32 logits, bias, top-k IDs/probabilities and tie handling |
| Routed/shared experts | FC1 storage, ReLU squared, FC2 weighting, routed scale, summation, TP/EP layout |
| Attention | Weight/head mapping, Q/K/V, attention backend and cache, projection/reduction |
| Final norm and LM head | Full logits, vocabulary slicing, output dtype and raw logprobs |
| Sampling | Actual backend, processed distribution, shared draws, then RNG/scheduling parity |
| Whole model | Fixed histories and forced tokens first, then greedy and stochastic generation |

Do not claim a row is complete solely because sampled tokens match or an
upstream numerical difference becomes smaller. Preserve intermediate results
so the first remaining divergence can be identified after each change.
