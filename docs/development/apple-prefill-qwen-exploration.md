# Apple Prefill Exploration for Qwen3.5 GGUF on Metal

## Goal

Plan a small, reproducible exploration for **cold prefill** and **general prompt processing**
performance on Apple Silicon in `llama.cpp`, with a focus on **Qwen3.5-style models** from
`Q8_0` down to `Q4_*`.

The intent is:

- stay within normal `llama.cpp` philosophy
- use existing build and benchmark tools
- introduce **no new dependencies**
- produce data that can justify or reject kernel work before we touch the hot path

This is an **exploration draft**, not a proposal to merge an unmeasured optimization.

## Scope

Primary scope:

- Apple Silicon Metal backend
- prompt processing / cold prefill
- Qwen3.5-family GGUFs
- quantizations from `Q8_0` to `Q4_*`

Secondary scope:

- general prefill, including prompt processing under larger context depths
- `flash-attn` interaction
- `n_batch` / `n_ubatch` interaction
- tensor API vs current simdgroup path on Apple9 / M3-class hardware

Out of scope for this draft:

- changing model architecture
- server-side prompt orchestration
- adding external benchmark frameworks
- adding dependencies from MLX, FlashInfer, KeOps, vLLM, etc.

## Why This Matters

For agentic coding workloads, the user-visible latency problem is often **cold or near-cold
prefill**, not decode. Repeated tool schemas, long developer prompts, and growing history make
prompt processing expensive even when decode is acceptable.

On Apple Silicon, `llama.cpp` is already strong, but there is evidence that prompt processing
still has headroom:

- recent Metal tensor API work in upstream `llama.cpp` showed large **prompt-processing**
  improvements on M5 while decode stayed almost unchanged
- MLX-native stacks report better throughput on some Apple workloads, especially when prompt/KV
  reuse is engineered aggressively
- our own live server sampling shows the active server mostly blocked on Metal command-buffer
  completion, with the CPU-side encode path dominated by `ggml_metal_op_mul_mat`

That combination suggests that the most promising local work is still inside the Metal prefill
path itself.

## Current Code Baseline

### CUDA path

The CUDA backend has stronger prefill support today:

- dense/batched matmul can use cuBLAS in `ggml/src/ggml-cuda/ggml-cuda.cu`
- quantized matmul uses dedicated MMQ kernels in
  `ggml/src/ggml-cuda/mmq.cu`
- flash attention uses head-size- and arch-specialized kernels in
  `ggml/src/ggml-cuda/fattn.cu`

This means prompt processing on CUDA benefits from a more mature GEMM/attention stack.

### Metal path

The Metal prompt-processing path is centered around:

- `ggml/src/ggml-metal/ggml-metal.metal`
  - `kernel_mul_mm`
  - `kernel_mul_mm_id`
  - flash-attention kernels
- `ggml/src/ggml-metal/ggml-metal-device.cpp`
  - pipeline selection
- `ggml/src/ggml-metal/ggml-metal-device.m`
  - device capability probing and tensor-API gating

Important current facts from the code:

- `kernel_mul_mm` uses explicit block dequantization into threadgroup memory plus repeated
  barriers and simdgroup matmul
- the source already contains a comment in the dequant/load path:
  `NOTE: this is massively slower.. WTF?`
- the newer Metal tensor API path exists behind `GGML_METAL_HAS_TENSOR`
- on pre-M5 / pre-A19 devices it is **disabled by default** in
  `ggml-metal-device.m`, with the current in-tree note:
  - `M2 Ultra: ~5% slower`
  - `M4, M4 Max: no significant difference`

So the code already tells us two things:

1. the current Metal quantized prefill path still has obvious optimization pressure
2. the straightforward "just enable tensor API everywhere" path is not yet proven on M3-class
   hardware

## What We Observed Locally

### Live production sample

We took a short read-only `sample(1)` profile of the live `llama-server` process during an active
FortBench run on this machine:

- hardware: Apple M3 Ultra
- OS: macOS 26.3.1
- live model: `Qwen3.5-122B-A10B-GGUF:Q8_0`
- server stack: `llama-server` + Responses API

This sample is **not** a pure cold-prefill microbenchmark, but it is still useful as real-world
evidence about where the server spends time when busy.

Main observations from the captured sample:

- `3409 / 3554` main-thread samples were inside
  `ggml_metal_synchronize -> -[_MTLCommandBuffer waitUntilCompleted]`
- the active compute/encode samples underneath `llama_context::decode()` were dominated by:
  - `ggml_metal_graph_compute`
  - `ggml_metal_op_mul_mat`
  - Metal compute enqueue / dispatch
  - command-buffer memory barriers and resource binding
- CPU-side sampling/sampler work was tiny by comparison

Interpretation:

- the server is mostly waiting for GPU work to finish
- the important CPU-visible work in the encode path is still centered on Metal graph encoding and
  quantized `mul_mat`
- this reinforces the idea that the right investigation target is **Metal prefill kernels and
  encode overhead**, not sampling logic

### Device capability snapshot

On this M3 Ultra, the current binaries report:

- `MTLGPUFamilyApple9`
- `MTLGPUFamilyMetal4`
- `has_simdgroup_mm = true`
- `has_tensor = false`

The last point is important: the device family is new enough for the tensor API capability check,
but the current runtime policy disables it by default on pre-M5 hardware.

## Related Upstream Evidence

### `llama.cpp` upstream

Useful recent upstream references:

- `ggml-org/llama.cpp#20962` `Optimize Metal Tensor API usage`
  - reported on M5 Max:
    - `F16 pp512`: `+95%`
    - `Q8_0 pp512`: `+62%`
    - `Q4_0 pp512`: `+58%`
    - decode largely unchanged
- `ggml-org/llama.cpp#21048`
  - follow-up fix for dimension constraints in `matmul2d`
- `ggml-org/llama.cpp#20342`
  - Qwen3.5-related Metal prompt-processing issue on older Apple GPUs, resolved through
    `fused_gdn_ch`
- `ggml-org/llama.cpp#15389`
  - tool calling can materially hurt `llama-server` performance

The main takeaway is:

- upstream has already shown that **prompt processing** can move substantially on Apple without
  changing decode much
- recent Qwen-specific Metal prompt-processing issues exist and have required targeted fixes

### Related projects and algorithms

These are reference points, not dependency candidates:

- **FlashAttention / FlashAttention-2**
  - IO-aware attention kernels
  - useful mainly as a reminder that memory movement often dominates attention performance
- **PagedAttention / vLLM**
  - strongest relevance is general KV and serving design, less direct for single-request cold
    prefill on local Metal
- **FlashInfer**
  - useful reference for kernel families and phase separation: prefill, decode, mixed batching
- **MLX / MLX-LM / vllm-mlx**
  - strongest Apple-native comparisons for prompt/KV handling
- **KeOps**
  - relevant conceptually for symbolic map-reduce style reductions without huge materialized
    matrices
  - less directly applicable to dense quantized transformer GEMMs, but worth keeping in mind as an
    inspiration for reduction-centric experimentation

## Working Hypotheses

These are the hypotheses the exploration should test.

### H1: quantized `mul_mat` is still the main Apple prefill bottleneck

Likely especially true for Qwen3.5 GGUFs in `Q8_0`, `Q6_K`, `Q5_K_*`, `Q4_K_*`.

Why:

- source inspection points to `kernel_mul_mm`
- live sampling points to `ggml_metal_op_mul_mat`
- decode improvements are usually much smaller than prompt-processing improvements in recent Apple
  Metal work

### H2: the current tensor API path is not ready to be enabled by default on M3, but may still be
worth microbenchmarking

Why:

- current in-tree note says no win or slight loss on older chips
- recent upstream work shows real gains on M5
- M3/Apple9 is exactly the kind of hardware where we should measure before deciding

### H3: `flash-attn` is likely secondary for the current local Qwen coding workload

Why:

- cold prefill on large agent prompts is dominated by repeated projection GEMMs and graph encode
- attention still matters, but it is not the first place to start without data

### H4: there is probably a simple, measurable plateau before any risky kernel rewrite

Likely candidates:

- better `n_batch` / `n_ubatch` choices per quant
- explicit tensor-API A/B on Apple9
- verifying whether Q4/Q5/Q6/Q8 behave differently enough to justify quant-specific tuning

## Exploration Principles

Keep this elegant and simple:

- no new dependencies
- no external benchmark harness
- no invasive server modifications
- no one-off notebooks
- everything should run from the repo with shell + built-in tools + Python stdlib only

The baseline tools are already here:

- `tools/llama-bench`
- `tools/batched-bench`
- `sample(1)` on macOS

## Proposed Microbenchmark Plan

### Baseline models

Use smaller Qwen3.5 models first to make iteration practical:

- `Qwen3.5-0.8B`
- `Qwen3.5-4B`
- `Qwen3.5-9B`

If time allows:

- `Qwen3.5-27B`

The goal is not to reproduce production latency exactly. The goal is to identify which prompt
processing trends scale reliably across Qwen3.5 family members and quant levels.

### Quantization matrix

Target quants where available:

- `Q8_0`
- `Q6_K`
- `Q5_K_M`
- `Q4_K_M`

Use whichever exact `Q4_*` / `Q5_*` files exist in the repo for the chosen model.

### Prefill matrix

For each `(model, quant)` pair:

- prompt sizes: `1024`, `4096`, `8192`, `16384`
- `n_gen = 0`
- `flash-attn = 0,1`
- `n_batch / n_ubatch`:
  - baseline: `2048 / 512`
  - variant: `1024 / 256`

### Variants

Start with only three variants:

1. `baseline`
   - current code and current default Metal policy
2. `tensor-enable`
   - same build, but with `GGML_METAL_TENSOR_ENABLE=1`
3. `tensor-disable`
   - same build, but with `GGML_METAL_TENSOR_DISABLE=1`

That keeps the first pass simple and directly answers the biggest immediate question for Apple9 /
M3 hardware.

### Metrics to keep

For each run:

- prompt throughput from `llama-bench`
- full JSONL result line from `llama-bench`
- device capability and runtime policy from stderr
- short hotspot sample from `sample(1)` for at least one representative run per variant

### Success criteria

Initial thresholds:

- `>= 10%` prompt-processing speedup:
  worth continuing
- `>= 20%`:
  strong signal
- `>= 30%`:
  enough to justify targeted kernel work

Stretch goals above that are possible, but should not be assumed on M3 just because M5 data exists.

## Candidate Implementation Directions If The Data Justifies It

Do **not** start here. First measure.

If the measurements justify code work, the likely candidates are:

1. **tensor API path tuning for Apple9 / M3**
   - only if `tensor-enable` beats baseline
2. **quantized `mul_mat` path cleanup**
   - reduce explicit staging / barrier pressure
   - improve dequant + matmul fusion for Q8/Q6/Q5/Q4
3. **quant-specific tuning**
   - if Q8 and Q4 have materially different best settings
4. **prefill-oriented graph encode cleanup**
   - only if sampling shows CPU-side encode overhead is still significant after GPU work improves

## What This Draft Does Not Claim

This draft does **not** claim:

- that MLX or another engine should replace `llama.cpp`
- that KeOps should be imported
- that the tensor API path should be enabled on M3 by default today
- that we already know which kernel rewrite is best

It only claims that we now have enough code evidence and upstream context to justify a disciplined,
small exploration.

## Deliverables

This exploration draft is paired with three simple helpers:

- `scripts/apple/bench-prefill-qwen.sh`
  - runs the prompt-processing matrix
- `scripts/apple/sample-prefill-hotpath.sh`
  - captures a short `sample(1)` hotspot profile for either a live PID or a spawned benchmark
- `scripts/apple/summarize-prefill-jsonl.py`
  - reduces raw `llama-bench` JSONL into CSV or Markdown summaries

These helpers are deliberately minimal and dependency-free.

## References

### `llama.cpp`

- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.m`
- `ggml/src/ggml-metal/ggml-metal-device.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/fattn.cu`
- `tools/llama-bench`
- `tools/batched-bench`

### Upstream GitHub references

- `ggml-org/llama.cpp#15389`
  - https://github.com/ggml-org/llama.cpp/issues/15389
- `ggml-org/llama.cpp#19431`
  - https://github.com/ggml-org/llama.cpp/issues/19431
- `ggml-org/llama.cpp#20342`
  - https://github.com/ggml-org/llama.cpp/issues/20342
- `ggml-org/llama.cpp#20962`
  - https://github.com/ggml-org/llama.cpp/pull/20962
- `ggml-org/llama.cpp#21048`
  - https://github.com/ggml-org/llama.cpp/pull/21048

### Related projects

- MLX-LM
  - https://github.com/ml-explore/mlx-lm
- vllm-mlx
  - https://github.com/waybarrios/vllm-mlx
- FlashInfer
  - https://github.com/flashinfer-ai/flashinfer
- KeOps
  - https://github.com/getkeops/keops

### Papers / algorithm references

- FlashAttention
  - https://arxiv.org/abs/2205.14135
- FlashAttention-2
  - https://arxiv.org/abs/2307.08691
- PagedAttention / vLLM
  - https://arxiv.org/abs/2309.06180
- Native LLM and MLLM Inference at Scale on Apple Silicon
  - https://arxiv.org/abs/2601.19139
