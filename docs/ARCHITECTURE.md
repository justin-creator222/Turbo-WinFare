# Architecture

How Turbo-WinFare is put together, and the details that are easy to get subtly wrong.

## Targets

From [CMakeLists.txt](../CMakeLists.txt):

* `turbo_engine_lib` (static) and `turbo_engine` (shared, `libturbo_engine.dll`) — the same
  engine core, built twice. Every executable links one of them.
* `turbo-winfare` — the CLI/GUI host, [src/main.cpp](../src/main.cpp).
* Eight test executables. They are standalone `main()`s with no framework, so any of them can be
  run directly: `.\build\test_gpu_kernels.exe`.

The engine links only system libraries: `d3d12 dxgi d3dcompiler dxguid ws2_32`. Nothing is
vendored.

## Engine core

Everything lives in the `gturbo::` namespace, with headers under [include/gturbo/](../include/gturbo/).

| Component | File | Responsibility |
|---|---|---|
| `ForwardRunner` | [src/runner.cpp](../src/runner.cpp) | Orchestrator. Owns the decode loop, resident weight buffers, and per-layer expert streamers. Entry points: `produce_token`, `generate`, `generate_text`, `generate_tokens`, `generate_chat`. |
| `D3D12Context` | [src/d3d12_context.cpp](../src/d3d12_context.cpp) | Device and queue setup, UMA buffer allocation, memory telemetry. |
| `ComputePipelineManager` | [src/pipeline.cpp](../src/pipeline.cpp) | Compiles the HLSL in [shaders/](../shaders/) at runtime via DXC and dispatches it. |
| `ExpertStreamer` | [src/streamer.cpp](../src/streamer.cpp) | Per-layer NVMe → UMA expert loading with an LFU/LRU DRAM slot cache. Opened lazily per layer. |
| `KVCacheManager` | [src/kv_cache.cpp](../src/kv_cache.cpp) | The per-layer FP32 K/V buffers and ring indexing (`physical_slot`) for **both** the GPU and CPU paths. `ForwardRunner` holds no KV buffers of its own. |
| `Tokenizer` | [src/tokenizer.cpp](../src/tokenizer.cpp) | HF `tokenizer.json` BPE with byte fallback. A default-constructed `Tokenizer` has no vocabulary and throws on `encode`. |
| Sampling | [src/sampling.cpp](../src/sampling.cpp) | Truncation, temperature, repetition penalty. One implementation, shared by both paths. |
| Detokenizer | [src/detokenizer.cpp](../src/detokenizer.cpp) | Streaming detokenization and `StreamingStopMatcher` for string stop sequences. |
| Format layer | [src/manifest.cpp](../src/manifest.cpp), [src/packed_experts.cpp](../src/packed_experts.cpp), [src/resident_index.cpp](../src/resident_index.cpp) | `GTurboManifestV1`, `PackedExpertsLayoutV1`, `ResidentIndexCodec`. Constants in [include/gturbo/format.hpp](../include/gturbo/format.hpp). |
| HTTP + API | [src/http.cpp](../src/http.cpp), [src/server.cpp](../src/server.cpp), [src/openai_api.cpp](../src/openai_api.cpp) | Request framing (Content-Length, chunked, 1 MiB limit, 413/415), the GUI server, and the OpenAI-compatible endpoints. |
| C ABI | [src/c_api.cpp](../src/c_api.cpp), [include/gturbo/c_api.h](../include/gturbo/c_api.h) | The stable `extern "C"` embedding boundary, exported from `libturbo_engine.dll`. |
| Model fetch | [src/model_fetch.cpp](../src/model_fetch.cpp) | Drives `tools/convert_hf_to_gturbo.py` as a child process for the in-GUI download, and probes the host for Python, the converter and free disk. |
| CPU reference | [src/cpu_reference.cpp](../src/cpu_reference.cpp) | Scalar FP32 forward pass. Touches no D3D12. Ground truth for the GPU path. |

`turbo-winfare.exe` is the only front-end. A Python `http.server` bridge (`run_gui.py`) used to
sit beside it; it was removed because it implemented neither `/v1/*` nor `GET /api/config`, which
the browser GUI requires, so generation under it never worked at all. The C ABI it bound to is
unaffected and still ships in `libturbo_engine.dll`.

The server binds `127.0.0.1` by default. It has no authentication, sends
`Access-Control-Allow-Origin: *`, and can load models and start multi-GB downloads, so exposing it
on a network is an explicit opt-in via `--host 0.0.0.0`.

## The per-layer dispatch graph

`produce_token` issues **two command lists per layer**, not one. The router's top-8 selection has
to be read back on the CPU before the experts it names can be streamed off disk:

```
[ norm -> QKV -> epilogue -> attention -> o_proj -> PostAttn -> router -> topK ]  submit, fence
        read back top-8 indices  ->  stream those 8 expert blocks
[ shared expert -> 8 routed experts -> LayerTail ]                                submit, fence
```

Part 2 is split again: the shared expert (which depends only on `dense_x`) and any experts
already resident in cache are submitted *before* the miss reads are issued, so GPU work overlaps
disk I/O. The result is ~92 submissions but only **31 fence waits** per token — one per layer
plus the head. Fence count is what costs, not submission count.

**Expert slots must outnumber the routed experts.** `plan_experts` pins every slot it returns and
`release_plan` unpins exactly those. Without pinning, a pool of 8 slots serving 8 routed experts
could evict its own earlier entries, and the caller would bind one expert's weights in place of
another's — output stays fluent, just wrong. The observed symptom was
*"The capital of France is **_** (This meaning is a small enough verb form sentence…"* — correct
prefix, then drift. [tests/test_streamer.cpp](../tests/test_streamer.cpp) pins this down,
including the case where one plan must not release another's slots.

## HLSL kernels

Shader Model 6.6, compiled at runtime by DXC. All 15 are verified against the CPU reference by
`test_gpu_kernels`:

`EmbedLookup`, `RMSNormK`, `GemvInt4`, `GemvInt8`, `QKVEpilogue`, `Attention`, `RouterTopK`,
`GeGLU`, `PostAttn`, `LayerTail`, `Softcap`, `ScaleAccum`, `MulBF16`, `LMHeadGreedy`,
`ArgmaxReduce`.

**All bindings are raw byte-address buffers with explicit byte offsets**
([shaders/Common.hlsli](../shaders/Common.hlsli)). An earlier generation mixed
`StructuredBuffer<float16_t>` (stride 2) with `Texture2D<uint>` declarations while the host bound
everything as stride-4 structured buffers, so the views silently disagreed with the declarations.
Raw buffers eliminate that class of bug entirely.

Activations are FP32. FP16 packing is a deliberate non-goal for now: it would make GPU-vs-CPU
diffs ambiguous and cost the token-for-token agreement gate.

### Two rules for kernel work

**Do not assume a wave width.** RDNA 3 runs a 256-thread group as **Wave64, not Wave32**. Any
kernel reducing across waves must use `WaveGetLaneCount()` and `WaveIsFirstLane()`. The first
`RMSNormK` hardcoded 32, wrote each wave's total into two groupshared slots, and produced output
scaled by exactly `1/sqrt(2)` — plausible-looking by inspection, caught immediately by
`test_gpu_kernels`.

**Never save a shader as UTF-8 with a BOM.** DXC rejects it with a misleading "non-ASCII
characters are not allowed" pointing at the wrong line. PowerShell's `Set-Content -Encoding utf8`
adds one silently — use `-Encoding ascii`. The loader strips a leading BOM defensively, but the
error is confusing enough to be worth avoiding.

Agreement with the CPU reference is judged with numpy's `allclose` rule
(`|got-want| <= 1e-5 + 1e-4*|want|`), not a pure relative test. GPU transcendentals differ from
libm by a couple of ULPs, and near-zero results then show a large *relative* error for ~1e-7
absolute. A real logic bug is never marginal — the Wave64 bug was off by 0.29 relative,
everywhere.

## Forward-pass constants that are easy to get subtly wrong

All verified against the reference's Metal kernels and against the checkpoint — not inferred.
Each of these produces fluent-but-wrong output if missed, which is the hardest failure to debug.

* **Attention scale is 1.0**, not `1/sqrt(head_dim)`. The query scaling is absorbed into `q_norm`.
* **RMSNorm applies `w`, not `1 + w`** (eps 1e-6). The offset is baked into the checkpoint.
* **Dequantization is `w = q * scale + bias`** (MLX affine, group 64), packed low-nibble-first.
* **Full-attention layers have no `v_proj`.** V reuses the raw `k_proj` output, takes a *no-scale*
  RMSNorm, and skips RoPE entirely.
* **Partial rotary applies to full layers only**: 64 of 256 pairs (`0.25 * 512 / 2`). Sliding-window
  layers rotate all 128.
* **Embeddings are scaled by `sqrt(2816)`.** The LM head is the same matrix (tied).
* **Router**: input is `rmsnorm_no_scale(hidden) * router.scale / sqrt(D)`, 8-bit weights, softmax
  over the **top-8 only**, times `per_expert_scale`. Ties go to the lower expert index.
* **Three distinct pre-FFN views**: the shared expert reads `pre_feedforward_layernorm`, the routed
  experts read `pre_feedforward_layernorm_2`, and the router reads the unscaled norm.
* **The shared expert is added with no routing weight.**
* **Sampling order is Top-P → Top-K → temperature**, not HuggingFace's order. `top_p < 1` with
  `temperature > 0` **requires** a `top_k` in 1..256; full-vocabulary nucleus sampling is
  deliberately unimplemented upstream and the shortlist optimization depends on that bound.
* **The repetition penalty is applied to already-softcapped logits.** Applied to raw logits it is
  very nearly a no-op, because real Gemma 4 logits reach the hundreds and sit deep in tanh
  saturation.

## KV cache

`KVCacheManager` owns FP32 K/V for both paths and indexes them as a ring:
`physical_slot(L, p) = p % capacity`. Sliding-window layers therefore need only `sliding_window`
slots regardless of total context, and only the 5 full-attention layers scale with context. That
is what makes 4096 tokens cost ~560 MB instead of ~1.85 GB.

Context cannot go above 4096 without first raising `ATTN_MAX_SPAN` in
[shaders/Attention.hlsl](../shaders/Attention.hlsl). Scores are staged in groupshared memory,
which is capped at 32 KB, so anything beyond ~8000 needs an online-softmax rewrite that never
materializes the full score span.

## Chat template

The Gemma 4 chat template is hardcoded C++ in [src/tokenizer.cpp](../src/tokenizer.cpp), using
Gemma 4's `<|turn>` / `<turn|>` markers (**not** Gemma 2/3's `<start_of_turn>`). The bundle's
`chat_template.jinja` is shipped but never read. That is deliberate — there is no Jinja engine
here — but it means a bundle with a different template would render silently wrong.
