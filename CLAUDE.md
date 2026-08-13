# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`Turbo-WinFare` (binary `turbo-winfare`, formerly `turbo-fieldfare-win`) is a native Windows 11 / AMD C++23 + DirectX 12 (DirectCompute) inference engine for **Gemma 4 26B-A4B** (MoE: 30 layers, 128 routed + 1 shared expert, ~3.88B active params/token). It targets the Lenovo Legion Go S (Ryzen Z1 Extreme APU, Radeon 780M, LPDDR5X Unified Memory). The design goal is a zero-copy streaming-MoE pipeline: expert weights stream from NVMe directly into host-coherent UMA memory (`D3D12_HEAP_TYPE_CUSTOM` + `WRITE_BACK`) so GPU compute shaders read them with no staging copy.

This C++ project is a **port of the Swift 6.2 / Metal reference implementation,
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare)** (Apache-2.0, Andrey
Mikhaylov). That project is the source-of-truth baseline for correctness and memory budgets. It
is **not vendored here** -- clone it separately if you want to read or diff against it, and treat
any local checkout as read-only: do not edit, build, or run the Swift model unless the user
explicitly asks. Attribution and the Apache-2.0 notice of modification live in [NOTICE](NOTICE).

## Build & test

The build uses **CMake + Ninja + MinGW g++ from `C:\w64devkit`**, NOT MSVC. Ninja can come from
anywhere on `PATH` (a Python venv install works fine).

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe
cmake --build build --config Release
ctest --test-dir build --output-on-failure   # test_format, test_tokenizer, test_gpu_kernels, test_convert
```

Run a single test binary directly (they are standalone, no framework):

```powershell
.\build\test_format.exe
.\build\test_gpu_kernels.exe
.\build\test_tokenizer.exe
```

CMake copies `gui/` and `shaders/` into the build directory on every build; shaders are resolved at runtime relative to the executable.

Toolchain bootstrap: `python tools/download_toolchain.py` installs w64devkit to `C:\w64devkit` **and** fetches `dxcompiler.dll` + `dxil.dll` into `build/`. Both are required -- **`C:\w64devkit\bin` must be on `PATH`** or g++ fails with `cannot execute 'as'`, and without DXC every shader fails to compile (there is deliberately no `cs_5_0` fallback).

## Running the app

Two front-ends both talk to the same engine core:

- **Native (canonical)**: `turbo-winfare.exe` (from [src/main.cpp](src/main.cpp)) starts an embedded C++ HTTP server ([src/server.cpp](src/server.cpp)) on `--port` (default 8080) and opens a browser. `--help` lists every flag; unknown arguments are an error (exit 2), not silently ignored.

  | Flag | Effect |
  |---|---|
  | `--model <dir>` | bundle directory (default `gemma-4-26b-a4b.gturbo`) |
  | `--prompt <text>` | one-shot CLI generation instead of the GUI |
  | `--gui` / `--no-open` | force the GUI / suppress the browser launch |
  | `--cpu` | scalar FP32 reference path; requires `--prompt` |
  | `--max-tokens <n>` | completion budget (both paths) |
  | `--temperature <f>` | 0 = greedy (both paths) |
  | `--context <n>` / `--slots <n>` | 0 = auto-size from installed RAM |
  | `--dump-tensors <dir>` | per-stage FP32 tensors for token 0 (`--cpu` only) |
  | `--port <n>` | GUI port |

- **Python bridge**: `python run_gui.py` -- an `http.server` that loads `libturbo_engine.dll` via `ctypes` and calls the C API. Kept as a convenience only; it is **not** at parity with the native server and does not push its config to the engine.

The GUI static assets live in [gui/](gui/) (`index.html`, `app.js`, `styles.css`).

**The GUI assets are UTF-8 and the server declares `charset=utf-8` on every one of them.**
`index.html` had been saved after a round-trip through cp1252, so its emoji were stored
double-encoded (`⚡` on disk as the literal three characters `âš¡`) -- valid UTF-8, so no tool
complained, and the browser faithfully rendered the mojibake. Rewriting a GUI file with an
editor or shell that guesses the encoding will silently reintroduce this; PowerShell's
`Set-Content` without `-Encoding utf8` is the usual culprit, the same trap as the shader BOM
above.

The sidebar seeds itself from `GET /api/config`, which reports the engine's *resolved*
context and slot count (a stored 0 means "auto-size from RAM"). Values hardcoded in
`index.html` are placeholders only -- before hydration existed the panel advertised a 62000
token context while the engine was auto-sized to 4096.

### OpenAI-compatible API

`--serve` starts the same server without opening a browser. `GET /health`,
`GET /v1/models`, and `POST /v1/chat/completions` (JSON or SSE) are implemented in
[src/openai_api.cpp](src/openai_api.cpp), matching the subset the reference supports
(the reference's `docs/OPENAI_SERVER.md`).

```powershell
.\build\turbo-winfare.exe --serve --port 8099
```
```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="unused")
c.chat.completions.create(model="gemma-4-26b-a4b-it",
                          messages=[{"role": "user", "content": "Hi"}], stream=True)
```

Unsupported parameters are **rejected, not ignored** (`n != 1`, `logprobs`,
presence/frequency penalties, `tool_choice: required`, image content parts). Silently
accepting one is worse than a 400, because the caller believes it applied. Generation is
serialized by a `RequestCoordinator` -- one at a time, `--queue-limit` waiters in strict FIFO,
then `429 queue_full`.

## Architecture

Build artifacts from [CMakeLists.txt](CMakeLists.txt):
- `turbo_engine_lib` (static) / `turbo_engine` (shared `libturbo_engine.dll`) -- the engine core, shared by every target.
- `turbo-winfare` -- CLI/GUI host.

(`turbo-repack-win` is gone -- it only ever wrote zero-filled placeholder bundles. Use
`python tools/convert_hf_to_gturbo.py`. The in-app repacker UI and `/api/repack` went with it.)

Engine core (all in `gturbo::` namespace, headers under [include/gturbo/](include/gturbo/)):
- `ForwardRunner` ([src/runner.cpp](src/runner.cpp)) -- orchestrator; owns the token decode loop, resident weight buffers, KV cache, and per-layer expert streamers. `produce_token` / `generate` / `generate_text` are the entry points.
- `D3D12Context` ([src/d3d12_context.cpp](src/d3d12_context.cpp)) -- device/queue setup, UMA buffer allocation, memory telemetry.
- `ComputePipelineManager` ([src/pipeline.cpp](src/pipeline.cpp)) -- loads/compiles the HLSL kernels in [shaders/](shaders/) and dispatches them.
- `ExpertStreamer` ([src/streamer.cpp](src/streamer.cpp)) -- per-layer NVMe->UMA expert loading with an LFU/LRU DRAM cache; opened lazily per layer.
- `KVCacheManager` ([src/kv_cache.cpp](src/kv_cache.cpp)) -- owns the per-layer FP32 K/V buffers and the ring indexing (`physical_slot`) for **both** paths. `ForwardRunner` holds no KV buffers of its own.
- `Tokenizer` ([src/tokenizer.cpp](src/tokenizer.cpp)) -- HF `tokenizer.json` BPE with byte fallback, Gemma 4 `<|turn>`/`<turn|>` markers (NOT Gemma 2/3 `<start_of_turn>`). A default-constructed `Tokenizer` holds no vocabulary and throws on `encode`.
- Format layer: `GTurboManifestV1` ([src/manifest.cpp](src/manifest.cpp)), `PackedExpertsLayoutV1` ([src/packed_experts.cpp](src/packed_experts.cpp)), `ResidentIndexCodec` ([src/resident_index.cpp](src/resident_index.cpp)), constants in [include/gturbo/format.hpp](include/gturbo/format.hpp).
- `HTTPServer` ([src/server.cpp](src/server.cpp)) and the `extern "C"` ABI in [src/c_api.cpp](src/c_api.cpp) (header [include/gturbo/c_api.h](include/gturbo/c_api.h)) -- the stable boundary `run_gui.py` binds to.

### HLSL kernels ([shaders/](shaders/))

SM 6.6, compiled at runtime by DXC. All bindings are **raw byte-address buffers** with
explicit byte offsets ([shaders/Common.hlsli](shaders/Common.hlsli)) -- the old kernels mixed
`StructuredBuffer<float16_t>` (stride 2) and `Texture2D<uint>` declarations while the host
bound everything as stride-4 structured buffers, so views silently disagreed with
declarations. Activations are FP32 for now; FP16 packing is a later optimization that would
make GPU-vs-CPU diffs ambiguous.

**All 15 kernels are verified** against the CPU reference by `test_gpu_kernels`:
`EmbedLookup`, `RMSNormK`, `GemvInt4`, `GemvInt8`, `QKVEpilogue`, `Attention`, `RouterTopK`,
`GeGLU`, `PostAttn`, `LayerTail`, `Softcap`, `ScaleAccum`, `MulBF16`, `LMHeadGreedy`,
`ArgmaxReduce`.

Agreement is judged with numpy's `allclose` (`|got-want| <= 1e-5 + 1e-4*|want|`), not a pure
relative test. GPU transcendentals differ from libm by a couple of ULPs and near-zero results
then show a big relative error for ~1e-7 absolute -- while a real logic bug is never marginal
(the Wave64 bug below was off by 0.29 relative, everywhere).

**Do not assume a wave width.** RDNA 3 runs a 256-thread group as Wave64, not Wave32. A
kernel that reduces across waves must use `WaveGetLaneCount()` / `WaveIsFirstLane()` -- the
first version of `RMSNormK` hardcoded 32, wrote each wave's total into two groupshared slots,
and produced output scaled by exactly `1/sqrt(2)`. `test_gpu_kernels` caught it immediately;
by inspection it looked like a plausible activation.

`build/shaders/` is refreshed on every **build** (not at configure time), so editing a `.hlsl`
takes effect on the next `cmake --build`.

**Never save a shader as UTF-8 with a BOM.** DXC rejects it with a misleading "non-ASCII
characters are not allowed" pointing at the wrong line. PowerShell's
`Set-Content -Encoding utf8` adds one silently -- use `-Encoding ascii`. The loader strips a
leading BOM defensively, but the error is confusing enough to be worth avoiding.

## `.gturbo` model format

A `.gturbo` "model" is a directory (see [gemma-4-26b-a4b.gturbo/](gemma-4-26b-a4b.gturbo/)):

```
manifest.json                    arch + per-file size/sha256
resident.bin                     16 KB index region, then all non-expert weights
tokenizer/tokenizer.json         HF BPE, 262,144 entries (plus config/chat template)
packed_experts/layout.json       one expertBlock description, not 3,840 copies
packed_experts/layer_00..29.bin  128 blocks of 3,358,720 B each
```

**Every sub-tensor and expert block is aligned to 16 KB** (`GTurboFormatV1::ALIGNMENT_BYTES`) so reads land on sector boundaries. Note the streamer deliberately does NOT use `FILE_FLAG_NO_BUFFERING` -- see the Performance section.

Weights are **MLX affine 4-bit, group 64, BF16 scale + BF16 bias** (routers 8-bit), copied through from the checkpoint unchanged -- there is no requantization step, and the "Q4_K_M group 32" scheme the old code described does not exist in this model.

Resident tensor names are the HuggingFace names with `language_model.` stripped, e.g. `model.layers.0.self_attn.q_proj.weight`. A quantized tensor is one index entry whose `scale_offset`/`bias_offset` point at its `.scales`/`.biases` regions.

## Current state -- read before assuming behavior works

**The engine generates correct Gemma 4 text on both the CPU reference and the GPU path, and
the two agree token-for-token under greedy decoding.** The forward pass is done. What is
still missing is the product surface above it -- see "Parity with the reference" below.

- **Stage 0 (done)** -- every fabricating path removed. No canned replies, no prompt echo, no
  synthesized vocabulary, no `cs_5_0` shader downgrade, no invented telemetry, no in-app
  repacker writing placeholder bundles. Missing weights/tokenizer/DXC are now hard errors.
- **Stage 1 (done)** -- real data pipeline. [tools/convert_hf_to_gturbo.py](tools/convert_hf_to_gturbo.py) now streams the pinned checkpoint (`mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464e`) into a 13.3 GB bundle without ever holding a shard in memory; the tokenizer is a real `tokenizer.json` BPE with byte fallback; `manifest.json`/`layout.json` are actually parsed.
- **Stage 2 (done)** -- [src/cpu_reference.cpp](src/cpu_reference.cpp) is a scalar FP32 forward pass that **produces correct Gemma 4 text**. It touches no D3D12 and is the ground truth for Stage 3.
- **Stage 3 (done)** -- the DirectCompute path works. All 15 kernels are verified against the CPU reference, and **GPU greedy output matches CPU greedy output token-for-token**.
- **Stage 4 (done)** -- performance. **2.34 -> 8.8 tok/s decode**, beating the Swift/Metal reference's 5.1-6.3 tok/s on an M2 Air by ~40%.

```powershell
.\build\turbo-winfare.exe --prompt "What is the capital of France?"
#   -> The capital of France is **Paris**.   (~8.8 tok/s decode)
```

## Parity with the reference -- what is actually missing

The forward pass matches. The product surface around it does not. Audited against the Swift
reference; in progress, roughly in this order:

| Gap | Status |
|---|---|
| Sampling (`temperature`/`top_p`/`top_k`/`repetition_penalty`/`seed`) | **done** -- [src/sampling.cpp](src/sampling.cpp), shared by both paths |
| String stop sequences | **done** -- `StreamingStopMatcher`, [src/detokenizer.cpp](src/detokenizer.cpp) |
| Streaming token output (engine + CLI) | **done** -- `generate_tokens` + `StreamCallback`; SSE still to come |
| Multi-turn engine entry point | **done** -- `generate_chat`; front-ends do not send history yet |
| Thought-channel suppression -- the generation prompt opens a `thought` channel and nothing parses it | not started |
| KV reuse across turns (prompt cache) -- every request re-prefills from scratch | not started |
| OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`, `/health`, SSE) | **done** -- [src/openai_api.cpp](src/openai_api.cpp) |
| Real HTTP framing (Content-Length, chunked, 1 MiB limit, 413/415) | **done** -- [src/http.cpp](src/http.cpp) |
| Tool calling and channel decoding -- note the generation prompt already opens a `thought` channel and **nothing suppresses it** | not started |
| Runtime integrity verification, symlink rejection | not started |
| Chunked/batched prefill (~5.7 tok/s here vs ~28 in the reference) | deferred |
| FP16 KV and context above 4096 | deferred |

`chat_template.jinja` ships in the bundle and is never read -- the template is hardcoded C++
([src/tokenizer.cpp](src/tokenizer.cpp)). That is deliberate, but it means a bundle with a
different template would render silently wrong.

## Performance

Measured on a 24 GB Ryzen/Radeon dev box, 24-token generation. `--slots N` overrides the
auto-sized expert cache; the breakdown below prints on every CLI run.

| Stage | decode | expert I/O | GPU wait | CPU other |
|---|---|---|---|---|
| Stage 3 baseline | 2.34 tok/s | 237 ms | 152 ms | 38 ms |
| + buffered I/O, parallel reads, skip-read-on-hit | 4.72 | 87 | 117 | 26 |
| + auto-sized slot pool (24/layer) | 5.66 | 44 | 115 | 24 |
| + allocator ring, 62 -> 31 fences | 6.04 | 44 | 103 | 29 |
| + hit-first split, shared expert overlapping I/O | 8.06 | 47 | 62 | 35 |
| + descriptor cache | 8.89 | 48 | 60 | 23 |
| + fused greedy LM head | **8.8-9.0** | 48 | 60 | 23 |

Reference points: M2 Air 5.1-6.3 tok/s; our 16-slot config (the 16 GB Legion Go S target)
reaches **6.54 tok/s**, so the reference is beaten on the constrained configuration too.

**Where the time goes now:** ~34% expert I/O, ~41% GPU wait, ~9% LM head, ~16% CPU. The
engine is still nowhere near compute- or DRAM-bound (Stage 3 measured 0.2% of peak ALU and
~6% of DRAM bandwidth), so the remaining headroom is in kernel efficiency, not hardware.

### Things that turned out to matter

- **The expert cache was counting hits it never acted on.** `load_expert` called
  `read_expert_unbuffered` unconditionally, so a 51.6% hit rate still re-read every byte.
  Skipping the read on a hit halved I/O volume outright.
- **`FILE_FLAG_NO_BUFFERING` was throwing away the OS page cache.** The reference leans on
  the equivalent macOS cache deliberately (`docs/SYSTEM_DESIGN.md:168`). Removing it, plus
  issuing the 8 misses concurrently instead of at queue depth 1, took expert I/O 237 -> 87 ms.
- **Overlapping GPU work with the reads was the single biggest scheduling win** (6.04 -> 8.06).
  The shared expert depends only on `dense_x`, and cache-hit experts need no I/O at all, so
  both are submitted *before* the reads are issued.
- **Slot pools do not scale freely.** They compete with the OS page cache for the same RAM.
  The pool is auto-sized from installed RAM (16/24/32 slots at <=16/24/>=32 GB); the
  reference measured 32 slots collapsing their 8 GB host from 5.6 to 1.58 tok/s.

### Measured and rejected

- **A second compute queue.** Before building one, the precondition was tested by giving the
  shared expert its own scratch buffers, removing the false dependency that serialized it
  against the routed experts. Result: **no change** (8.75-8.79 vs 8.59-8.99, GPU wait flat at
  ~60.7 ms). Each GEMV already launches thousands of threadgroups and saturates the device,
  so there is no idle capacity for a second queue to exploit. The independent scratch buffers
  were kept (they remove a real false dependency) but bought nothing measurable.
- **Fusing the greedy LM head** was structurally right -- it drops a full Softcap pass and the
  1 MB logit readback -- but only moved 13.4 -> 13.3 ms. That GEMV is bandwidth-bound reading
  the 396 MiB tied embedding table, which fusion does not change.

### Tier 4 (partial): what was tried after Stage 4

- **Ring-buffer KV cache -- kept.** Sliding-window layers now use `sliding_window` slots via
  `physical_slot(L, p) = p % capacity`, independent of total context; only the 5 full-attention
  layers scale with it. **Context went 1024 -> 4096** (2048 on <=16 GB) for 560 MB of KV,
  versus the ~1.85 GB that raising context without the ring would have cost. Decode speed is
  unchanged, as expected -- this is a capability change, not a speed one.
- **`Load4` vectorized activations -- reverted, measured NEUTRAL.** Interleaved A/B over 4
  rounds: scalar median 7.84 tok/s, Load4 median 7.80, spread +/-0.1. DXC already merges the
  scalar loads into wide fetches, so hoisting them by hand changes nothing. A first,
  non-interleaved comparison appeared to show a 15% regression; that was machine drift, not
  the change. See the warning comment in [shaders/Common.hlsli](shaders/Common.hlsli).

**Benchmarking note.** Decode measured 8.6-9.0 tok/s earlier in the Stage 4 session and
7.1-7.9 tok/s later, with bit-identical shader code. Run-to-run drift on this machine is
larger than most single optimizations, so **compare variants interleaved in one session**, not
across sessions. The Stage 4 progression table was measured sequentially and its relative
gains hold, but treat its absolute numbers as one session's snapshot.

### Not yet done

Deferred deliberately; each would need the token-for-token gate re-run: FP16 activations
(halves activation bandwidth, but changes numerics enough to cost us that gate), lane
occupancy on routed `down_proj` (34% -- only 11 of 32 lanes busy), kernel fusion, and batched
prefill with RDNA3 WMMA. Prefill is currently ~5.7 tok/s and is the reference's weak spot too
(~28 tok/s), so it is the biggest remaining opportunity.

Context can go above 4096 only after `ATTN_MAX_SPAN` in [shaders/Attention.hlsl](shaders/Attention.hlsl)
is raised; scores are staged in groupshared, which is capped at 32 KB, so beyond ~8000 needs
an online-softmax rewrite that never materializes the full score span.

```powershell
.\build\turbo-winfare.exe --cpu --prompt "What is the capital of France?"
#   -> The capital of France is **Paris**.
```

~0.1 tok/s (roughly 14 s/token), which is expected: it is a single-threaded reference whose
job is to be right, not fast. `--dump-tensors <dir>` writes per-stage FP32 tensors for the
first token (`embed`, `layerN_hidden`, `final_norm`, `logits`) so Stage 3 can diff each
kernel against it. Other flags: `--max-tokens`, `--temperature` (0 = greedy).

### Forward-pass constants that are easy to get subtly wrong

All verified against the reference's Metal kernels and the checkpoint -- not inferred. Each
of these produces fluent-but-wrong output if missed, which is the hardest failure to debug:

- **Attention scale is 1.0**, not `1/sqrt(head_dim)`. The query scaling is absorbed into `q_norm`.
- **RMSNorm applies `w`, not `1 + w`** (eps 1e-6). The offset is baked into the checkpoint.
- **Dequant is `w = q * scale + bias`** (MLX affine, group 64), packed low-nibble-first.
- **Full-attention layers have no `v_proj`** -- V reuses the raw `k_proj` output, then takes a *no-scale* RMSNorm and skips RoPE entirely.
- **Partial rotary applies to full layers only**: 64 of 256 pairs (`0.25 * 512 / 2`). SWA layers rotate all 128.
- **Embedding is scaled by `sqrt(2816)`**; the LM head is the same matrix (tied).
- **Router**: input is `rmsnorm_no_scale(hidden) * router.scale / sqrt(D)`, 8-bit weights, softmax over the **top-8 only**, times `per_expert_scale`. Ties go to the lower expert index.
- **Three distinct pre-FFN views**: shared expert reads `pre_feedforward_layernorm`, routed experts read `pre_feedforward_layernorm_2`, the router reads the unscaled norm.
- **The shared expert is added with no routing weight.**
- **Sampling order is Top-P -- Top-K -- temperature**, not the usual HuggingFace order.
  Implemented once in [src/sampling.cpp](src/sampling.cpp) and used by both paths.
  `top_p < 1` with `temperature > 0` **requires** a `top_k` in 1..256 -- full-vocabulary
  nucleus sampling is deliberately unimplemented upstream, and the shortlist optimization
  depends on that bound. The repetition penalty must be applied to the **already-softcapped**
  logits; applied to raw logits it is very nearly a no-op, because real Gemma 4 logits reach
  the hundreds and sit deep in tanh saturation.

### The dispatch graph ([src/runner.cpp](src/runner.cpp))

`produce_token` is **two command lists per layer**, not one. The router's top-8 has to be read
on the CPU before the experts it selects can be streamed, so each layer is:

```
[norm -> QKV -> epilogue -> attention -> o_proj -> PostAttn -> router -> topK]  submit, fence
   read back top-8 indices  ->  stream those 8 expert blocks
[shared expert -> 8 routed experts -> LayerTail]                                submit, fence
```

Part 2 is split again -- the shared expert and any cache-hit experts are submitted *before*
the miss reads are issued -- so it is ~92 submissions but only **31 fence waits** per token
(one per layer, plus the head). The fence count is what costs; see the Performance section.

**Expert slots must outnumber the routed experts.** `plan_experts` pins every slot it hands
back, and `release_plan` unpins exactly those. Without pinning, a pool of 8 slots serving 8
routed experts could evict its own earlier entries, and the caller would bind one expert's
weights in place of another's -- the output stays fluent, just wrong. That bug produced
*"The capital of France is **_** (This meaning is a small enough verb form sentence..."* --
correct prefix, then drift. `tests/test_streamer.cpp` pins this down, including the case
where one plan must not release another's slots.

## The model bundle

`gemma-4-26b-a4b.gturbo/` is now **real** (13.3 GB), built by
`python tools/convert_hf_to_gturbo.py`. Verified against the Swift reference:

| | Ours | Reference |
|---|---|---|
| expert stride | 3,358,720 B | 3,358,720 B |
| `packed_experts/` total | 12,897,484,800 B | 12,897,484,800 B |
| resident payload + index | 1,353,689,148 + 81,920 | 1,353,771,068 |
| index sha256 | `bf198c9f--c850b13` | `bf198c9f--c850b13` |

`resident.bin` is ~7.5 MB larger than the reference's `model_weights.bin` because each
scales/biases sub-region is also 16 KB-aligned, for unbuffered DMA.

**`expert_stride` is 3,358,720 -- not the 692,224 that was hardcoded everywhere before.** Any
code computing an expert file offset from the old value read the wrong expert entirely.

`layout.json` describes the nine sub-tensor offsets **once** (`expertBlock`) instead of
enumerating all 3,840 (layer, expert) pairs; that is why it is 3.4 KB rather than 6.9 MB.

## Published documentation

The repository's public docs are [README.md](README.md),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/GTURBO_FORMAT.md](docs/GTURBO_FORMAT.md),
[docs/PERFORMANCE.md](docs/PERFORMANCE.md), and [CONTRIBUTING.md](CONTRIBUTING.md). They are
derived from this file -- **when you change behavior documented here, check whether one of them
needs the same change.**

The older `weights.md`, `Performance.md`, and `spec.md` working notes are untracked (see
[.gitignore](.gitignore)): they describe pre-Stage-0 behavior and embed local absolute paths.
If a local copy is present, trust this file and the code over it.

Legal and release files: [LICENSE](LICENSE) (Apache-2.0), [NOTICE](NOTICE) (attribution to the
upstream work plus the §4(b) notice of modification), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). **Never commit weights, `.gturbo` bundles, or
build output** -- upstream's terms forbid redistributing weights, and `.gitignore` is the only
thing standing between a `git add -A` and a 17 GB push.
