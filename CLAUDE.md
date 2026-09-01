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
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # test_format, test_tokenizer, test_gpu_kernels, test_convert
```

Run a single test binary directly (they are standalone, no framework):

```powershell
.\build\test_format.exe
.\build\test_gpu_kernels.exe
.\build\test_tokenizer.exe
```

CMake copies `gui/` and `shaders/` into the build directory on every build; shaders are resolved at runtime relative to the executable.

**`-DCMAKE_BUILD_TYPE=Release` at CONFIGURE time is load-bearing, and `--config Release` on the
build line is not.** `--config` only means anything to a multi-config generator (Visual Studio,
Xcode); Ninja is single-config, so it accepted the flag and ignored it. Nothing set
`CMAKE_BUILD_TYPE`, so CMake emitted `FLAGS = -std=gnu++23` with **no `-O` at all** and the
engine was built at `-O0` for the whole life of the project -- every performance figure
recorded before 2026-08-31 was measured on an unoptimized binary. Configure now defaults to
`Release` when no build type is given, and prints the effective flags:

```
-- turbo-winfare: CMAKE_BUILD_TYPE='Release'
-- turbo-winfare: CXX flags = ' -O3 -DNDEBUG'
```

If that second line ever shows no `-O`, stop and fix it before measuring anything. There is a
guard in [CMakeLists.txt](CMakeLists.txt) for the specific way this went wrong -- the cache in
`build/` had `CMAKE_CXX_FLAGS_RELEASE`, `_DEBUG`, `_RELWITHDEBINFO` and `_MINSIZEREL` all
blanked, so selecting a build type changed nothing. A *clean* cache populates them correctly,
which is what made it so hard to see: it reproduced only in the tree you already had.

`-march=native` is available as `-DTURBO_NATIVE_ARCH=ON` and is **off by default**, because
`turbo-winfare.exe` is a shipped artifact and the build host is not necessarily the Zen 4
target. `-ffast-math` is deliberately never enabled: the GPU path is gated on matching the
scalar CPU reference token-for-token, and relaxing IEEE semantics on one side of that
comparison would retire the only end-to-end correctness check the project has.

**The test suite must be built in a way that lets it fail.** `Release` defines `NDEBUG`, which
compiles `assert` out entirely -- and 293 assertions across eight of the nine test binaries are
plain `assert`. [tests/check.hpp](tests/check.hpp) redefines `assert` so it survives `NDEBUG`,
and every test `.cpp` includes it **last**, after any `<cassert>`: `<cassert>` has no include
guard and re-defines `assert` on each inclusion, so a header pulled in later would silently
undo it. Without that header, `assert(1 == 2)` in a Release build exits 0. Verified both ways.

Toolchain bootstrap: `python tools/download_toolchain.py` installs w64devkit to `C:\w64devkit` **and** fetches `dxcompiler.dll` + `dxil.dll` into `build/`. Both are required -- **`C:\w64devkit\bin` must be on `PATH`** or g++ fails with `cannot execute 'as'`, and without DXC every shader fails to compile (there is deliberately no `cs_5_0` fallback).

The `turbo-winfare` target also compiles [assets/turbo-winfare.rc](assets/turbo-winfare.rc)
with `windres`, which embeds `assets/turbo-winfare.ico` as icon resource **1** -- the shell
picks the lowest resource ID, so that number is load-bearing. The `.ico` is committed;
[tools/make_icon.ps1](tools/make_icon.ps1) regenerates it from `gui/logo.png` and only needs
running when the artwork changes. Two things it encodes that are easy to lose:

- **Every entry is an uncompressed DIB, including 256x256.** Windows reads PNG-compressed ICO
  entries fine, but many other consumers assume a `BITMAPINFOHEADER` and render a PNG payload
  as noise -- .NET Framework's own `System.Drawing.Icon` among them, which is what caught it
  here. All-DIB costs ~150 KB and is a file you can actually read back and verify.
- The artwork is a small squircle on a large dark canvas, so the script crops to the measured
  border box and cuts the corners to transparent along the *same* radius (~170 at the cropped
  scale). A mismatched radius reads as a doubled corner.

Note `gui/logo.png` is really a **JPEG** -- it starts with the JFIF magic, not the PNG
signature. Browsers sniff the content so the GUI is unaffected, but anything pointing a real
PNG decoder at it (Python's stdlib, for one) fails on a file that looks like it should work.

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
  | `--host <addr>` | Bind address; `127.0.0.1` (default) or `0.0.0.0` |

`run_gui.py` (a Python `http.server` bridge over the C ABI) **has been removed.** It registered
no `/v1/*` routes and no `GET /api/config`, and `gui/app.js` requires both -- so generation under
it was not merely degraded, it never ran at all, and it silently served the sidebar's
pre-hydration placeholder values. `libturbo_engine.dll` and the `extern "C"` ABI stay; only the
bridge is gone.

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

**The GUI exposes every runtime engine setting**, and nothing it shows is a literal. Panels:
Model Repository, Get a Model, Memory & DRAM Cache, Generation Parameters, Sampling &
Determinism (repetition penalty / seed / stop sequences), Diagnostics (the per-token phase
breakdown plus `uma_upload_fallbacks`), and a read-only Server panel for the launch-fixed
values. Three classes of bug were removed here and each is worth not reintroducing:

- **The stop button was unreachable.** `sendPrompt()` returned on an empty textarea *before*
  checking `isGenerating`, and sending clears the textarea -- so during a generation the box
  was always empty and the button did nothing. The `isGenerating` branch must stay first.
- **The model id was hardcoded** as `gemma-4-26b-a4b-it` in the `/v1` payload, so `--model-id`
  made every message 404 while the HUD still read READY. It now comes from `GET /v1/models`.
- **`meta-arch` / `meta-topk` / `meta-quant` were literals no code ever wrote.** `/api/models`
  now reports `layers`, `experts`, `top_k`, `expert_stride`, `model_id` and `quantization` for
  the active bundle, and the GUI renders those. The slot-size estimate uses the reported
  stride rather than a constant in JavaScript.

`POST /api/config` validates into a copy and commits once. It used to write each field into
`config_` as it parsed, then reject -- so a refused `top_k: 999` was *stored*, and the next
request (which does not resend top_k) was refused for a value the user never set.

The GUI loads no external resources. It used to pull Inter and JetBrains Mono from
`fonts.googleapis.com`, which blocks first paint on a DNS timeout -- on an offline machine,
which is the whole point of a local inference engine.

### Model acquisition (`/api/download`)

`ModelFetcher` ([src/model_fetch.cpp](src/model_fetch.cpp)) runs
`tools/convert_hf_to_gturbo.py --progress-json` as a child process and parses one JSON object
per line. Building a bundle is a ~14.6 GB stream-and-repack with a 16 KB-aligned layout and a
manifest of sha256s; that logic exists, is tested, and is the only thing that has ever
produced a loadable bundle. A C++ reimplementation would be a second copy that must stay
bit-identical forever, plus a TLS stack the engine does not otherwise link.

Three things there are load-bearing:

- **`HF_TOKEN` goes in the child's environment, never on the command line.** A command line is
  readable by every process on the machine.
- **`PYTHONUNBUFFERED=1`**, or Python block-buffers the pipe and the GUI sits at 0% for
  minutes before the first 8 KB flushes.
- **The output name is an allowlist** (`[A-Za-z0-9._-]+\.gturbo`, no `..`): it arrives over
  HTTP and becomes a directory the server creates.

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

  **A lost device does not stop this engine on its own, so it is checked for explicitly.**
  Every HRESULT on the submit path is now inspected -- `Close()`, `Signal()`,
  `SetEventOnCompletion()` and both `Reset()` calls -- and a failure throws naming the call
  site plus `GetDeviceRemovedReason()`. All five used to be discarded, and the resulting
  failure mode is far worse than a crash: on a removed device `GetCompletedValue()` returns
  `UINT64_MAX`, so every fence wait returns instantly; the router-index readback still maps
  and hands back *stale* bytes, so the same eight experts look like cache hits in every layer;
  and the greedy token readback returns whatever was there before. The engine keeps running
  and reports a large tokens/sec attached to garbage output. A sibling engine chased exactly
  that as a performance result before noticing the output was blank.

  `wait_for_fence` also waits with a 60 s bound rather than `INFINITE` -- a hung device used
  to hang the process with it -- and `ForwardRunner::initialize()` probes `device_ok()` once
  after committing the 1.29 GB of resident weights, so an over-greedy allocation fails at load
  with a clear message instead of surfacing far downstream.
- `ComputePipelineManager` ([src/pipeline.cpp](src/pipeline.cpp)) -- loads/compiles the HLSL kernels in [shaders/](shaders/) and dispatches them.
- `ExpertStreamer` ([src/streamer.cpp](src/streamer.cpp)) -- per-layer NVMe->UMA expert loading with an LFU/LRU DRAM cache; opened lazily per layer.

  **`ExpertPlan` is move-only and releases its own pins.** The explicit `release_plan` call in
  the decode loop sits ~115 lines after `plan_experts`, with the whole routed-expert encode in
  between; anything throwing in that window used to leak the plan's pins permanently, and the
  *next* token then died with "no evictable slot" -- a recoverable error becoming a dead runner
  one token later.

  **`fetch_misses` drains the reads it issued before unwinding.** It used to clear
  `read_pending` on every miss without waiting, so a failure partway through left the kernel
  still writing into slots the streamer believed were idle; the destructor then skipped them
  and the next `issue_read` would `ResetEvent` and overwrite an `OVERLAPPED` mid-transfer, into
  a UMA page the GPU is also reading. `issue_read` now also rejects a slot whose read has not
  retired, which is the runtime detector for that state -- there is no deterministic unit test
  for it, because a 3.3 MB read served from the page cache completes inline and nothing is ever
  pending by the time an error is raised.
- `KVCacheManager` ([src/kv_cache.cpp](src/kv_cache.cpp)) -- owns the per-layer FP32 K/V buffers and the ring indexing (`physical_slot`) for **both** paths. `ForwardRunner` holds no KV buffers of its own.
- `Tokenizer` ([src/tokenizer.cpp](src/tokenizer.cpp)) -- HF `tokenizer.json` BPE with byte fallback, Gemma 4 `<|turn>`/`<turn|>` markers (NOT Gemma 2/3 `<start_of_turn>`). A default-constructed `Tokenizer` holds no vocabulary and throws on `encode`.
- Format layer: `GTurboManifestV1` ([src/manifest.cpp](src/manifest.cpp)), `PackedExpertsLayoutV1` ([src/packed_experts.cpp](src/packed_experts.cpp)), `ResidentIndexCodec` ([src/resident_index.cpp](src/resident_index.cpp)), constants in [include/gturbo/format.hpp](include/gturbo/format.hpp).
- `HTTPServer` ([src/server.cpp](src/server.cpp)) and the `extern "C"` ABI in [src/c_api.cpp](src/c_api.cpp) (header [include/gturbo/c_api.h](include/gturbo/c_api.h)) -- the stable embedding boundary exported from `libturbo_engine.dll`.
- `ModelFetcher` ([src/model_fetch.cpp](src/model_fetch.cpp)) -- runs `tools/convert_hf_to_gturbo.py` as a child process so a bundle can be built from the GUI, and probes the host for Python, the converter script and free disk.

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

**Byte offsets are checked for BIT-IDENTICAL agreement, not tolerance.** Every original GEMV
case passed `0` for `w_off`/`s_off`/`b_off`/`x_off`/`out_off`/`row_base` and bound three
*separate* buffers -- which the engine never does. Production binds one `resident.bin` buffer as
`{RES, RES, RES, x}` at three 16 KB-aligned offsets, and chunks the 262,144-row LM head with
`row_base > 0`. So the offset arithmetic -- the code that decides *which expert's weights a
routed GEMV actually reads* -- had no coverage at all. It does now, and the check is equality
against the same matrix run at offset zero: relocating a tensor cannot change the arithmetic, so
there is no tolerance to argue about, and an offset wrong by less than a rounding error is still
caught. Verified by breaking the kernel three ways: zeroing `w_off` fails 200/200 rows, zeroing
`out_off` 200/200, zeroing `row_base` exactly the 72 rows of the second chunk -- and in all
three the *old* zero-offset cases stayed green.

Do not add a second tolerance-based CPU comparison on a fresh random matrix: one was tried and
failed on row 135 of 200 (0.0109138 against 0.0109309), a 1.7e-5 disagreement in a 2,816-term
FP32 sum whose terms are of order 1. That is cancellation, not a bug, and it is why the offset
check is an equality check.

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
- **Stage 4 (done)** -- performance. **2.34 -> 8.8 tok/s decode** at the time (all of it on an `-O0` build, unknowingly), beating the Swift/Metal reference's 5.1-6.3 tok/s on an M2 Air. With the build fixed and re-measured 2026-08-31: **9.9 tok/s at 24 slots, 16.2 at 44**. See Performance.

```powershell
.\build\turbo-winfare.exe --prompt "What is the capital of France?"
#   -> The capital of France is **Paris**.   (~9.9 tok/s decode at 24 slots)
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
| GUI exposure of stop sequences / `seed` / `repetition_penalty` | **done** -- Sampling & Determinism panel; `ServerConfig` stores all three |
| GUI exposure of per-response usage, TTFT and true stop reason | **done** -- `stream_options.include_usage` plus the `x_turbo` chunk extension |
| In-app model download | **done** -- `/api/download*` drives the converter; see `ModelFetcher` |
| KV reuse across turns (prompt cache) -- every request re-prefills from scratch | not started |
| OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`, `/health`, SSE) | **done** -- [src/openai_api.cpp](src/openai_api.cpp) |
| Real HTTP framing (Content-Length, chunked, 1 MiB limit, 413/415) | **done** -- [src/http.cpp](src/http.cpp) |
| Tool calling and channel decoding -- note the generation prompt already opens a `thought` channel and **nothing suppresses it** | not started |
| Runtime integrity verification, symlink rejection | not started |
| Expert-cache opt-in above 44 slots (descriptor heap derived from slot count) | **done** |
| GUI/HTTP reload actually applying slots/context | **done** -- `/api/load_model` used to discard them |
| Chunked/batched prefill (~7.6 tok/s here vs ~28 in the reference) | deferred |
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

**Every row above was measured on an `-O0` build** (see Build & test). They are kept because
their *relative* gains are what the table is for and those still hold -- each was an A/B against
the row above it on the same binary. Do not compare them against anything measured after
2026-08-31.

### The `-O0` -> `-O3` measurement (2026-08-31)

Interleaved A/B/A/B, 3 rounds, idle machine, disk queue confirmed 0, page cache warm. Same
source, two build directories differing only in `CMAKE_CXX_FLAGS_RELEASE`. Medians:

| | `-O0` | `-O3` | change |
|---|---|---|---|
| CPU reference (`--cpu`, s/forward pass) | 8.71 | **2.31** | **3.8x faster** |
| Decode, `--slots 24` | 9.01 tok/s | **9.48** | +5.2% |
| Decode, `--slots 44` | 14.96 tok/s | **16.09** | +7.6% |
| Prefill, `--slots 44` | 7.17 tok/s | **7.58** | +5.7% |
| CPU-other bucket, `--slots 44` | 7.98 ms | **3.86** | -52% |

The shape is what the phase breakdown predicts and is worth internalizing before optimizing
anything else here: decode is ~41% expert I/O and ~50% GPU wait, so **compiler optimization
buys single digits on the GPU path and nothing structural**. It is the `--cpu` reference --
pure scalar C++ with no I/O and no GPU -- that moves 3.8x. `--cpu` is now ~0.43 tok/s
(2.3 s/token), not the ~0.1 tok/s previously documented.

**Two workload traps caught while measuring this**, both of which produced wrong numbers first:

1. **`--max-tokens 120` does not give you 120 tokens.** `"What is the capital of France?"`
   stops at end-of-turn after 8, so the budget never binds and the run is an 8-token workload
   wearing a 120-token label. The giveaway was cache hit rates identical to the 24-token runs.
   Use an open-ended prompt and **check `over N forward passes`** in the output: 144 passes is
   a real 120-token decode, 28 is not.
2. **Absolute decode figures below are not comparable to the Stage 4 table above**, and not
   only because of `-O0`. This machine now auto-sizes to 32 slots, which means it reports
   >= 30 GB installed where the Stage 4 box was 24 GB. More RAM changes the page-cache
   competition that the whole `--slots` argument rests on -- see below.

### Trading RAM for speed (`--slots`) -- opt-in, defaults unchanged

The auto-size ladder (16/24/32 slots at <=16/<=24/>=32 GB) is deliberately conservative and
stays that way; raising the cache is an explicit opt-in via `--slots N` or the GUI sidebar
followed by a model reload. Interleaved, 120 tokens, idle machine:

Original measurement, 24 GB box, `-O0` build:

| slots | pool | peak RSS | cache hit | decode |
|---|---|---|---|---|
| 24 (auto there) | 2.3 GB | 4.3 GB | 64.8% | **7.0** |
| 44 | 4.1 GB | 6.2 GB | 82.7% | **9.2** |
| 64 | 6.0 GB | ~8.1 GB | 90.3% | 7.8-8.9 |

Re-measured 2026-08-31 on `-O3`, >= 30 GB installed, interleaved (not swept in order), 3
rounds, 120 real decode tokens confirmed by `over 144 forward passes`:

| slots | pool | cache hit | decode (median of 3) |
|---|---|---|---|
| 16 | 1.5 GB | 56.8% | 7.54 |
| 24 | 2.3 GB | 69.5% | 9.87 |
| 32 (auto here) | 3.1 GB | 72.2% | ~14 |
| 44 | 4.1 GB | 86.3% | **16.20** |
| 64 | 6.0 GB | 92.0% | **17.38** |

Hit rates line up with the original within a couple of points, so the workload is comparable.
Peak RSS was not re-measured -- nothing in this work changed an allocation size.

**The "64 slots is not faster" finding no longer holds on this machine, and that is a
statement about the machine, not about the code.** 64 was slower than 44 on the 24 GB box
because a 6 GB pool squeezed the OS page cache the streamer deliberately leans on. With
>= 30 GB there is room for both, and 64 is now the fastest configuration measured. The
underlying claim -- that the pool competes with the page cache and more is not automatically
better -- is unchanged; where the crossover sits is a property of installed RAM. **The
auto-size ladder still stops at 32 and should stay there:** it has to be safe on the 16 GB
Legion Go S, where the original finding still applies exactly as written.

Footprint is `1.29 GB resident + KV + slots * 30 * 3.2 MB`.

Two ceilings bound it, both now reported at startup rather than later:

- **Descriptor heap.** Derived from the slot count by
  `ComputePipelineManager::descriptors_for()`. It was hardcoded at 65,536, which fit exactly 44
  slots and then threw `Descriptor heap exhausted` **mid-generation**, because tables are
  created lazily as slots are touched. `initialize_pipelines()` takes the capacity as a
  required argument so the resolve-slots-first ordering cannot be reversed silently.
- **The adapter's shared-memory budget** (~12.2 GB here, roughly half of installed RAM) is the
  real cap on UMA allocation, not installed RAM. Overshooting it does not degrade, it fails --
  and it used to surface as an unrelated-looking `Failed to map ...` far downstream.

**Where the time goes now:** ~41% expert I/O, ~50% GPU wait, ~9% other CPU, with the
LM head accounting for ~10% that *spans* the GPU-wait and CPU buckets rather than
forming a fourth one. The three buckets are disjoint and sum to 100%; the LM head is a
single `submit_and_wait`, so its fence is inside "GPU wait" and its recording inside
"CPU other". Counting it as a peer double-counted the fence and drove the residual
negative -- the CLI printed `CPU other: -2.16 ms (-1.93%)` once the optimized build shrank
real CPU work below the size of the overlap. The
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

**Two measurement traps that have each already produced a confident wrong conclusion.**

1. **Background disk I/O invalidates everything.** An audit once reported a ~40% regression
   against this file. There was none -- a game download was running. Same binary, same flags,
   bit-identical workload (38,985 MB in 12,171 reads, 64.78% hit rate both times): 5.07 tok/s
   contended vs **6.03** idle, I/O throughput 1957 vs **3306 MB/s**. Decode is I/O-bound, so
   anything touching the NVMe competes directly and also evicts the expert pages the streamer
   relies on the OS page cache to hold. Check `Get-Counter
   '\PhysicalDisk(_Total)\Current Disk Queue Length'` reads 0 before believing a number.
2. **A sequential sweep measures page-cache warmth, not your variable.** Sweeping
   `--slots 16,24,32,40` in order shows everything getting faster as each run inherits a warmer
   cache. That artefact turned a real +27% into an apparent +50%. Alternate A/B/A/B for >=3
   rounds and compare medians: interleaved spread within a config is +/-0.05 tok/s, sequential
   is over 1 tok/s.

Also **check the exit code** -- a failed run exits non-zero with a diagnostic, but a grep for
metric lines renders it as an empty row that reads like a slow result rather than a crash.

**Benchmarking note.** Decode measured 8.6-9.0 tok/s earlier in the Stage 4 session and
7.1-7.9 tok/s later, with bit-identical shader code. Run-to-run drift on this machine is
larger than most single optimizations, so **compare variants interleaved in one session**, not
across sessions. The Stage 4 progression table was measured sequentially and its relative
gains hold, but treat its absolute numbers as one session's snapshot.

### Open: an intermittent crash under repeated back-to-back runs

Twice during one benchmarking loop (out of ~40 runs that session, and also seen before any of
the recent changes) a CLI run exited **139** with no metrics. Both failures immediately
followed a run at a *different* `--slots` value. It has not reproduced in 23 consecutive runs
since, including deliberate stress alternating 64 and 24 slots, so **it is not understood and
not fixed** -- do not assume a clean benchmark loop means it is gone. If you hit it, capture
the full stdout+stderr and the exit code rather than a grep of the metric lines.

Suspicion, unconfirmed: allocation pressure when the previous process's multi-GB commitment
has not been reclaimed yet. Note expert slot buffers are SRV-only and so may fall back to an
UPLOAD heap (see `create_uma_buffer`), which lets the pool over-commit and fail elsewhere.

**2026-08-31 -- still not reproduced, but the instrumentation to catch it now exists.** 14
consecutive `-O3` runs alternating `--slots 24` and `--slots 64` at 120 tokens: all exit 0,
**zero UPLOAD-heap fallbacks**. That does not clear the suspicion, it just means the pool never
came close to the budget on this machine (>= 30 GB installed, 16 GB adapter shared-memory
budget). Three things changed that make a future occurrence diagnosable rather than mysterious:

- The UPLOAD fallback is no longer silent. It warns on the first occurrence naming the buffer
  and size, counts every one, and reports the count as `memory.uma_upload_fallbacks` in
  `/api/telemetry`. **If that number is not 0, believe it before any other theory.**
- Every discarded HRESULT on the submit path is checked, and failures name
  `GetDeviceRemovedReason()`. A lost device used to keep the engine running -- see the
  D3D12Context notes below.
- `fetch_misses` had a genuine memory-corruption path on its failure branch (in-flight reads
  abandoned without draining). That is fixed, and while it needs a *failed read* to trigger --
  which none of these runs had -- it is exactly the shape of latent bug that produces an
  unexplained 139.

### Not yet done

Deferred deliberately; each would need the token-for-token gate re-run: FP16 activations
(halves activation bandwidth, but changes numerics enough to cost us that gate), lane
occupancy on routed `down_proj` (34% -- only 11 of 32 lanes busy), kernel fusion, and batched
prefill with RDNA3 WMMA. Prefill is currently ~7.6 tok/s and is the reference's weak spot too
(~28 tok/s), so it is the biggest remaining opportunity.

Context can go above 4096 only after `ATTN_MAX_SPAN` in [shaders/Attention.hlsl](shaders/Attention.hlsl)
is raised; scores are staged in groupshared, which is capped at 32 KB, so beyond ~8000 needs
an online-softmax rewrite that never materializes the full score span.

```powershell
.\build\turbo-winfare.exe --cpu --prompt "What is the capital of France?"
#   -> The capital of France is **Paris**.
```

~0.43 tok/s (2.3 s/token; it was ~0.1 tok/s before the build was fixed), which is expected: it is a single-threaded reference whose
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

## Bundle path resolution -- the stale placeholder in `build/`

`build/gemma-4-26b-a4b.gturbo` is a 7.2 MB **placeholder** left by the retired repacker. It
has a `manifest.json` and a `packed_experts/layout.json`, so nothing short of parsing rejects
it -- its `layout.json` predates the `expertBlock` format and reports
`expertStride: 88604672`.

Because of it, **a bare bundle name must never be resolved by an existence check.** Use
`resolve_bundle_path()` ([src/manifest.cpp](src/manifest.cpp)), which validates each
candidate with `bundle_loads()` and searches the working directory, then the executable's
directory, then one level up. An explicit path (absolute, or containing a separator) is taken
literally.

This bit once already: `/api/load_model` took the GUI's raw `"gemma-4-26b-a4b.gturbo"` and
resolved it against the working directory. Launched from `build/` -- which is what
double-clicking does -- startup loaded the real bundle (it resolved properly) while a reload
loaded the placeholder and failed with `layout: missing required field 'expertBlock'`.
`/api/models` had the same flaw and offered the placeholder in the dropdown.

Two consequences worth keeping:

- `/api/models` lists only bundles that actually parse, across every search root, and
  compares `is_active` on **canonicalized** paths -- the runner stores an absolute path while
  the list emits a bare name, so a string `==` never matched and left the architecture fields
  null.
- `swap_runner` parses and cross-validates the new bundle **before** releasing the old
  runner. The release-first ordering exists so two models are never committed at once
  (2 x 12.9 GB at 128 slots), but without the up-front validation a mistyped path destroyed a
  perfectly good loaded model.

## The server is loopback-only, and why

`HTTPServer` binds `127.0.0.1` unless `--host 0.0.0.0` is passed. It has no authentication, sends
`Access-Control-Allow-Origin: *`, and exposes endpoints that load models and start multi-GB
downloads, so the default had to be the safe one.

It previously bound `INADDR_ANY` unconditionally, and the static file handler concatenated the
request target straight into `fs::path("gui" + path)` with nothing inspecting it. That combination
was a verified arbitrary file read from any machine that could reach the port:

```
GET /../../../../../Windows/win.ini   ->  200, file contents
```

`request_path_is_safe()` ([src/server.cpp](src/server.cpp)) now rejects any target whose decoded
form contains a `..` component, a backslash, a colon or a NUL, **before** dispatch -- so the guard
covers the `/api` and `/v1` routes too, not just static files. It percent-decodes first, because
`%2e%2e%2f` reaches the same place without a literal `..`. `tests/test_server.cpp` pins both forms.

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
