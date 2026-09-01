<p align="center">
  <img src="docs/assets/logo.png" alt="Turbo-WinFare Logo" width="160" height="160" style="border-radius: 32px;" />
</p>

<h1 align="center">Turbo-WinFare</h1>

<p align="center">
  <b>Native DirectX 12 MoE Neural Inference Engine for Windows Handhelds & iGPUs</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/DirectX-12%20(DirectCompute)-3b82f6?style=flat-square" alt="DirectX 12" />
  <img src="https://img.shields.io/badge/Platform-Windows%2011-0078D6?style=flat-square" alt="Windows 11" />
  <img src="https://img.shields.io/badge/Model-Gemma%204%2026B--A4B-8b5cf6?style=flat-square" alt="Gemma 4" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-10b981?style=flat-square" alt="License" />
</p>

> **Turbo-WinFare is a Windows port derived from
> [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), copyright Andrey Mikhaylov,
> licensed under Apache-2.0.**
>
> It is an independent project. It is **not** affiliated with, sponsored by, or endorsed by
> Andrey Mikhaylov or the TurboFieldfare project, and it is **not** affiliated with, sponsored
> by, or endorsed by Google. Gemma is a trademark of Google LLC.
>
> **Please report Windows, Direct3D, and build problems here — never upstream.**

Run **Gemma 4 26B-A4B** on a Windows handheld or laptop iGPU, by streaming its experts off NVMe
instead of holding them in memory.

Turbo-WinFare is a native Windows 11 inference engine written in C++23 on Direct3D 12
(DirectCompute). Gemma 4 26B-A4B is a mixture-of-experts model: 30 layers, 128 routed experts
plus one shared expert per layer, of which only ~3.88B parameters are active per token. Only the
1.35 GB of non-expert weights stay resident; the expert blocks for the eight experts each layer
actually routes to are read from disk per token, straight into host-coherent unified memory
(`D3D12_HEAP_TYPE_CUSTOM` + `WRITE_BACK`), so the compute shaders read them with no staging copy.

Developed and tested on a Lenovo Legion Go S (Ryzen Z1 Extreme, Radeon 780M, LPDDR5X unified
memory).

**Status:** the forward pass is complete and verified — all 15 HLSL kernels agree with a scalar
FP32 CPU reference, and GPU greedy decoding matches CPU greedy decoding token for token.
Sampling, stop sequences, streaming output, a browser GUI, and a local OpenAI-compatible server
are done. See [Parity with the reference](#parity-with-the-reference) for what is not.

The test suite is ten binaries run by `ctest`; only `test_gpu_kernels` needs a Direct3D 12
device, and none of them need the 13.3 GB model bundle.

---

## Sibling project — Turbo-WinFare Dense

[**Turbo-WinFare Dense**](https://github.com/justin-creator222/Turbo-WinFare-Dense) applies the
same idea to a *dense* model instead of a mixture of experts. It is a **Vulkan 1.3** engine that
runs **Gemma 4 31B Dense** on the same class of hardware (Legion Go S, Radeon 780M). Where this
repository streams the eight experts each layer routes to, Dense keeps 45 of the model's 60
layers resident and reads the other 15 from NVMe on every token — about 4 GB per token — and it
uses Gemma 4 E2B as a speculative draft model. Measured at **1.08–1.13 tok/s** greedy decode at
4096 context; E2B on its own is fully resident and runs at ~15 tok/s.

Separate codebase, separate container format (`.g4dense`), separate GUI and OpenAI-compatible
server. The two projects share the design brief — an APU whose driver will not hand out as much
memory as the model needs — not code.

## Requirements

| | |
|---|---|
| OS | Windows 11 |
| GPU | Direct3D 12, **Shader Model 6.6** with wave intrinsics. Only AMD RDNA 3 (Radeon 780M) has been tested. |
| RAM | 16 GB minimum, 24 GB+ recommended. Context and the expert-slot pool auto-size from installed RAM: under 22 GB, 2048 tokens and 16 slots per layer; from 22 GB, 4096 tokens and 24 slots; from 30 GB, 32 slots. `--slots` raises the pool explicitly — see [Trading RAM for speed](#trading-ram-for-speed). |
| Disk | ~14 GB free on an NVMe SSD for the model bundle, plus ~15 GB transient during conversion. |
| Toolchain | CMake ≥ 3.20, Ninja, MinGW-w64 g++ (w64devkit). **MSVC is not supported.** |
| Python | 3.10+ (standard library only) for the toolchain and model-conversion scripts. |

Expert streaming is disk-latency sensitive. A SATA SSD or a spinning disk will work but will be
much slower than the numbers below, which were all measured on NVMe.

## Build

```powershell
python tools/download_toolchain.py     # w64devkit -> C:\w64devkit, DXC -> build/
```

Then put the toolchain on `PATH` **before configuring** — this is not optional:

```powershell
$env:PATH = "C:\w64devkit\bin;" + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Two failure modes account for nearly every build problem:

* `cannot execute 'as'` — `C:\w64devkit\bin` is not on `PATH`. g++ cannot find its assembler.
* Every shader fails to compile — `dxcompiler.dll` / `dxil.dll` are missing from `build/`. The
  engine compiles HLSL at runtime as `cs_6_6` and has **no `cs_5_0` fallback by design**: the
  old fallback `#define`d the wave intrinsics to identity, which silently reduced every
  cross-lane reduction to one lane's partial value. Re-run `python tools/download_toolchain.py --dxc`.

`shaders/` and `gui/` are copied into `build/` on every *build* (not at configure time), so
editing a `.hlsl` takes effect on the next `cmake --build`.

## Get the model

Weights are **not** distributed with this repository.

```powershell
python tools/convert_hf_to_gturbo.py --output gemma-4-26b-a4b.gturbo
```

This downloads the pinned revision `0d77464e` of
[`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)
and repacks it locally into a 13.3 GB `.gturbo` bundle. It streams — peak memory stays at a few
MB regardless of the 15 GB input — and it does **not** requantize: the checkpoint is already MLX
affine 4-bit (group 64, BF16 scale + bias; routers 8-bit) and every quantized value is copied
through byte for byte.

The weights are governed by Google's Gemma terms, not by this repository's license. Do not
redistribute them or a converted bundle. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Run

```powershell
# One-shot CLI generation
.\build\turbo-winfare.exe --prompt "What is the capital of France?"
#   -> The capital of France is **Paris**.

# Browser GUI on http://127.0.0.1:8080
.\build\turbo-winfare.exe

# Headless OpenAI-compatible server
.\build\turbo-winfare.exe --serve --port 8099
```

Unknown arguments are a hard error (exit 2), never silently ignored, and every numeric flag is
range-checked at parse time — `--context abc` and `--slots -1` are usage errors, not a crash or a
silent fallback to the default.

| Flag | Effect |
|---|---|
| `--model <dir>` | `.gturbo` bundle directory (default `gemma-4-26b-a4b.gturbo`) |
| `--prompt <text>` | Run one CLI generation instead of starting the GUI |
| `--gui` | Start the GUI (default when `--prompt` is absent) |
| `--cpu` | Scalar FP32 reference path; requires `--prompt` |
| `--max-tokens <n>` | Completion token budget (default 32). Alias: `--max-new` |
| `--temperature <f>` | 0 = greedy (default) |
| `--top-p <f>` | Nucleus mass in (0,1]; requires `--top-k` when < 1 |
| `--top-k <n>` | 1..256, or 0 to disable |
| `--repetition-penalty <f>` | > 0; 1 disables |
| `--seed <n>` | Makes sampled output reproducible |
| `--stop <text>` | Stop string; repeatable |
| `--quiet` | Suppress the performance footer |
| `--context <n>` | Max context in tokens, 1..4096; 0 auto-sizes from RAM. Alias: `--max-context` |
| `--slots <n>` | Expert cache slots per layer; 0 auto-sizes from RAM. Alias: `--expert-cache-slots` |
| `--dump-tensors <dir>` | Per-stage FP32 tensors for the first token (`--cpu` only) |
| `--port <n>` | HTTP port (default 8080) |
| `--host <addr>` | Bind address. `127.0.0.1` (default) or `0.0.0.0` to accept connections from other machines |
| `--serve` | Start the server without opening a browser |
| `--model-id <name>` | Model name the OpenAI endpoints expect |
| `--queue-limit <n>` | Requests allowed to wait for the engine (default 4) |
| `--no-open` | Do not launch a browser with the GUI |
| `--help` | Show usage |

The three aliases exist so a command line written for the Swift reference runs here unchanged.
`--context` is capped at 4096 by `ATTN_MAX_SPAN` in `shaders/Attention.hlsl`; asking for more is
rejected at load rather than silently truncated. `--slots` has no fixed ceiling any more — see
[Trading RAM for speed](#trading-ram-for-speed).

### Which bundle gets loaded

A **bare** bundle name (`gemma-4-26b-a4b.gturbo`, the default) is searched for in the working
directory, then beside the executable, then one level up, and each candidate is validated by
actually parsing its `manifest.json` and `packed_experts/layout.json`. A path that is absolute or
contains a separator is taken literally and never second-guessed.

The parse-don't-probe rule matters because a 7.2 MB placeholder bundle from the retired repacker
may still be sitting in `build/`. It has both files, so an existence check selects it happily —
and since double-clicking the exe makes `build\` the working directory, that is exactly when it
would win. Startup prints the absolute path it resolved to.

### Browser GUI

Launching without `--prompt` serves the GUI at `http://127.0.0.1:8080` and opens a browser
(`--no-open` suppresses that). The sidebar seeds every control from `GET /api/config`, so it shows
the configuration the engine actually resolved rather than a value baked into the HTML — including
the distinction between what is *live* and what has been requested but needs a reload.

Context and slot changes take effect on the next **Load Model**, and that reload genuinely applies
them; the model dropdown lists only bundles that parse. A reload or unload while a generation is
in flight answers `409` instead of tearing the engine down underneath it, and all generation — GUI
and OpenAI alike — is serialized through one lock, since the runner has a single KV cache and one
set of GPU scratch buffers.

Every runtime setting the engine supports has a control:

| Panel | What it exposes |
|---|---|
| Model Repository | Bundle list, load/unload, and the loaded bundle's real architecture, routed top-K and quantization (read from its manifest, not hardcoded) |
| Get a Model | Downloads and repacks the pinned checkpoint without leaving the browser — see below |
| Memory & DRAM Cache | Context, expert slots, eviction policy, cache flush |
| Generation Parameters | Temperature, top-P, top-K, max tokens |
| Sampling & Determinism | Repetition penalty, seed, and up to four stop sequences |
| Diagnostics | The per-token phase breakdown (expert I/O, GPU wait, LM head, CPU), prefill rate, I/O calls, and a warning if `uma_upload_fallbacks` is ever non-zero |
| Server | The launch-fixed values — version, bind address, port, OpenAI model id, queue limit, context ceiling |

Each response carries a footer with its prompt/completion token counts, time-to-first-token,
decode rate and stop reason. A rejected setting is reported and the control snaps back to the
value the engine actually holds; nothing in the sidebar displays a number the engine has not
confirmed.

The page loads no external resources, so it works with no network at all.

### Getting a model from the GUI

The **Get a Model** panel runs `tools/convert_hf_to_gturbo.py` server-side and streams its
progress back, so a fresh install needs no command line. It checks Python, the converter script
and free disk up front and says which one is missing rather than failing later. A HuggingFace
token is only needed for a gated repository; it is passed to the converter through its
environment, never on a command line, and is not stored or logged. **Cancel** keeps the
`.partial` directory so **Resume** continues rather than restarting the ~14.6 GB transfer.

Downloads are refused while a generation is running: decode is I/O-bound and the two contend for
the same NVMe.

There is one front-end. A second one, a Python `http.server` bridge (`run_gui.py`) that loaded
`libturbo_engine.dll` through `ctypes`, has been removed: it registered no `/v1/*` routes and no
`GET /api/config`, both of which the browser GUI requires, so chat under it was not degraded but
entirely non-functional. `libturbo_engine.dll` and the `extern "C"` ABI in
[include/gturbo/c_api.h](include/gturbo/c_api.h) remain — they are a supported embedding
boundary, independent of that bridge.

### OpenAI-compatible API

`GET /health`, `GET /v1/models`, and `POST /v1/chat/completions` (JSON or SSE) are implemented,
matching the subset the reference supports.

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="unused")
c.chat.completions.create(model="gemma-4-26b-a4b-it",
                          messages=[{"role": "user", "content": "Hi"}], stream=True)
```

Unsupported parameters are **rejected with a 400, not ignored** — `n != 1`, `logprobs`,
presence/frequency penalties, `tool_choice: required`, and image content parts. Silently
accepting one is worse than an error, because the caller believes it applied.

Generation is serialized: one request at a time, `--queue-limit` waiters in strict FIFO, then
`429 queue_full`. GUI and OpenAI requests share the same lock rather than only serializing among
themselves — the runner has one KV cache and one set of GPU scratch buffers, so two concurrent
generations would interleave inside the forward pass and corrupt each other's activations,
producing output that stays fluent and is simply wrong.

## Performance

Measured on a >= 30 GB Ryzen/Radeon desktop box, `-O3`, 120 decode tokens, machine idle.

| | Turbo-WinFare | TurboFieldfare (reference) |
|---|---|---|
| Decode | **9.9 tok/s** at `--slots 24`, **16.2 tok/s** at `--slots 44` | 5.1–6.3 tok/s on an 8 GB M2 Air |
| Decode, constrained | **7.5 tok/s** at 16 slots (the 16 GB Legion Go S target) | — |
| Prefill | ~7.6 tok/s | ~28 tok/s |

> **Figures published before 2026-08-31 were measured on an unoptimized build.** `CMakeLists.txt`
> set no `CMAKE_BUILD_TYPE`, and the `--config Release` in the documented build command is
> ignored by Ninja, so the engine compiled at `-O0`. Building it properly is worth **+5–8% on
> decode** and **3.8x on the `--cpu` reference**; the small decode gain is expected, because
> decode is ~41% disk I/O and ~50% GPU wait. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

Where decode time goes: ~41% expert I/O, ~50% GPU wait, ~9% other CPU — three disjoint buckets
summing to 100%, with the LM head a further ~10% that *spans* the GPU-wait and CPU buckets rather
than forming a fourth. The engine is nowhere near compute- or bandwidth-bound, so the remaining
headroom is in kernel efficiency and batched prefill, not in hardware. [docs/PERFORMANCE.md](docs/PERFORMANCE.md) has the full
optimization history, including the changes that were measured and *rejected*.

`--cpu` runs a single-threaded scalar FP32 reference at ~0.43 tok/s (2.3 s/token). Its job is to be right, not
fast; it is the ground truth the GPU kernels are verified against, and `--dump-tensors` writes
its per-stage activations for diffing.

### Trading RAM for speed

Defaults stay conservative. Raising the expert cache is an explicit opt-in — `--slots N`, or the
sidebar control followed by a reload. Interleaved, 120 tokens, machine idle:

| slots per layer | pool | cache hit | decode (median of 3) |
|---|---|---|---|
| 16 | 1.5 GB | 56.8% | 7.54 tok/s |
| 24 | 2.3 GB | 69.5% | 9.87 tok/s |
| 32 *(auto-sized here)* | 3.1 GB | 72.2% | ~14 tok/s |
| 44 | 4.1 GB | 86.3% | **16.20 tok/s** |
| 64 | 6.0 GB | 92.0% | **17.38 tok/s** |

More than doubling the pool roughly doubles throughput here. **This is machine-dependent, and it
used to come out the other way:** on the original 24 GB box, 64 slots was *slower* than 44 despite
a 90% hit rate, because a 6 GB pool squeezed the OS page cache the streamer deliberately relies
on. With >= 30 GB there is room for both. The principle is unchanged — the pool and the page cache
compete for the same physical memory, and more is not automatically better — but where the
crossover falls depends on installed RAM, so measure on your own machine rather than copying a
number from this table. **The auto-size ladder still stops at 32 on purpose:** it has to be safe
on the 16 GB Legion Go S, where the original result holds exactly. Footprint is predictable:

```
peak RSS  ≈  1.29 GB (resident weights)  +  KV cache  +  slots × 30 × 3.2 MB
```

Two ceilings bound `--slots`, and both now report themselves at startup rather than failing later:

* **The descriptor heap**, which is now derived from the slot count instead of being hardcoded at
  65,536. That constant fit exactly 44 slots and then threw `Descriptor heap exhausted`
  *mid-generation*, because descriptor tables are created lazily as slots are first touched.
* **The adapter's shared-memory budget** — about half of installed RAM (12.2 GB here), not
  installed RAM itself. Overshooting it does not degrade, it fails.

> **Benchmarking warnings.** Three traps have each already produced a confident wrong conclusion
> on this project:
>
> 1. **Run-to-run drift** is larger than most single optimizations — the same binary measured
>    8.6–9.0 tok/s in one session and 7.1–7.9 in another. Compare variants **interleaved within
>    one session**, never across sessions.
> 2. **Background disk I/O invalidates the measurement.** Decode is I/O-bound. A concurrent game
>    download made a bit-identical workload read 5.07 tok/s instead of 6.03. Confirm
>    `Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length'` is idle first.
> 3. **A sequential sweep measures page-cache warmth, not your variable.** Sweeping `--slots` in
>    increasing order shows everything getting faster; that artefact once turned a real +27% into
>    an apparent +50%. Alternate A/B/A/B for at least three rounds and compare medians.
>
> And check the exit code — a failed run exits non-zero, but a `grep` for the metric lines renders
> it as an empty row that reads like a slow result rather than a crash.

### Known issue

Rarely, and not reproducibly, a CLI run has exited **139** with no output during back-to-back
benchmarking loops — twice in about forty runs in one session, then not once in the twenty-three
consecutive runs that tried to provoke it again. Both failures followed a run at a different
`--slots` value, which points at allocation pressure from the previous process's multi-GB
commitment not yet being reclaimed. **This is not understood and not fixed.** If you hit it,
please file the full stdout, stderr, and exit code.

## Parity with the reference

The forward pass and format match. The product surface around them is still catching up.

**Matches:**

* `.gturbo` bundle format, byte for byte — expert stride 3,358,720, `packed_experts/` total
  12,897,484,800 B, and an identical resident index sha256. Bundles are interchangeable between
  the two implementations.
* Forward-pass numerics: GPU greedy output matches this port's CPU reference token for token,
  and all 15 kernels pass `test_gpu_kernels`.
* Sampling, including the reference's Top-P → Top-K → temperature ordering (not HuggingFace's)
  and applying the repetition penalty to already-softcapped logits.
* String stop sequences, streaming token output, `/health`, `/v1/models`,
  `/v1/chat/completions` with SSE.
* Multi-turn chat, via the engine's `generate_chat` entry point — though the front-ends do not
  send history yet.
* The expert cache is configurable without a fixed ceiling: the descriptor budget is derived from
  the slot count, and the GUI's slot and context changes are genuinely applied on reload.

**Not yet implemented:**

| Gap | Note |
|---|---|
| Thought-channel suppression | The generation prompt opens a `thought` channel and nothing parses it |
| Tool calling and channel decoding | — |
| KV reuse across turns (prompt cache) | Every request re-prefills from scratch; no `--prompt-cache-mode` |
| Chunked/batched prefill | ~7.6 tok/s here vs ~28 in the reference. Biggest remaining opportunity |
| FP16 KV, context above 4096 | Needs an online-softmax rewrite of `Attention.hlsl` |
| `--messages-file`, `--expert-cache-policy`, `--prefill`, `--prefill-chunk-tokens`, `--rdadvise` | Not stubbed — a flag that parses and does nothing is worse than a clean error |
| Repack/installer GUI | Use `tools/convert_hf_to_gturbo.py` |
| Runtime integrity verification, symlink rejection | — |

`chat_template.jinja` ships in the bundle and is never read — the chat template is hardcoded
C++. That is deliberate, but a bundle with a different template would render silently wrong.

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — engine components, the per-layer dispatch
  graph, the HLSL kernels, and the forward-pass constants that are easy to get subtly wrong.
* [docs/GTURBO_FORMAT.md](docs/GTURBO_FORMAT.md) — the on-disk bundle format.
* [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — what made it fast, and what didn't.
* [CONTRIBUTING.md](CONTRIBUTING.md) — how to change a kernel without breaking correctness.
* [CLAUDE.md](CLAUDE.md) — orientation notes, also useful to human contributors.

The upstream Swift/Metal implementation is not vendored here. Clone
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) separately if you want to
diff against it.

## Reporting issues

**Windows, Direct3D, driver, and build issues belong in
[this repository's issue tracker](https://github.com/justin-creator222/Turbo-WinFare/issues)
only.** Do not open them on `drumih/turbo-fieldfare`, and do not send Windows-specific code
there as a pull request. Upstream is the right place only for questions about the macOS/Metal
implementation itself.

## License and model terms

Turbo-WinFare's source and documentation are licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution to the original work and the
notice of modification required by Apache-2.0 §4(b), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the model and dependency review.

Model weights are not included and are governed by their own terms.
