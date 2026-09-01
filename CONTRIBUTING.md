# Contributing to Turbo-WinFare

Thanks for taking a look. A few things are specific to this project and will save you time.

## Scope, and the upstream boundary

Turbo-WinFare is a Windows/Direct3D 12 port of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) (Swift/Metal, macOS).

* **Windows, Direct3D, driver, and build issues belong here.** Do not open them upstream.
* **Do not send Windows-specific C++/HLSL to `drumih/turbo-fieldfare` as a pull request.**
  Upstream is a macOS/Metal project and has not asked for a multi-platform core.
* If you find a bug that is genuinely in the *shared design* — a wrong forward-pass constant, a
  format ambiguity — it is worth mentioning upstream, politely, as an issue.

## Setting up

```powershell
python tools/download_toolchain.py
$env:PATH = "C:\w64devkit\bin;" + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

MSVC is not supported. `spec.md`-era notes suggesting otherwise are obsolete.

Most tests need neither a GPU nor a model: `test_format`, `test_tokenizer`, `test_streamer`,
`test_sampling`, `test_detokenizer`, `test_server`, `test_openai`, `test_convert`. CI runs
exactly those. `test_gpu_kernels` needs a D3D12 device with Shader Model 6.6 and must be run
locally.

## The correctness gate

This project's whole claim is that GPU output is *right*, not just plausible. Wrong inference
output is fluent, and fluent-but-wrong is the hardest failure mode to notice.

**If you touch a kernel, a forward-pass constant, the streamer, or the KV cache, you must run
both of these and say so in the PR:**

1. `.\build\test_gpu_kernels.exe` — all 15 kernels against the CPU reference.
2. The token-for-token greedy check:
   ```powershell
   .\build\turbo-winfare.exe --cpu --prompt "What is the capital of France?" --max-tokens 12
   .\build\turbo-winfare.exe       --prompt "What is the capital of France?" --max-tokens 12
   ```
   Greedy output from the two paths must be identical. `--dump-tensors <dir>` (with `--cpu`)
   writes per-stage FP32 tensors for the first token if you need to find *where* they diverge.

Kernel agreement uses numpy's `allclose` rule (`|got-want| <= 1e-5 + 1e-4*|want|`), not a pure
relative test — GPU transcendentals differ from libm by a couple of ULPs and near-zero results
then show a huge relative error for ~1e-7 absolute. A real logic bug is never marginal.

## Rules that exist because something broke

* **No fabricating paths.** Never make a missing model, tokenizer, shader compiler, or telemetry
  value degrade into a plausible-looking substitute. Missing weights, a missing tokenizer, and a
  missing DXC are all hard errors on purpose. There is deliberately **no `cs_5_0` shader
  fallback** — the old one `#define`d the wave intrinsics to identity and silently reduced every
  cross-lane reduction to one lane's partial value.
* **Save shaders as ASCII with no BOM.** `Set-Content -Encoding ascii`, never `-Encoding utf8`.
  DXC rejects a BOM with a misleading "non-ASCII characters are not allowed" pointing at the
  wrong line.
* **Do not assume a wave width.** RDNA 3 runs a 256-thread group as Wave64. Use
  `WaveGetLaneCount()` / `WaveIsFirstLane()`.
* **Do not add a flag that parses and does nothing.** A clean "unknown argument" error is better
  than a silently ignored option. Unknown arguments exit 2 by design.
* **Never commit weights, `.gturbo` bundles, or build output.** `.gitignore` covers these; check
  `git status` before committing anyway.

## Benchmarking

Run-to-run drift on this class of machine is larger than most single optimizations. **Compare
variants interleaved within one session**, several rounds each, and report medians. A
before-session vs after-session comparison is not evidence. See
[docs/PERFORMANCE.md](docs/PERFORMANCE.md), which records two ideas that looked like wins or
losses under sequential measurement and turned out to be neither.

## Style

Match the surrounding code: C++23, `gturbo::` namespace, headers in `include/gturbo/`. Comments
in this codebase explain *why*, especially where a previous approach was wrong — that convention
is worth keeping.

## Licensing of contributions

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license as the project. If you port logic from another
project, say so in the PR so it can be recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
