## What this changes

<!-- What and why. If it fixes an issue, link it. -->

## Checklist

- [ ] `ctest --test-dir build --output-on-failure` passes locally.
- [ ] No weights, `.gturbo` bundles, build output, or binaries (`.dll`, `.exe`, `.a`, `.zip`) are
      included in this PR.
- [ ] Any new shader file is saved as **ASCII with no BOM** (`Set-Content -Encoding ascii`).
- [ ] No new flag parses and then does nothing; unknown arguments still exit 2.

### If this touches a kernel, a forward-pass constant, the streamer, or the KV cache

- [ ] `.\build\test_gpu_kernels.exe` passes — all 15 kernels.
- [ ] Greedy output still matches between paths, on the same prompt:
      `--cpu --prompt "..." --max-tokens 12` vs `--prompt "..." --max-tokens 12`.
      Paste both outputs below.

### If this claims a performance change

- [ ] Variants were compared **interleaved within one session**, several rounds each, with
      medians reported. Cross-session comparisons are not evidence — see
      [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Test output

<!-- Paste the relevant output. -->

## Environment

<!-- GPU, driver version, Windows build, RAM. -->
