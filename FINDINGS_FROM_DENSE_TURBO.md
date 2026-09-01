# Findings from Turbo-WinFare Dense

**Date:** 2026-08-31 · **Author:** Claude Opus 5 · **For:** Turbo-WinFare (this repo)

Turbo-WinFare Dense — the Gemma 4 31B Dense engine in `Code/Dense Turbo` — was built from this
project's structure. Over ten rounds of work it hit a number of problems, and several of them
turned out to be inherited rather than new. This is a list of the ones that look relevant here.

**Nothing in this repository was modified.** I read the source to check each item; this file is
the only thing added. No build, no benchmark, no commit.

**How to read it.** Every claim below is marked with how it was established:

- **Verified here** — I read this repo's code and confirmed the condition exists.
- **Verified absent** — I checked and it does *not* apply. These are worth as much as the
  positives; they are items you can skip.
- **Divergent** — Dense measured the opposite of what this repo measured. Neither result is
  obviously wrong.

I did not run this engine, so nothing below is a measurement *of this code*. The numbers quoted
are Dense's, on the same machine (Ryzen Z1 Extreme / Radeon 780M / 32 GB), and they indicate
size, not a prediction.

---

## 1. The submit path discards its error codes — **verified here**

`src/d3d12_context.cpp:103` `submit_command_list()`:

```cpp
list->Close();                                  // HRESULT discarded
command_queue_->ExecuteCommandLists(1, lists);  // returns void
const uint64_t val = signal_fence();            // Signal()'s HRESULT discarded
```

`GetDeviceRemovedReason` does not appear anywhere in `src/`. So if the device is lost — which on
an APU most plausibly follows an over-greedy allocation — `Close()` and `Signal()` begin failing,
every subsequent submit silently does nothing, and the engine keeps running.

**Why this is worth fixing first.** Dense had the same gap on the Vulkan side and it cost a full
round. A failing `vkQueueSubmit` presented as **~0 ms of GPU time and "1.93 TPS" alongside empty
output** — a 20× apparent speedup that was pure failure, and it was chased as a *performance
result* before anyone noticed the output was blank. The failure mode is not a crash; it is a
plausible-looking number.

Dense now checks every submit and probes the device once after the memory import completes,
rather than discovering the problem mid-generation. The equivalent here is checking the two
HRESULTs and calling `GetDeviceRemovedReason` when either fails.

## 2. Streaming reads do not overlap compute — **verified here**

`src/streamer.cpp:243` `fetch_misses()` issues every miss and then waits for all of them:

```cpp
for (size_t i : plan.misses) issue_read(...);   // all in flight -- good
for (size_t i : plan.misses) await_read(...);   // ...then block the caller
```

The reads overlap *each other*, which is the harder half and is already right. What they do not
overlap is GPU work: the caller blocks until the last one lands.

Dense went through two stages here.

1. Splitting issue from await, so reads for layer *N+3* are in flight while layer *N* computes.
2. Moving the read itself onto worker threads.

The second was the single largest win in the project's history: **+50% throughput, 0.75 → 1.13
tok/s**, with stream I/O falling from 322 ms to 90 ms per token. The reason it mattered so much
is worth stating plainly, because Dense got it wrong for three rounds: buffered overlapped reads
*complete inline* when the data is already in the page cache, so `ReadFile` blocks the calling
thread even though the handle is `FILE_FLAG_OVERLAPPED`. If that thread is the one submitting GPU
work, nothing overlaps.

**This repo already made the right call on buffering** (`streamer.cpp:56`, and the reasoning
there is sound — Dense eventually arrived at the same place from the opposite direction). The
remaining question is only whether the calling thread is the one that must not block.

How much this is worth depends on the I/O share of a token here, which I have not measured. With
3.3 MB expert reads rather than 286 MB layers, it may be small.

## 3. A stated requirement that is 2.4× the measured result — **verified here**

`Performance.md:13` and `:192` require **"> 22 tokens/sec"** decode. `CLAUDE.md:197` and `:337`
record the measured figure as **8.6–9.0 tok/s**.

Dense carried the same shape of problem: `config/tiers.json` set a 2.5 TPS target that reported
`meets_target: false` for the entire life of the project, because it was written before anything
ran. It was eventually replaced with the measured operating point plus the arithmetic showing
what the hardware permits, and the old projections kept only as a record of what had been
assumed.

A target nothing has ever met stops being a target. Either it is reachable and the gap is a bug
worth finding, or it is not and the document should say what is.

## 4. Load4 for weight loads — **divergent**

`shaders/Common.hlsli:119` documents a measurement and a warning:

> MEASURED: replacing these scalar loads with two Load4 per w iteration is SLOWER on RDNA 3 —
> 7.3-7.8 tok/s against 8.6-9.0 for this version… Do not "optimize" this without measuring.

**Dense measured the opposite**, three runs each way: GPU phase **853 → 748 ms, about 12% faster**
for identical arithmetic, with the kernel going from ~33 GB/s to 41–50 GB/s. That change is in
Dense today.

I checked whether the obvious explanation applies and **it does not**: your tensors are 16 KB
aligned (`tools/convert_hf_to_gturbo.py:81`), so a `Load4` lands on a 16-byte boundary and your
test was valid. Dense's weights were *not* aligned at first — a 2-byte `layer_scalar` put every
projection at offset 2 (mod 16) — and there the vectorised load produced **plausible-looking
garbage**, which is worth knowing if you ever revisit this.

So this is a genuine divergence, not an error on either side. The conditions differ in ways that
could matter: your inner loop reads activations as scalars and accumulates `dot` and `sum`
separately, Dense reads activations as `float4` and folds scale/bias per group. Register pressure
is the likely discriminator, and your note about it is probably right for your loop shape.

Worth one re-measure if the kernel ever becomes the bottleneck; not worth it otherwise.

## 5. Things Dense tried that did **not** work — skip these

Offered so you do not spend the rounds.

| Idea | Dense's measured result |
|---|---|
| **Stage activations in LDS.** Your gemv, like Dense's, re-reads the whole activation vector from global memory once per output row — 16 of every 20 memory instructions. | **+0.8% end-to-end.** A purpose-built, bank-conflict-free version with identical arithmetic moved the GPU phase 607 → 594 ms. Activations were already being served cheaply from L2; they dominate the instruction count without being the bottleneck. |
| **Import a memory-mapped file view** so the GPU reads the page cache with no copy. | The driver **refuses** it outright (`VK_ERROR_INVALID_EXTERNAL_HANDLE`). Closed on Vulkan; may differ on D3D12. |
| **Raise the residency cap** to whatever the driver will accept. | Worse, not better. Asking for more got *fewer* layers on one attempt (34 against a reliable 45), because a mid-sequence refusal aborts the import loop and dumps the remainder into streaming. A conservative cap is what makes residency reproducible. |
| **Cooperative matrix / WMMA.** Supported on the 780M (11 configurations). | Does not help decode: batch-1 GEMV is bandwidth-bound, not compute-bound. Prefill only. |

## 6. Checked here, and **does not apply**

Dense had these; this repo does not. Listed so nobody re-audits them.

| Dense's problem | Status here |
|---|---|
| GUI read telemetry keys the server never emitted, so the phase panel fell back to **hardcoded literals** and displayed invented performance data | **Absent.** Every key `gui/app.js` reads is emitted by `src/server.cpp:483`. The contract is intact. |
| GUI called three endpoints that did not exist, so "Reset KV Cache" reported success while doing nothing | **Absent.** The endpoint sets match exactly. |
| Acceptance rate multiplied by 100 twice, displaying 46.7% as "4670%" | **Absent.** |
| Unbuffered I/O costing more than half the read bandwidth | **Absent, and deliberately so.** `streamer.cpp:56` chose buffered for documented reasons. Dense spent three rounds arriving at the same conclusion. |

## 7. Two process lessons, which cost more than any single bug

**Check which resource is saturated before optimising it.** Dense spent rounds 6, 7 and 8
improving the GPU phase. Two of those changes measured neutral and were honestly recorded as
neutral — but nobody asked *why two independent GPU improvements would both come to nothing*. The
answer was one division: 4.145 GB read per token at the streamer's measured 3.09 GB/s is 1,341 ms,
against a 1,328 ms token. The engine was disk-bound the whole time, and both halves of that
division had been sitting in the ground-truth JSON for months.

**A test that cannot fail is worse than no test.** Two shipped in Dense. The gemv parity test
passed `0` for all three byte offsets, so it never exercised the offset arithmetic the real model
uses — and it passed while the engine produced garbage. The attention cases all fit inside two
tiles, so they could not detect a broken online-softmax rescale. Both were found by deliberately
breaking the thing under test and watching the test stay green. Worth doing to your parity suite
once.

Related: a reference implementation written from the same assumptions as the engine cannot catch
a shared misconception. Seven forward-pass bugs survived GPU-vs-CPU parity in Dense's round 5
because *both* paths shared each mistake. They were only found by transcribing a reference
directly from the upstream modelling source.

## 8. What I did not check

Not an exhaustive audit. I looked for things Dense had already paid for, which biases toward
those and away from anything unique to a mixture-of-experts design. Untouched:

- the router, expert selection, and anything MoE-specific
- correctness of the forward pass or its constants
- the KV cache, sampling, tokenizer, and the OpenAI API surface
- the `.gturbo` format beyond its alignment
- everything under `tests/`

If it would help, the equivalent Dense documents are `docs/PERFORMANCE.md` (measurements and the
ideas they killed) and the per-round reports `docs/ROUND3_REPORT.md` … `ROUND9_REPORT.md`, which
carry the reasoning including the wrong turns.
