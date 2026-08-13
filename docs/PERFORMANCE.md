# Performance

All numbers below were measured on a 24 GB Ryzen/Radeon development box generating 24 tokens.
`--slots N` overrides the auto-sized expert cache; the breakdown prints on every CLI run unless
`--quiet`.

> **Read this before comparing anything.** Run-to-run drift on this machine is larger than most
> single optimizations: the same bit-identical binary measured 8.6–9.0 tok/s in one session and
> 7.1–7.9 tok/s in another. **Compare variants interleaved within one session.** The progression
> table below was measured sequentially and its *relative* gains hold, but treat the absolute
> numbers as one session's snapshot.

## Where it got to

| Stage | decode | expert I/O | GPU wait | CPU other |
|---|---|---|---|---|
| Stage 3 baseline | 2.34 tok/s | 237 ms | 152 ms | 38 ms |
| + buffered I/O, parallel reads, skip-read-on-hit | 4.72 | 87 | 117 | 26 |
| + auto-sized slot pool (24/layer) | 5.66 | 44 | 115 | 24 |
| + allocator ring, 62 → 31 fences | 6.04 | 44 | 103 | 29 |
| + hit-first split, shared expert overlapping I/O | 8.06 | 47 | 62 | 35 |
| + descriptor cache | 8.89 | 48 | 60 | 23 |
| + fused greedy LM head | **8.8–9.0** | 48 | 60 | 23 |

Reference points: TurboFieldfare reaches 5.1–6.3 tok/s on an 8 GB M2 Air. The 16-slot
configuration here — the 16 GB Legion Go S target — reaches **6.54 tok/s**, so the reference is
beaten on the constrained configuration too.

**Where the time goes now:** ~34% expert I/O, ~41% GPU wait, ~9% LM head, ~16% other CPU. The
engine is nowhere near compute- or DRAM-bound — Stage 3 measured 0.2% of peak ALU and ~6% of DRAM
bandwidth — so the remaining headroom is in kernel efficiency, not hardware.

## What turned out to matter

**The expert cache was counting hits it never acted on.** `load_expert` called
`read_expert_unbuffered` unconditionally, so a 51.6% hit rate still re-read every byte from disk.
Skipping the read on a hit halved I/O volume outright. A cache hit rate is not evidence that a
cache is doing anything.

**`FILE_FLAG_NO_BUFFERING` was throwing away the OS page cache.** The reference leans on the
equivalent macOS cache deliberately. Removing the flag, plus issuing the 8 misses concurrently
instead of at queue depth 1, took expert I/O from 237 ms to 87 ms. The streamer therefore does
**not** use unbuffered I/O, despite the 16 KB alignment in the file format being designed for it.

**Overlapping GPU work with the reads was the single biggest scheduling win** (6.04 → 8.06). The
shared expert depends only on `dense_x`, and cache-hit experts need no I/O at all, so both are
submitted *before* the miss reads are issued.

**Slot pools do not scale freely.** They compete with the OS page cache for the same RAM. The
pool is auto-sized from installed RAM (16/24/32 slots at ≤16/24/≥32 GB). The reference measured
32 slots collapsing their 8 GB host from 5.6 to 1.58 tok/s.

**Fence count is the cost, not submission count.** The allocator ring took the per-token fence
waits from 62 to 31 (one per layer plus the head) while leaving ~92 submissions in place.

## Measured and rejected

Both of these were structurally reasonable ideas that produced no measurable gain. They are
recorded so nobody spends the time twice.

**A second compute queue.** Before building one, the precondition was tested by giving the shared
expert its own scratch buffers, removing the false dependency that serialized it against the
routed experts. Result: **no change** — 8.75–8.79 vs 8.59–8.99, GPU wait flat at ~60.7 ms. Each
GEMV already launches thousands of threadgroups and saturates the device, so there is no idle
capacity for a second queue to exploit. The independent scratch buffers were kept, since they do
remove a real false dependency, but they bought nothing measurable.

**Fusing the greedy LM head** was structurally right — it drops a full Softcap pass and a 1 MB
logit readback — but moved 13.4 ms to 13.3 ms. That GEMV is bandwidth-bound reading the 396 MiB
tied embedding table, which fusion does not change.

**`Load4` vectorized activations — reverted, measured NEUTRAL.** Interleaved A/B over 4 rounds:
scalar median 7.84 tok/s, `Load4` median 7.80, spread ±0.1. DXC already merges the scalar loads
into wide fetches, so hoisting them by hand changes nothing. A first, non-interleaved comparison
appeared to show a 15% regression; that was machine drift, not the change — which is exactly why
the warning at the top of this file exists. See the comment in
[../shaders/Common.hlsli](../shaders/Common.hlsli).

## Capability changes

**Ring-buffer KV cache — kept.** Sliding-window layers now use `sliding_window` slots via
`physical_slot(L, p) = p % capacity`, independent of total context; only the 5 full-attention
layers scale with it. **Context went 1024 → 4096** (2048 on ≤16 GB) for 560 MB of KV, versus the
~1.85 GB that raising context without the ring would have cost. Decode speed is unchanged, as
expected — this is a capability change, not a speed one.

## Not yet done

Each of these would require re-running the CPU-vs-GPU token-for-token gate:

* **Batched/chunked prefill with RDNA3 WMMA.** Prefill is ~5.7 tok/s here and is the reference's
  weak spot too (~28 tok/s). This is the biggest remaining opportunity.
* **FP16 activations.** Halves activation bandwidth, but changes numerics enough to cost the
  token-for-token agreement gate.
* **Lane occupancy on routed `down_proj`** — currently 34%, only 11 of 32 lanes busy.
* **Kernel fusion** beyond the LM head.
