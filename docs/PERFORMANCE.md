# Performance

All numbers below were measured on a 24 GB Ryzen/Radeon development box generating 24 tokens.
`--slots N` overrides the auto-sized expert cache; the breakdown prints on every CLI run unless
`--quiet`.

> **Read this before comparing anything.** Run-to-run drift on this machine is larger than most
> single optimizations: the same bit-identical binary measured 8.6–9.0 tok/s in one session and
> 7.1–7.9 tok/s in another. **Compare variants interleaved within one session.** The progression
> table below was measured sequentially and its *relative* gains hold, but treat the absolute
> numbers as one session's snapshot.

## Measurement hygiene — two traps that have already cost a day

Both of these produced confident, wrong conclusions. Neither is visible in the output.

**1. Background disk I/O invalidates the measurement completely.** An audit once reported a
≈40% regression against this document. There was no regression: a game download was running.
Same binary, same flags, *bit-identical* workload (38,985 MB in 12,171 reads, 64.78% cache hit
both times) — the only difference was the machine:

| 24 slots, 120 tokens | contended | idle |
|---|---|---|
| decode | 5.07 tok/s | **6.03** |
| expert I/O | 107.9 ms | **81.9** |
| I/O throughput | 1957 MB/s | **3306** |
| CPU other | 11.3 ms | **4.3** |
| GPU wait | 78.1 ms | 79.3 (unchanged) |

Decode is I/O-bound, so anything touching the NVMe competes directly — and a large download also
evicts the expert pages the streamer deliberately leans on the OS page cache to hold. Confirm the
disk is idle before believing any number:

```powershell
(Get-Counter '\PhysicalDisk(_Total)\Current Disk Queue Length' -SampleInterval 1 -MaxSamples 3).CounterSamples.CookedValue
```

**2. A sequential sweep measures page-cache warmth, not the variable you changed.** Sweeping
`--slots 16, 24, 32, 40` in order shows every configuration getting faster, because each run
inherits a warmer cache from the last. That artefact once turned a real +27% into an apparent
+50%, and made 16 slots tie with 24. **Alternate configurations A/B/A/B for at least three
rounds and compare medians.** Interleaved, the spread within one configuration is ±0.05 tok/s;
sequentially it is over 1 tok/s.

**Check the exit code.** A failed run prints a diagnostic and exits non-zero, but a `grep` for
metric lines shows an empty row that looks like a slow result rather than a crash.

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

## Trading RAM for speed (`--slots`)

Defaults are deliberately conservative and **do not change**: the ladder auto-sizes to 16 / 24 / 32
slots per layer at ≤16 / ≤24 / ≥32 GB installed. Raising it is an explicit opt-in, either
`--slots N` on the CLI or the sidebar control followed by a model reload.

Measured on the 24 GB box, interleaved, 120 tokens, machine idle:

| slots | pool | peak RSS | cache hit | decode |
|---|---|---|---|---|
| 16 | 1.5 GB | ~3.3 GB | 54.1% | 6.0 |
| 24 *(auto here)* | 2.3 GB | 4.3 GB | 64.8% | **7.0** |
| 44 | 4.1 GB | 6.2 GB | 82.7% | **9.2** |
| 64 | 6.0 GB | ~8.1 GB | 90.3% | 7.8–8.9 |

**About +30% for roughly 2 GB**, and the sweet spot on this machine is around 44. Note that 64
slots is *not* faster despite a 90% hit rate: the pool competes with the OS page cache for the
same physical memory, so past a point more cache buys hit rate and gives back throughput. The
same effect is why the auto-size ladder stops at 32. Footprint is predictable:

```
peak RSS  ≈  1.29 GB (resident weights)  +  KV cache  +  slots × 30 × 3.2 MB
```

Two ceilings bound `--slots`, and both now report themselves at startup rather than failing
later:

- **Descriptor heap.** Capacity is derived from the slot count by
  `ComputePipelineManager::descriptors_for()`. Previously hardcoded at 65,536, which fit exactly
  44 slots and then threw `Descriptor heap exhausted` *mid-generation*, dozens of tokens in.
- **The adapter's shared-memory budget**, typically about half of installed RAM (12.2 GB here),
  is the real cap on UMA allocation — not installed RAM. Overshooting it does not degrade, it
  fails, so a warning fires at load when the pool plus resident weights will not fit.

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
