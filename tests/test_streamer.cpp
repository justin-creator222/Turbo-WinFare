// Expert cache slot policy: hits, eviction order, and pinning.
//
// The pinning rules are the reason this file exists. A plan that loses one of its slots to a
// later eviction does not crash and does not produce garbage -- the caller binds a different
// expert's weights and the model keeps writing fluent text that is simply wrong. That is the
// hardest class of bug to notice by reading output, so it is asserted here instead.

#include "gturbo/streamer.hpp"
#include "gturbo/d3d12_context.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// Last include on purpose: check.hpp redefines assert so it survives NDEBUG, and <cassert>
// (pulled in by anything included after it) would silently define assert back to a no-op.
#include "check.hpp"

namespace fs = std::filesystem;

namespace {

// Small enough to keep the test fast; the streamer never inspects the contents, only the
// offsets it reads from.
constexpr uint64_t kStride = 64 * 1024;
constexpr int kExpertCount = 16;

std::string make_layer_file() {
    fs::path p = fs::temp_directory_path() / "gturbo_test_layer.bin";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    std::vector<char> block(static_cast<size_t>(kStride));
    for (int e = 0; e < kExpertCount; ++e) {
        // Stamp each block with its expert id so a misdirected read is detectable.
        std::fill(block.begin(), block.end(), static_cast<char>(e));
        out.write(block.data(), static_cast<std::streamsize>(block.size()));
    }
    out.close();
    return p.string();
}

int expert_of(const gturbo::ExpertSlot* slot) {
    return static_cast<const unsigned char*>(slot->host_ptr)[0];
}

} // namespace

int main() {
    // Unbuffered. A pinning bug surfaces as an uncaught "no evictable slot" exception,
    // and std::terminate does not flush std::cout -- so a buffered failure prints nothing at
    // all, losing every PASS line that would have said how far the run got.
    std::cout.setf(std::ios::unitbuf);
    std::cout << "[TEST] Running expert streamer cache tests...\n";

    auto ctx = std::make_shared<gturbo::D3D12Context>();
    try {
        ctx->initialize(false);
    } catch (const std::exception& ex) {
        // Same soft-skip as test_format: these tests need real UMA buffers, and a machine
        // with no D3D12 device should not fail the suite.
        std::cout << "  [NOTE] No D3D12 device (" << ex.what() << "); skipping.\n";
        return 0;
    }

    const std::string layer_path = make_layer_file();

    // ---- Cold pool: every request is a miss, and every slot ends up pinned. ----
    {
        gturbo::ExpertStreamer s(ctx, layer_path, 8, kStride, gturbo::EvictionPolicy::LFU);
        s.initialize();

        auto plan = s.plan_experts(0, {1, 2, 3});
        assert(plan.slots.size() == 3);
        assert(plan.hits.empty());
        assert(plan.misses.size() == 3);
        for (auto* slot : plan.slots) assert(slot->pinned);

        s.fetch_misses(plan);
        // The read landed on the right offset for each request.
        assert(expert_of(plan.slots[0]) == 1);
        assert(expert_of(plan.slots[1]) == 2);
        assert(expert_of(plan.slots[2]) == 3);

        s.release_plan(plan);
        assert(s.total_misses() == 3 && s.total_hits() == 0);
        std::cout << "  [PASS] Cold pool: all misses, correct offsets, pinned during the plan.\n";

        // ---- Warm pool: the same experts are hits and need no I/O. ----
        uint64_t io_before = s.total_io_calls();
        auto plan2 = s.plan_experts(0, {1, 2, 3});
        assert(plan2.hits.size() == 3);
        assert(plan2.misses.empty());
        s.fetch_misses(plan2);
        s.release_plan(plan2);
        assert(s.total_io_calls() == io_before);   // a hit must not re-read
        assert(s.total_hits() == 3);
        std::cout << "  [PASS] Warm pool: hits served without I/O.\n";
    }

    // ---- release_plan must release only its own slots. ----
    //
    // It used to unpin the entire pool, which happened to be harmless while exactly one plan
    // was ever in flight. The moment two overlap -- queued requests, or double-buffered
    // prefill tiles -- one plan's experts become evictable while it is still binding them.
    {
        gturbo::ExpertStreamer s(ctx, layer_path, 8, kStride, gturbo::EvictionPolicy::LFU);
        s.initialize();

        auto held = s.plan_experts(0, {10, 11});
        s.fetch_misses(held);

        auto other = s.plan_experts(0, {12, 13});
        s.fetch_misses(other);
        s.release_plan(other);

        // `held` was never released, so its slots must still be pinned.
        for (auto* slot : held.slots) {
            assert(slot->pinned && "release_plan unpinned a slot belonging to another plan");
        }
        // And still hold the right experts.
        assert(expert_of(held.slots[0]) == 10);
        assert(expert_of(held.slots[1]) == 11);

        s.release_plan(held);
        for (auto* slot : held.slots) assert(!slot->pinned);
        std::cout << "  [PASS] release_plan releases only its own slots.\n";
    }

    // ---- LRU evicts the least recently used; LFU the least frequently used. ----
    {
        // Pool of 2. Load A and B, then touch A so it is both more recent and more frequent,
        // then request C: both policies must evict B, which proves the scan runs at all.
        gturbo::ExpertStreamer s(ctx, layer_path, 2, kStride, gturbo::EvictionPolicy::LRU);
        s.initialize();

        auto p1 = s.plan_experts(0, {4});
        s.fetch_misses(p1); s.release_plan(p1);
        auto p2 = s.plan_experts(0, {5});
        s.fetch_misses(p2); s.release_plan(p2);
        auto p3 = s.plan_experts(0, {4});          // touch 4
        s.fetch_misses(p3); s.release_plan(p3);

        auto p4 = s.plan_experts(0, {6});          // must evict 5, not 4
        s.fetch_misses(p4); s.release_plan(p4);

        auto probe = s.plan_experts(0, {4});
        assert(probe.hits.size() == 1 && "LRU evicted the recently used expert");
        s.release_plan(probe);
        std::cout << "  [PASS] LRU keeps the recently used slot.\n";
    }

    {
        gturbo::ExpertStreamer s(ctx, layer_path, 2, kStride, gturbo::EvictionPolicy::LFU);
        s.initialize();

        for (int i = 0; i < 3; ++i) {                  // 7 gets frequency 3
            auto p = s.plan_experts(0, {7});
            s.fetch_misses(p); s.release_plan(p);
        }
        auto p8 = s.plan_experts(0, {8});              // 8 gets frequency 1
        s.fetch_misses(p8); s.release_plan(p8);

        auto p9 = s.plan_experts(0, {9});              // must evict 8
        s.fetch_misses(p9); s.release_plan(p9);

        auto probe = s.plan_experts(0, {7});
        assert(probe.hits.size() == 1 && "LFU evicted the frequently used expert");
        s.release_plan(probe);
        std::cout << "  [PASS] LFU keeps the frequently used slot.\n";
    }

    // ---- A batch cannot be as large as the pool. ----
    //
    // With K experts and exactly K slots, the batch can evict its own earlier entries and the
    // caller binds one expert's weights in place of another's. The pool must be strictly
    // larger, and asking for otherwise is an error rather than a silent corruption.
    {
        gturbo::ExpertStreamer s(ctx, layer_path, 4, kStride, gturbo::EvictionPolicy::LFU);
        s.initialize();

        bool threw = false;
        try {
            auto p = s.plan_experts(0, {1, 2, 3, 4});
            s.release_plan(p);
        } catch (const gturbo::GTurboFormatError&) {
            threw = true;
        }
        assert(threw && "a batch the size of the pool must be rejected");
        std::cout << "  [PASS] Batch sized to the pool is rejected.\n";
    }

    // ---- A failed read must not leave a slot claiming an expert it does not hold. ----
    //
    // find_or_evict_slot stamps expert_id onto the victim BEFORE the bytes arrive, so a batch
    // that fails partway would otherwise serve a later request a *hit* full of the previous
    // expert's weights -- fluent output, wrong model, no error anywhere. Verified by
    // deliberately removing the invalidation and watching this go red.
    //
    // SCOPE, stated because the neighbouring bug is easy to assume is covered here and is
    // not: this does NOT exercise fetch_misses' other failure-path obligation, draining reads
    // that are still in flight. With a 64 KB stride served from the OS page cache every
    // ReadFile completes inline, so by the time await_read throws there is nothing pending,
    // and reverting the drain leaves this test green. A fixture that reliably has I/O
    // outstanding at throw time would have to race the page cache, which is a flaky test
    // rather than a real one. The runtime guard in issue_read is what actually catches that
    // case -- reusing a slot whose read has not retired is rejected instead of silently
    // overwriting a live OVERLAPPED.
    {
        // A file one block short of what the plan asks for: expert kExpertCount-1 reads past
        // the end and comes back short, while the earlier experts in the batch succeed.
        fs::path truncated = fs::temp_directory_path() / "gturbo_test_layer_short.bin";
        {
            std::ifstream in(layer_path, std::ios::binary);
            std::vector<char> bytes(static_cast<size_t>(kStride) * (kExpertCount - 1));
            in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }

        gturbo::ExpertStreamer s(ctx, truncated.string(), 8, kStride,
                                 gturbo::EvictionPolicy::LFU);
        s.initialize();

        bool threw = false;
        {
            auto plan = s.plan_experts(0, {0, 1, 2, kExpertCount - 1});
            try {
                s.fetch_misses(plan);
            } catch (const gturbo::GTurboFormatError&) {
                threw = true;
            }
            assert(threw && "a read past the end of the layer file must be reported");
            // fetch_misses releases the plan itself on failure.
            assert(!plan.valid() && "a failed fetch must leave the plan released");
        }

        // Nothing pinned, and nothing claiming an expert it does not hold.
        auto reuse = s.plan_experts(0, {0, 1, 2});
        assert(reuse.slots.size() == 3 &&
               "the pool must still be usable after a failed batch");
        assert(reuse.hits.empty() &&
               "a slot whose read failed must not be served as a hit");
        s.fetch_misses(reuse);
        for (size_t i = 0; i < reuse.slots.size(); ++i) {
            assert(expert_of(reuse.slots[i]) == static_cast<int>(i) &&
                   "slot holds another expert's bytes after a failed batch");
        }
        s.release_plan(reuse);

        std::error_code ec;
        fs::remove(truncated, ec);
        std::cout << "  [PASS] A failed read leaves no slot claiming an expert it lacks.\n";
    }

    // ---- An abandoned plan releases itself. ----
    //
    // release_plan is called ~115 lines after plan_experts in the decode loop, with the whole
    // routed-expert encode in between. Anything throwing in that window used to leak the
    // plan's pins for good: the slots stayed pinned, and the NEXT token died with "no
    // evictable slot" -- a recoverable error becoming a dead runner one token later.
    {
        gturbo::ExpertStreamer s(ctx, layer_path, 5, kStride, gturbo::EvictionPolicy::LFU);
        s.initialize();

        for (int round = 0; round < 4; ++round) {
            // Scope exit stands in for the exception unwind. Note release_plan is never
            // called; if the destructor did not release, round 1 would already fail.
            auto plan = s.plan_experts(0, {round, round + 1, round + 2, round + 3});
            assert(plan.slots.size() == 4);
            for (auto* slot : plan.slots) assert(slot->pinned);
        }

        auto after = s.plan_experts(0, {0, 1, 2, 3});
        assert(after.slots.size() == 4 &&
               "abandoned plans leaked their pins: the pool is exhausted");
        std::cout << "  [PASS] An abandoned plan releases its pins on scope exit.\n";
    }

    // ---- Slot-count matrix: the counters that justify raising --slots. ----
    //
    // The measured case for a bigger expert cache rests entirely on hit rate, so the
    // relationship between pool size and hits is asserted here rather than inferred from a
    // benchmark. Wall-clock is deliberately not involved: it drifts with page-cache warmth
    // and background I/O by more than the effect being measured.
    {
        // A fixed, deliberately skewed trace -- real routing is far from uniform, and a
        // uniform trace would understate what the cache buys.
        std::vector<std::vector<int>> trace;
        for (int step = 0; step < 40; ++step) {
            std::vector<int> batch;
            for (int j = 0; j < 4; ++j) {
                // Two thirds of requests land on experts 0..3, the rest spread over 0..15.
                const int r = (step * 7 + j * 5) % 12;
                batch.push_back(r < 8 ? (r % 4) : (r % kExpertCount));
            }
            // plan_experts requires distinct ids within a batch; de-duplicate.
            std::vector<int> uniq;
            for (int id : batch) {
                bool seen = false;
                for (int u : uniq) seen = seen || (u == id);
                if (!seen) uniq.push_back(id);
            }
            trace.push_back(uniq);
        }

        double prev_hit_rate = -1.0;
        for (size_t slots : {5u, 6u, 8u, 12u, 16u}) {
            gturbo::ExpertStreamer s(ctx, layer_path, slots, kStride,
                                     gturbo::EvictionPolicy::LFU);
            s.initialize();

            uint64_t expected_reads = 0;
            for (const auto& batch : trace) {
                auto plan = s.plan_experts(0, batch);
                // Every slot handed back must be pinned, or a later expert in the same
                // batch could evict an earlier one and the caller would bind the wrong
                // weights -- fluent output, wrong model.
                for (auto* slot : plan.slots) {
                    assert(slot != nullptr && slot->pinned && "planned slots must be pinned");
                }
                assert(plan.hits.size() + plan.misses.size() == batch.size());
                expected_reads += plan.misses.size();

                s.fetch_misses(plan);

                // Contents must match the id requested -- the stamp catches a misdirected
                // read that counters alone would not.
                for (size_t i = 0; i < batch.size(); ++i) {
                    assert(expert_of(plan.slots[i]) == batch[i] && "slot holds another expert");
                }
                s.release_plan(plan);
            }

            // I/O must be performed exactly for misses and never for hits. This is the
            // regression that cost the most: load_expert once re-read unconditionally, so
            // the cache tracked a hit rate it never acted on and every byte was fetched
            // twice over.
            assert(s.total_io_calls() == expected_reads && "reads must equal misses exactly");
            assert(s.total_misses() == expected_reads);
            assert(s.total_bytes_read() == expected_reads * kStride);

            // More cache must never mean fewer hits.
            assert(s.hit_rate_pct() >= prev_hit_rate - 1e-9 &&
                   "hit rate must be monotonically non-decreasing in slot count");
            prev_hit_rate = s.hit_rate_pct();

            assert(s.total_cache_memory_bytes() == slots * kStride);
        }
        std::cout << "  [PASS] Hit rate is monotonic in slot count; reads equal misses exactly.\n";

        // Once the pool can hold the whole working set, steady state is zero I/O -- the
        // property that makes a larger --slots worth its memory.
        {
            gturbo::ExpertStreamer s(ctx, layer_path, kExpertCount + 1, kStride,
                                     gturbo::EvictionPolicy::LFU);
            s.initialize();
            for (int pass = 0; pass < 3; ++pass) {
                for (const auto& batch : trace) {
                    auto plan = s.plan_experts(0, batch);
                    s.fetch_misses(plan);
                    s.release_plan(plan);
                }
                if (pass == 0) {
                    // Everything resident after one pass; nothing may be re-read after that.
                    const uint64_t after_warm = s.total_io_calls();
                    assert(after_warm <= kExpertCount && "warm-up must not re-read experts");
                }
            }
            assert(s.total_io_calls() <= kExpertCount &&
                   "a pool larger than the working set must reach zero steady-state I/O");
            std::cout << "  [PASS] A pool covering the working set stops reading entirely.\n";
        }
    }

    std::error_code ec;
    fs::remove(layer_path, ec);

    std::cout << "[TEST] All expert streamer tests passed.\n";
    return 0;
}
