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

    std::error_code ec;
    fs::remove(layer_path, ec);

    std::cout << "[TEST] All expert streamer tests passed.\n";
    return 0;
}
