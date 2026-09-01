// Capacity planning: the descriptor budget and the expert-slot sizing ladder.
//
// Both are pure logic that used to be reachable only through ForwardRunner::initialize(),
// which hard-requires a tokenizer, a D3D12 device, resident.bin and 30 layer files. That is
// why the descriptor ceiling shipped: nothing could assert on it without the full 13.3 GB
// bundle. These run in milliseconds on any machine, GPU or not.

#include "gturbo/pipeline.hpp"
#include "gturbo/runner.hpp"
#include "gturbo/format.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

// Last include on purpose: check.hpp redefines assert so it survives NDEBUG, and <cassert>
// (pulled in by anything included after it) would silently define assert back to a no-op.
#include "check.hpp"

using gturbo::ComputePipelineManager;
using gturbo::ForwardRunner;

namespace {

// The real formula, restated independently of the implementation. If this and
// descriptors_for() ever disagree, one of them has drifted from the dispatch graph.
//   3 sets per (layer, expert slot)  -- expert_gemv gate/up/down
//   3 sets per layer                 -- k epilogue, v epilogue, Attention
//   20 layer-invariant sets
// times 16 descriptors per set (8 SRV + 8 UAV). No safety margin: this is the count that
// must fit, and descriptors_for() is expected to sit strictly above it.
uint64_t required_descriptors(uint64_t slots, uint64_t layers) {
    return (3ULL * layers * slots + 3ULL * layers + 20ULL) * 16ULL;
}

bool resolve_throws(size_t requested, uint64_t ram, int top_k, int num_experts) {
    try {
        ForwardRunner::resolve_slots(requested, ram, top_k, num_experts);
        return false;
    } catch (const gturbo::GTurboFormatError&) {
        return true;
    }
}

constexpr uint64_t GB = 1024ULL * 1024ULL * 1024ULL;
constexpr int kLayers = 30;      // Gemma 4 26B-A4B
constexpr int kTopK = 8;
constexpr int kNumExperts = 128;

} // namespace

int main() {
    std::cout << "[TEST] Descriptor budget and expert-slot sizing\n";

    // ---- Descriptor budget ------------------------------------------------
    // Every slot count the engine can actually be asked for must fit, with the
    // headroom descriptors_for() promises.
    for (size_t s : {9u, 16u, 24u, 32u, 44u, 64u, 96u, 128u}) {
        const uint32_t got = ComputePipelineManager::descriptors_for(s, kLayers);
        const uint64_t need = required_descriptors(s, kLayers);
        assert(got >= need && "descriptor budget must cover the real binding-set count");
        assert(got <= ComputePipelineManager::kMaxShaderVisibleDescriptors);
    }
    std::cout << "  [PASS] Budget covers 9..128 slots and stays within the tier limit.\n";

    // The regression this whole change exists for: a hardcoded 65536 fit 44 slots and not
    // 45, which is exactly where --slots started dying mid-generation. Pinning both sides
    // of that boundary is what proves the old ceiling is understood rather than guessed at.
    assert(required_descriptors(44, kLayers) <= ComputePipelineManager::kLegacyDescriptorCapacity);
    assert(required_descriptors(45, kLayers) > ComputePipelineManager::kLegacyDescriptorCapacity);
    assert(ComputePipelineManager::max_slots_for_capacity(
               ComputePipelineManager::kLegacyDescriptorCapacity, kLayers) == 44);
    std::cout << "  [PASS] The legacy 65536-descriptor heap is reproduced at exactly 44 slots.\n";

    // Above 44 the derived capacity must actually grow -- otherwise the ceiling has just
    // moved rather than been removed.
    assert(ComputePipelineManager::descriptors_for(64, kLayers) >
           ComputePipelineManager::kLegacyDescriptorCapacity);
    assert(ComputePipelineManager::descriptors_for(128, kLayers) >
           ComputePipelineManager::descriptors_for(64, kLayers));
    // Monotonic in slots: more cache must never ask for fewer descriptors.
    uint32_t prev = 0;
    for (size_t s = 9; s <= 128; ++s) {
        const uint32_t got = ComputePipelineManager::descriptors_for(s, kLayers);
        assert(got >= prev && "descriptor budget must be monotonic in slot count");
        prev = got;
    }
    std::cout << "  [PASS] Budget grows monotonically past the old ceiling.\n";

    // max_slots_for_capacity is the inverse used to tell a user what their device allows,
    // so it must never over-promise.
    for (uint32_t cap : {65536u, 131072u, 262144u, 1000000u}) {
        const size_t s = ComputePipelineManager::max_slots_for_capacity(cap, kLayers);
        assert(required_descriptors(s, kLayers) <= cap);
        assert(required_descriptors(s + 1, kLayers) > cap);
    }
    std::cout << "  [PASS] max_slots_for_capacity never over-promises.\n";

    // ---- Slot sizing ladder ----------------------------------------------
    // Auto-sizing (requested == 0). Thresholds are 22 GB and 30 GB.
    assert(ForwardRunner::resolve_slots(0, 8 * GB,  kTopK, kNumExperts) == 16);
    assert(ForwardRunner::resolve_slots(0, 16 * GB, kTopK, kNumExperts) == 16);
    assert(ForwardRunner::resolve_slots(0, 24 * GB, kTopK, kNumExperts) == 24);
    assert(ForwardRunner::resolve_slots(0, 32 * GB, kTopK, kNumExperts) == 32);
    assert(ForwardRunner::resolve_slots(0, 64 * GB, kTopK, kNumExperts) == 32);
    std::cout << "  [PASS] Auto-size ladder: 16 / 24 / 32 at the 22 and 30 GB thresholds.\n";

    // Exact boundaries, so a future edit cannot slide a threshold by a gigabyte unnoticed.
    assert(ForwardRunner::resolve_slots(0, 22 * GB, kTopK, kNumExperts) == 24);
    assert(ForwardRunner::resolve_slots(0, 22 * GB - 1, kTopK, kNumExperts) == 16);
    assert(ForwardRunner::resolve_slots(0, 30 * GB, kTopK, kNumExperts) == 32);
    assert(ForwardRunner::resolve_slots(0, 30 * GB - 1, kTopK, kNumExperts) == 24);
    std::cout << "  [PASS] Threshold boundaries are exact.\n";

    // A zero RAM reading means "no device to ask". It must assume the LOW tier -- guessing
    // high would commit a big pool on a machine that cannot hold it.
    assert(ForwardRunner::resolve_slots(0, 0, kTopK, kNumExperts) == 16);
    std::cout << "  [PASS] Unknown RAM assumes the low tier, not the high one.\n";

    // An explicit request wins over the ladder, in both directions.
    assert(ForwardRunner::resolve_slots(9,  64 * GB, kTopK, kNumExperts) == 9);
    assert(ForwardRunner::resolve_slots(44, 8 * GB,  kTopK, kNumExperts) == 44);
    std::cout << "  [PASS] An explicit --slots overrides the RAM ladder.\n";

    // Clamped, not rejected: a pool bigger than the expert count can never fill, so
    // honouring it would only waste memory. Correcting that is lossless.
    assert(ForwardRunner::resolve_slots(129, 64 * GB, kTopK, kNumExperts) == kNumExperts);
    assert(ForwardRunner::resolve_slots(100000, 64 * GB, kTopK, kNumExperts) == kNumExperts);
    assert(ForwardRunner::resolve_slots(128, 64 * GB, kTopK, kNumExperts) == 128);
    std::cout << "  [PASS] Requests above num_experts clamp rather than fail.\n";

    // Hard error, not a clamp: the pool must strictly exceed the routed batch or
    // plan_experts has no unpinned slot to evict and eviction deadlocks. This is the one
    // bound that is NOT lossless to fix on the user's behalf.
    assert(resolve_throws(1, 64 * GB, kTopK, kNumExperts));
    assert(resolve_throws(8, 64 * GB, kTopK, kNumExperts));
    assert(!resolve_throws(9, 64 * GB, kTopK, kNumExperts));
    std::cout << "  [PASS] A pool at or below top_k_experts is rejected.\n";

    // The clamp must not be able to push a value below the floor: with a hypothetical
    // model whose expert count barely exceeds top_k, clamping still has to leave room to
    // evict. Guards against a future model config making these two rules collide.
    assert(ForwardRunner::resolve_slots(64, 64 * GB, /*top_k=*/8, /*num_experts=*/9) == 9);
    assert(resolve_throws(64, 64 * GB, /*top_k=*/8, /*num_experts=*/8));
    std::cout << "  [PASS] Clamping to num_experts cannot land below the eviction floor.\n";

    // Whatever the ladder resolves to must be representable in the descriptor heap --
    // this is the invariant that ties the two halves of this file together.
    for (uint64_t ram : {8 * GB, 16 * GB, 24 * GB, 32 * GB, 64 * GB, 128 * GB}) {
        const size_t s = ForwardRunner::resolve_slots(0, ram, kTopK, kNumExperts);
        assert(ComputePipelineManager::descriptors_for(s, kLayers) >=
               required_descriptors(s, kLayers));
    }
    std::cout << "  [PASS] Every auto-sized configuration fits its derived heap.\n";

    std::cout << "[TEST] All capacity checks passed.\n";
    return 0;
}
