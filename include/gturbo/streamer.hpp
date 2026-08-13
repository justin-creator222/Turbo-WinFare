#pragma once

#include "gturbo/format.hpp"
#include "gturbo/d3d12_context.hpp"
#include <windows.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

namespace gturbo {

enum class EvictionPolicy {
    LFU,
    LRU
};

struct ExpertSlot {
    int expert_id{-1};
    int layer_id{-1};
    ComPtr<ID3D12Resource> buffer; // UMA host-coherent RAM
    void* host_ptr{nullptr};
    uint64_t last_used_timestamp{0};
    uint32_t frequency{0};
    // Persistent per-slot event and OVERLAPPED, so a batch can have all its reads in flight
    // at once. Creating these per read cost 240 kernel-object create/destroy pairs per token.
    HANDLE event{nullptr};
    OVERLAPPED ov{};
    bool read_pending{false};
    // Set while a slot belongs to the batch currently being assembled, so a later expert in
    // the same batch cannot evict an earlier one. Without this, a batch of K experts against
    // K slots can overwrite its own entries and the caller ends up holding slot pointers
    // whose contents belong to a different expert.
    bool pinned{false};
};

class ExpertStreamer {
public:
    ExpertStreamer(std::shared_ptr<D3D12Context> ctx,
                   const std::string& layer_file_path,
                   size_t slot_count,
                   uint64_t expert_stride,
                   EvictionPolicy policy = EvictionPolicy::LFU);
    ~ExpertStreamer();

    void initialize();

    // Splitting the plan from the fetch is what makes hit-first dispatch possible: the
    // caller can queue GPU work for the experts already resident, and for the shared expert
    // which needs no expert data at all, while the misses are still in flight.
    struct ExpertPlan {
        std::vector<ExpertSlot*> slots;   // one per request, in request order
        std::vector<size_t> hits;         // indices into `slots` needing no I/O
        std::vector<size_t> misses;       // indices into `slots` awaiting a read
        ExpertStreamer* owner{nullptr};

        bool valid() const { return owner != nullptr; }
    };

    // Resolves the cache and pins every slot. Performs no I/O.
    ExpertPlan plan_experts(int layer, const std::vector<int>& expert_ids);
    // Issues every miss read at once, then waits for all of them.
    void fetch_misses(ExpertPlan& plan);
    // Unpins. Safe to call more than once; called automatically by fetch_misses on failure.
    void release_plan(ExpertPlan& plan);

    size_t slot_count() const { return slots_.size(); }
    uint64_t expert_stride() const { return expert_stride_; }
    uint64_t total_bytes_read() const { return total_bytes_read_.load(); }
    uint64_t total_io_calls() const { return total_io_calls_.load(); }
    uint64_t total_cache_memory_bytes() const { return slots_.size() * expert_stride_; }
    uint64_t total_hits() const { return total_hits_.load(); }
    uint64_t total_misses() const { return total_misses_.load(); }
    double hit_rate_pct() const {
        uint64_t h = total_hits_.load();
        uint64_t m = total_misses_.load();
        return (h + m == 0) ? 0.0 : (double)h / (double)(h + m) * 100.0;
    }

    void clear_cache();
    void set_eviction_policy(EvictionPolicy policy) { policy_ = policy; }
    EvictionPolicy eviction_policy() const { return policy_; }

private:
    std::shared_ptr<D3D12Context> ctx_;
    std::string layer_file_path_;
    size_t slot_count_;
    uint64_t expert_stride_;
    EvictionPolicy policy_;

    HANDLE file_handle_{INVALID_HANDLE_VALUE};

    std::vector<std::unique_ptr<ExpertSlot>> slots_;
    std::mutex lock_;
    uint64_t global_clock_{0};
    std::atomic<uint64_t> total_bytes_read_{0};
    std::atomic<uint64_t> total_io_calls_{0};
    std::atomic<uint64_t> total_hits_{0};
    std::atomic<uint64_t> total_misses_{0};

    // `was_hit` reports whether the slot already held this expert. The caller MUST skip the
    // read when it did -- this used to always read, so the cache tracked a hit rate it never
    // acted on and every byte was fetched twice over.
    ExpertSlot* find_or_evict_slot(int expert_id, bool& was_hit);
    void issue_read(ExpertSlot* slot, uint64_t file_offset, size_t count);
    void await_read(ExpertSlot* slot, size_t count);
};

} // namespace gturbo
