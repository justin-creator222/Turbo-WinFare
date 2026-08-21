#include "gturbo/streamer.hpp"
#include <iostream>
#include <algorithm>

namespace gturbo {

ExpertStreamer::ExpertStreamer(std::shared_ptr<D3D12Context> ctx,
                               const std::string& layer_file_path,
                               size_t slot_count,
                               uint64_t expert_stride,
                               EvictionPolicy policy)
    : ctx_(ctx), layer_file_path_(layer_file_path), slot_count_(slot_count),
      expert_stride_(expert_stride), policy_(policy) {
}

ExpertStreamer::~ExpertStreamer() {
    // Drain anything still in flight before the OVERLAPPED structures die with the slots.
    for (auto& slot : slots_) {
        if (slot && slot->read_pending) {
            DWORD n = 0;
            GetOverlappedResult(file_handle_, &slot->ov, &n, TRUE);
            slot->read_pending = false;
        }
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
    for (auto& slot : slots_) {
        if (slot && slot->event) CloseHandle(slot->event);
    }
}

void ExpertStreamer::initialize() {
    std::lock_guard<std::mutex> guard(lock_);
    
    // Allocate UMA memory slots
    slots_.reserve(slot_count_);
    for (size_t i = 0; i < slot_count_; ++i) {
        auto slot = std::make_unique<ExpertSlot>();
        std::string name = "ExpertSlot_" + std::to_string(i);
        // Read-only: expert weights are only ever bound as SRVs (GemvInt4 takes the slot as
        // t0/t1/t2). So this one may fall back to an UPLOAD heap under memory pressure.
        slot->buffer = ctx_->create_uma_buffer(expert_stride_, name, /*needs_uav=*/false);
        
        void* ptr = nullptr;
        D3D12_RANGE read_range{0, 0};
        HRESULT hr = slot->buffer->Map(0, &read_range, &ptr);
        if (FAILED(hr)) {
            throw GTurboFormatError("Failed to map UMA buffer pointer for " + name);
        }
        slot->host_ptr = ptr;
        slot->expert_id = -1;
        slots_.push_back(std::move(slot));
    }

    // Buffered asynchronous I/O -- deliberately NOT FILE_FLAG_NO_BUFFERING.
    //
    // No-buffering bypasses the Windows cache manager, so every expert read hit the SSD and
    // the 12.9 GB expert set was never cached by the OS. The reference relies on exactly the
    // opposite: it opens plain O_RDONLY and treats the unified buffer cache as a free
    // second-chance layer above its own slot pool (docs/SYSTEM_DESIGN.md:168).
    //
    // FILE_FLAG_SEQUENTIAL_SCAN is deliberately absent -- these are random 3.3 MB reads
    // scattered across a 410 MB file, and that hint would evict the pages we want kept.
    std::wstring wpath(layer_file_path_.begin(), layer_file_path_.end());
    file_handle_ = CreateFileW(
        wpath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw GTurboFormatError("Failed to open expert layer file '" + layer_file_path_ +
                                "' (Win32 error " + std::to_string(GetLastError()) + ")");
    }

    // One event per slot, created once. These used to be created and destroyed per read --
    // 240 kernel-object create/destroy pairs per token.
    for (auto& slot : slots_) {
        slot->event = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (!slot->event) {
            throw GTurboFormatError("CreateEvent failed for an expert slot (Win32 error " +
                                    std::to_string(GetLastError()) + ")");
        }
    }
}

ExpertSlot* ExpertStreamer::find_or_evict_slot(int expert_id, bool& was_hit) {
    global_clock_++;
    was_hit = false;

    // Check for hit
    for (auto& slot : slots_) {
        if (slot->expert_id == expert_id) {
            slot->last_used_timestamp = global_clock_;
            slot->frequency++;
            total_hits_++;
            was_hit = true;
            return slot.get();
        }
    }

    // Miss
    total_misses_++;

    // Select victim slot using LFU / LRU heuristic. Pinned slots are never candidates:
    // they belong to the batch being assembled right now.
    ExpertSlot* victim = nullptr;
    for (auto& slot : slots_) {
        if (slot->expert_id == -1 && !slot->pinned) {
            victim = slot.get();
            break;
        }
    }

    if (!victim) {
        if (policy_ == EvictionPolicy::LRU) {
            uint64_t oldest = UINT64_MAX;
            for (auto& slot : slots_) {
                if (slot->pinned) continue;
                if (slot->last_used_timestamp < oldest) {
                    oldest = slot->last_used_timestamp;
                    victim = slot.get();
                }
            }
        } else { // LFU with recency as the tie-break
            uint32_t min_freq = UINT32_MAX;
            uint64_t oldest = UINT64_MAX;
            for (auto& slot : slots_) {
                if (slot->pinned) continue;
                if (slot->frequency < min_freq ||
                    (slot->frequency == min_freq && slot->last_used_timestamp < oldest)) {
                    min_freq = slot->frequency;
                    oldest = slot->last_used_timestamp;
                    victim = slot.get();
                }
            }
        }
    }

    if (!victim) {
        throw GTurboFormatError(
            "Expert cache has no evictable slot: every slot is pinned by the current batch. "
            "The slot pool must be larger than the number of experts routed per token.");
    }

    victim->expert_id = expert_id;
    victim->last_used_timestamp = global_clock_;
    victim->frequency = 1;
    return victim;
}

void ExpertStreamer::clear_cache() {
    std::lock_guard<std::mutex> guard(lock_);
    for (auto& slot : slots_) {
        slot->expert_id = -1;
        slot->frequency = 0;
        slot->last_used_timestamp = 0;
    }
    total_hits_ = 0;
    total_misses_ = 0;
}

// Starts an asynchronous read into the slot's UMA buffer and returns immediately.
// Paired with await_read. Splitting issue from wait is what lets a batch put all its
// misses in flight at once instead of serializing at queue depth 1.
void ExpertStreamer::issue_read(ExpertSlot* slot, uint64_t file_offset, size_t count) {
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw GTurboFormatError("Expert file '" + layer_file_path_ + "' is not open");
    }

    ResetEvent(slot->event);
    slot->ov = OVERLAPPED{};
    slot->ov.Offset = static_cast<DWORD>(file_offset & 0xFFFFFFFF);
    slot->ov.OffsetHigh = static_cast<DWORD>((file_offset >> 32) & 0xFFFFFFFF);
    slot->ov.hEvent = slot->event;

    DWORD bytes_read = 0;
    BOOL ok = ReadFile(file_handle_, slot->host_ptr, static_cast<DWORD>(count),
                       &bytes_read, &slot->ov);
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        throw GTurboFormatError("Expert read failed on '" + layer_file_path_ + "' at offset " +
                                std::to_string(file_offset) + " (Win32 error " +
                                std::to_string(GetLastError()) + ")");
    }
    // With buffered I/O a read served from the page cache often completes synchronously;
    // the event is still signalled, so await_read handles both cases identically.
    slot->read_pending = true;
}

void ExpertStreamer::await_read(ExpertSlot* slot, size_t count) {
    if (!slot->read_pending) return;

    DWORD bytes_read = 0;
    BOOL ok = GetOverlappedResult(file_handle_, &slot->ov, &bytes_read, TRUE);
    slot->read_pending = false;
    if (!ok) {
        throw GTurboFormatError("Expert read failed on '" + layer_file_path_ +
                                "' (Win32 error " + std::to_string(GetLastError()) + ")");
    }
    if (bytes_read != count) {
        throw GTurboFormatError("Short expert read on '" + layer_file_path_ + "': got " +
                                std::to_string(bytes_read) + " of " +
                                std::to_string(count) + " bytes");
    }

    // Count only successful I/O, and only actual reads -- cache hits are not I/O.
    total_io_calls_++;
    total_bytes_read_ += bytes_read;
}

ExpertStreamer::ExpertPlan ExpertStreamer::plan_experts(int layer,
                                                        const std::vector<int>& expert_ids) {
    if (expert_ids.size() >= slots_.size()) {
        throw GTurboFormatError(
            "Expert batch of " + std::to_string(expert_ids.size()) +
            " needs a slot pool strictly larger than itself, but the pool has " +
            std::to_string(slots_.size()) + " slots.");
    }

    ExpertPlan plan;
    plan.owner = this;
    plan.slots.resize(expert_ids.size(), nullptr);

    // Slot bookkeeping only -- the lock is never held across an I/O wait.
    std::lock_guard<std::mutex> guard(lock_);
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        bool was_hit = false;
        ExpertSlot* s = find_or_evict_slot(expert_ids[i], was_hit);
        s->layer_id = layer;
        s->pinned = true;
        plan.slots[i] = s;
        // A hit already holds this expert's bytes; re-reading it is pure waste.
        (was_hit ? plan.hits : plan.misses).push_back(i);
    }
    return plan;
}

void ExpertStreamer::fetch_misses(ExpertPlan& plan) {
    if (!plan.valid() || plan.misses.empty()) return;
    try {
        // Every miss goes in flight before any of them is waited on.
        for (size_t i : plan.misses) {
            ExpertSlot* s = plan.slots[i];
            issue_read(s, static_cast<uint64_t>(s->expert_id) * expert_stride_,
                       static_cast<size_t>(expert_stride_));
        }
        for (size_t i : plan.misses) {
            await_read(plan.slots[i], static_cast<size_t>(expert_stride_));
        }
    } catch (...) {
        // A failed read leaves stale bytes behind, so invalidate every miss rather than
        // letting a later lookup treat it as a hit.
        {
            std::lock_guard<std::mutex> guard(lock_);
            for (size_t i : plan.misses) {
                plan.slots[i]->expert_id = -1;
                plan.slots[i]->frequency = 0;
                plan.slots[i]->read_pending = false;
            }
        }
        release_plan(plan);
        throw;
    }
}

void ExpertStreamer::release_plan(ExpertPlan& plan) {
    if (!plan.valid()) return;
    std::lock_guard<std::mutex> guard(lock_);
    // Release only this plan's slots. Unpinning the whole pool happened to work while
    // exactly one plan was ever in flight, but it silently drops a pin this plan does not
    // own -- so the moment two plans overlap (queued requests, or double-buffered prefill
    // tiles) one plan's experts become evictable while it is still binding them.
    for (auto* slot : plan.slots) {
        if (slot) slot->pinned = false;
    }
    plan.owner = nullptr;
}

} // namespace gturbo
