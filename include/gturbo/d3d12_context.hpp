#pragma once

#include "gturbo/format.hpp"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <memory>

namespace gturbo {

using Microsoft::WRL::ComPtr;

class D3D12Context {
public:
    D3D12Context();
    ~D3D12Context();

    void initialize(bool enable_debug_layer = false);

    ComPtr<ID3D12Device> device() const { return device_; }
    ComPtr<ID3D12CommandQueue> command_queue() const { return command_queue_; }
    
    // Command List reuse
    // Hands back the next command list in the ring, waiting only if that particular list is
    // still executing. A single allocator forced a full GPU drain before every recording,
    // which structurally prevented any CPU/GPU overlap.
    ComPtr<ID3D12GraphicsCommandList> reset_command_list();

    // Submits the current list without blocking. Returns the fence value to wait on later.
    uint64_t submit_command_list(ID3D12GraphicsCommandList* list);

    // Synchronization
    uint64_t signal_fence();
    void wait_for_fence(uint64_t fence_value);
    void flush_gpu();

    // True while the device is healthy. A removed device does NOT stop this engine on its
    // own: GetCompletedValue() starts returning UINT64_MAX so every fence wait returns
    // instantly, the router-index readback still maps and hands back stale bytes so the same
    // eight experts look like cache hits every layer, and the greedy token readback returns
    // whatever was there before. The result is a large tokens/sec number attached to garbage
    // output -- a failure that reads as a performance win. Probe explicitly instead.
    bool device_ok() const;

    // Human-readable device-removal reason, or an empty string when the device is fine.
    std::string device_removed_reason() const;

    // Memory Allocation Helper for UMA Host-Coherent RAM.
    //
    // `needs_uav` must be true for any buffer a shader will WRITE. It defaults to true
    // because that is the overwhelming majority here and because the failure mode of
    // getting it wrong is silent.
    //
    // When the host-coherent CUSTOM/L0 allocation fails -- which is exactly what memory
    // pressure from a large expert slot pool provokes -- there is a fallback onto an UPLOAD
    // heap. An UPLOAD-heap resource cannot carry ALLOW_UNORDERED_ACCESS, so creating a UAV
    // over one is an invalid call that, with the debug layer off, yields an undefined
    // descriptor and garbage writes rather than an error. So the fallback is offered ONLY
    // for read-only buffers; a UAV-capable request fails loudly instead of downgrading.
    ComPtr<ID3D12Resource> create_uma_buffer(uint64_t size_bytes, const std::string& name = "",
                                             bool needs_uav = true);

    // Number of UMA allocations that had to fall back to an UPLOAD heap. Non-zero means the
    // host-coherent budget is under pressure; surfaced in /api/telemetry.
    uint32_t uma_fallback_count() const { return uma_fallback_count_; }

    std::string adapter_name() const { return adapter_name_; }
    uint64_t dedicated_video_memory() const { return dedicated_vram_; }
    uint64_t shared_system_memory() const { return shared_ram_; }

    struct SystemMemoryInfo {
        uint64_t process_working_set_bytes{0};
        uint64_t process_private_bytes{0};
        uint64_t total_system_ram_bytes{0};
        uint64_t avail_system_ram_bytes{0};
        uint64_t gpu_dedicated_vram_used{0};
        uint64_t gpu_shared_ram_used{0};
    };

    SystemMemoryInfo query_memory_info() const;

private:
    // Throws a GTurboFormatError naming `where` when `hr` failed, folding in the device
    // removal reason when there is one. Every D3D12 call that returns an HRESULT on the
    // submit path goes through this -- Close(), Signal(), SetEventOnCompletion() and both
    // Reset() calls used to discard theirs.
    void throw_on_hr(HRESULT hr, const char* where) const;

    // A single fence wait is one layer's dispatches; nothing here legitimately takes a
    // minute. Waiting INFINITE meant a hung or removed device hung the process with it,
    // leaving taskkill as the only way out.
    static constexpr unsigned long kFenceWaitTimeoutMs = 60000;

    ComPtr<IDXGIFactory4> factory_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> command_queue_;
    // A ring of allocator/list pairs. Each remembers the fence value of its last submission
    // so it is only waited on when genuinely still in flight.
    static constexpr size_t kFrameRing = 4;
    struct Frame {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        uint64_t fence_value{0};
    };
    Frame frames_[kFrameRing];
    size_t frame_index_{0};
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_{nullptr};
    uint64_t fence_value_{0};

    std::string adapter_name_;
    uint64_t dedicated_vram_{0};
    uint64_t shared_ram_{0};
    // How many buffers fell back from the host-coherent CUSTOM/L0 heap to an UPLOAD heap.
    // The fallback used to be completely silent, which is why it could be named as a suspect
    // in the unexplained mid-benchmark crash and neither confirmed nor ruled out.
    uint32_t uma_fallback_count_{0};
};

} // namespace gturbo
