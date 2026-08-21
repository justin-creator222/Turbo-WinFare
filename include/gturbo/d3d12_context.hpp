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
};

} // namespace gturbo
