#include "gturbo/d3d12_context.hpp"
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#include <iostream>
#include <cstdio>
#include <string>

namespace gturbo {

D3D12Context::D3D12Context() {}

D3D12Context::~D3D12Context() {
    flush_gpu();
    if (fence_event_) {
        CloseHandle(fence_event_);
    }
}

void D3D12Context::initialize(bool enable_debug_layer) {
    if (enable_debug_layer) {
        ComPtr<ID3D12Debug> debug_controller;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
            debug_controller->EnableDebugLayer();
        }
    }

    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory_));
    if (FAILED(hr)) {
        throw GTurboFormatError("Failed to create DXGI Factory");
    }

    // Pick first high-performance DirectX 12 hardware adapter (e.g. AMD Radeon 780M)
    for (UINT adapter_index = 0; factory_->EnumAdapters1(adapter_index, &adapter_) != DXGI_ERROR_NOT_FOUND; ++adapter_index) {
        DXGI_ADAPTER_DESC1 desc;
        adapter_->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        if (SUCCEEDED(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            std::wstring wname(desc.Description);
            adapter_name_ = std::string(wname.begin(), wname.end());
            dedicated_vram_ = desc.DedicatedVideoMemory;
            shared_ram_ = desc.SharedSystemMemory;
            break;
        }
    }

    if (!adapter_) {
        // Fallback to WARP software renderer if hardware DX12 device is unavailable
        factory_->EnumWarpAdapter(IID_PPV_ARGS(&adapter_));
        adapter_name_ = "WARP Software Adapter";
    }

    hr = D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_));
    if (FAILED(hr)) {
        throw GTurboFormatError("Failed to create D3D12 Device");
    }

    // Create DirectCompute Command Queue
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue_));
    if (FAILED(hr)) {
        throw GTurboFormatError("Failed to create Compute Command Queue");
    }

    for (size_t i = 0; i < kFrameRing; ++i) {
        hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&frames_[i].allocator));
        if (FAILED(hr)) {
            throw GTurboFormatError("Failed to create Command Allocator " + std::to_string(i));
        }
        hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        frames_[i].allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&frames_[i].list));
        if (FAILED(hr)) {
            throw GTurboFormatError("Failed to create Command List " + std::to_string(i));
        }
        frames_[i].list->Close();
    }

    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) {
        throw GTurboFormatError("Failed to create Fence");
    }

    fence_event_ = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!fence_event_) {
        // Not pedantry: WaitForSingleObject(NULL, ...) fails immediately instead of waiting,
        // so every fence wait would return at once and every readback would race the GPU.
        throw GTurboFormatError("Failed to create the GPU fence event (Win32 error " +
                                std::to_string(GetLastError()) + ")");
    }
}

std::string D3D12Context::device_removed_reason() const {
    if (!device_) return "no device";
    const HRESULT reason = device_->GetDeviceRemovedReason();
    switch (reason) {
        case S_OK:
            return std::string();
        case DXGI_ERROR_DEVICE_HUNG:
            return "device hung (a shader ran away, or the TDR watchdog fired)";
        case DXGI_ERROR_DEVICE_REMOVED:
            return "device removed (driver reset, or the adapter was reconfigured)";
        case DXGI_ERROR_DEVICE_RESET:
            return "device reset after an invalid command";
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
            return "driver internal error";
        case DXGI_ERROR_INVALID_CALL:
            return "invalid call";
        default: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(reason));
            return std::string("device removed, reason ") + buf;
        }
    }
}

bool D3D12Context::device_ok() const {
    return device_ && device_->GetDeviceRemovedReason() == S_OK;
}

void D3D12Context::throw_on_hr(HRESULT hr, const char* where) const {
    if (SUCCEEDED(hr)) return;

    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
    std::string msg = std::string(where) + " failed (HRESULT " + buf + ")";

    // The removal reason matters more than the HRESULT: once the device is gone every later
    // call fails identically, so the first failure and the hundredth are indistinguishable
    // without it. And the alternative to throwing is worse than a crash -- submits silently
    // do nothing, fences complete instantly, and the engine reports a fast generation made
    // entirely of stale tokens.
    const std::string reason = device_removed_reason();
    if (!reason.empty()) {
        msg += ": " + reason +
               ". On this APU the usual trigger is an over-greedy UMA allocation -- lower "
               "--slots or --context.";
    }
    throw GTurboFormatError(msg);
}

ComPtr<ID3D12GraphicsCommandList> D3D12Context::reset_command_list() {
    Frame& f = frames_[frame_index_];
    frame_index_ = (frame_index_ + 1) % kFrameRing;

    // Only wait if THIS frame's previous submission is still running. With a ring of 4 that
    // is almost never true, so recording proceeds while earlier lists are still executing.
    wait_for_fence(f.fence_value);
    // A failed Reset leaves the list CLOSED, and recording into a closed list is a silent
    // no-op until Close() reports it -- which was also unchecked. The layer would dispatch
    // nothing and the next kernel would read whatever the buffers held last token.
    throw_on_hr(f.allocator->Reset(), "ID3D12CommandAllocator::Reset");
    throw_on_hr(f.list->Reset(f.allocator.Get(), nullptr), "ID3D12GraphicsCommandList::Reset");
    return f.list;
}

uint64_t D3D12Context::submit_command_list(ID3D12GraphicsCommandList* list) {
    // ExecuteCommandLists returns void, so Close() is the only place a malformed or
    // device-lost command list can report itself before the work silently vanishes.
    throw_on_hr(list->Close(), "ID3D12GraphicsCommandList::Close");
    ID3D12CommandList* lists[] = {list};
    command_queue_->ExecuteCommandLists(1, lists);
    const uint64_t val = signal_fence();
    // Record the fence against whichever frame owns this list, so reset_command_list knows
    // when it is safe to recycle.
    for (auto& f : frames_) {
        if (f.list.Get() == list) { f.fence_value = val; break; }
    }
    return val;
}

uint64_t D3D12Context::signal_fence() {
    uint64_t val = ++fence_value_;
    // If this fails the fence never reaches `val`. wait_for_fence would then block on a
    // value that can no longer arrive -- or, on a removed device, return instantly and let
    // the caller read a buffer the GPU never wrote.
    throw_on_hr(command_queue_->Signal(fence_.Get(), val), "ID3D12CommandQueue::Signal");
    return val;
}

void D3D12Context::wait_for_fence(uint64_t fence_value) {
    if (fence_->GetCompletedValue() >= fence_value) return;

    throw_on_hr(fence_->SetEventOnCompletion(fence_value, fence_event_),
                "ID3D12Fence::SetEventOnCompletion");

    // Bounded, not INFINITE. A hung device used to hang this process with it, leaving
    // taskkill as the only way out of a stalled generation.
    const DWORD wait = WaitForSingleObject(fence_event_, kFenceWaitTimeoutMs);
    if (wait == WAIT_OBJECT_0) return;

    const std::string reason = device_removed_reason();
    throw GTurboFormatError(
        std::string("GPU fence wait ") +
        (wait == WAIT_TIMEOUT ? "timed out after 60 s" : "failed") +
        " at fence value " + std::to_string(fence_value) +
        (reason.empty()
             ? std::string(": the device still reports healthy, so this is a genuinely long "
                           "or stalled dispatch rather than a lost device")
             : ": " + reason));
}

void D3D12Context::flush_gpu() {
    uint64_t val = signal_fence();
    wait_for_fence(val);
}

ComPtr<ID3D12Resource> D3D12Context::create_uma_buffer(uint64_t size_bytes, const std::string& name,
                                                       bool needs_uav) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_CUSTOM;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;

    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment = 0;
    res_desc.Width = size_bytes;
    res_desc.Height = 1;
    res_desc.DepthOrArraySize = 1;
    res_desc.MipLevels = 1;
    res_desc.Format = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count = 1;
    res_desc.SampleDesc.Quality = 0;
    res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags = needs_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                               : D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> resource;
    HRESULT hr = device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &res_desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource)
    );

    if (FAILED(hr)) {
        // An UPLOAD heap cannot carry ALLOW_UNORDERED_ACCESS, so this fallback is only
        // sound for buffers that are read and never written by a shader. Downgrading a
        // UAV-capable request here used to produce a resource that CreateUnorderedAccessView
        // cannot legally describe -- undefined descriptor, garbage writes, no diagnostic.
        if (needs_uav) {
            throw GTurboFormatError(
                "Failed to allocate a host-coherent UMA buffer of " +
                std::to_string(size_bytes) + " bytes" +
                (name.empty() ? std::string() : " for '" + name + "'") +
                ". This buffer must be shader-writable, so there is no read-only fallback. "
                "The usual cause is memory pressure from a large expert slot pool -- lower "
                "--slots.");
        }

        D3D12_HEAP_PROPERTIES upload_props{};
        upload_props.Type = D3D12_HEAP_TYPE_UPLOAD;

        res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = device_->CreateCommittedResource(
            &upload_props,
            D3D12_HEAP_FLAG_NONE,
            &res_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&resource)
        );
        if (FAILED(hr)) {
            throw GTurboFormatError("Failed to allocate UMA buffer of size " + std::to_string(size_bytes));
        }

        // Say so. This fallback swaps host-coherent WRITE_BACK memory for write-combined
        // UPLOAD memory, which changes both the performance characteristics of every read
        // through it and how far the slot pool can over-commit before something unrelated
        // fails. It used to leave no trace whatsoever, which is why it could be named as a
        // suspect for the intermittent mid-benchmark crash and then neither confirmed nor
        // ruled out. Warn once, count always.
        if (++uma_fallback_count_ == 1) {
            std::cout << "      WARNING: host-coherent UMA allocation failed for '"
                      << (name.empty() ? std::string("<unnamed>") : name) << "' ("
                      << (size_bytes / (1024 * 1024)) << " MB); fell back to an UPLOAD heap. "
                         "The adapter's shared-memory budget is under pressure -- lower "
                         "--slots or --context.\n";
        }
    }

    if (!name.empty()) {
        std::wstring wname(name.begin(), name.end());
        resource->SetName(wname.c_str());
    }
    return resource;
}

D3D12Context::SystemMemoryInfo D3D12Context::query_memory_info() const {
    SystemMemoryInfo info{};

    // Process memory counters
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        info.process_working_set_bytes = static_cast<uint64_t>(pmc.WorkingSetSize);
        info.process_private_bytes = static_cast<uint64_t>(pmc.PrivateUsage);
    }

    // Global system physical RAM
    MEMORYSTATUSEX mem_status{};
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        info.total_system_ram_bytes = mem_status.ullTotalPhys;
        info.avail_system_ram_bytes = mem_status.ullAvailPhys;
    }

    // DXGI Video Memory Query
    if (adapter_) {
        ComPtr<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(adapter_.As(&adapter3))) {
            DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info))) {
                info.gpu_dedicated_vram_used = local_info.CurrentUsage;
            }
            DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info))) {
                info.gpu_shared_ram_used = non_local_info.CurrentUsage;
            }
        }
    }

    return info;
}

} // namespace gturbo
