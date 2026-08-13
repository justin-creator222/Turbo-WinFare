#include "gturbo/d3d12_context.hpp"
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#include <iostream>

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
}

ComPtr<ID3D12GraphicsCommandList> D3D12Context::reset_command_list() {
    Frame& f = frames_[frame_index_];
    frame_index_ = (frame_index_ + 1) % kFrameRing;

    // Only wait if THIS frame's previous submission is still running. With a ring of 4 that
    // is almost never true, so recording proceeds while earlier lists are still executing.
    wait_for_fence(f.fence_value);
    f.allocator->Reset();
    f.list->Reset(f.allocator.Get(), nullptr);
    return f.list;
}

uint64_t D3D12Context::submit_command_list(ID3D12GraphicsCommandList* list) {
    list->Close();
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
    command_queue_->Signal(fence_.Get(), val);
    return val;
}

void D3D12Context::wait_for_fence(uint64_t fence_value) {
    if (fence_->GetCompletedValue() < fence_value) {
        fence_->SetEventOnCompletion(fence_value, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

void D3D12Context::flush_gpu() {
    uint64_t val = signal_fence();
    wait_for_fence(val);
}

ComPtr<ID3D12Resource> D3D12Context::create_uma_buffer(uint64_t size_bytes, const std::string& name) {
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
    res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

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
    }

    if (!name.empty()) {
        std::wstring wname(name.begin(), name.end());
        resource->SetName(wname.c_str());
    }
    return resource;
}

ComPtr<ID3D12Resource> D3D12Context::create_gpu_buffer(uint64_t size_bytes, const std::string& name) {
    return create_uma_buffer(size_bytes, name);
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
