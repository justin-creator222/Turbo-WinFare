#include "gturbo/pipeline.hpp"
#include <windows.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <cstring>

namespace fs = std::filesystem;

namespace gturbo {

static fs::path get_executable_directory() {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(NULL, path, MAX_PATH);
    if (length > 0) {
        return fs::path(path).parent_path();
    }
#endif
    return fs::current_path();
}

static fs::path resolve_shader_path(const std::string& relative_path) {
    fs::path exe_dir = get_executable_directory();
    
    std::vector<fs::path> candidates = {
        exe_dir / relative_path,
        exe_dir / ".." / relative_path,
        exe_dir / ".." / ".." / relative_path,
        fs::current_path() / relative_path,
        fs::path(relative_path)
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
            return fs::canonical(candidate);
        }
    }

    return fs::path();
}

class DxcCompiler {
public:
    static DxcCompiler& instance() {
        static DxcCompiler inst;
        return inst;
    }

    bool is_available() const { return available_; }

    ComPtr<ID3DBlob> compile(const std::string& hlsl_source, const std::string& entry_point, const std::string& target_profile) {
        if (!available_) return nullptr;

        IDxcBlobEncoding* src_blob = nullptr;
        HRESULT hr = utils_->CreateBlobFromPinned(hlsl_source.data(), static_cast<UINT32>(hlsl_source.size()), DXC_CP_UTF8, &src_blob);
        if (FAILED(hr) || !src_blob) {
            throw GTurboFormatError("DXC CreateBlobFromPinned failed");
        }

        DxcBuffer src_buffer{};
        src_buffer.Ptr = src_blob->GetBufferPointer();
        src_buffer.Size = src_blob->GetBufferSize();
        src_buffer.Encoding = DXC_CP_UTF8;

        std::wstring w_entry(entry_point.begin(), entry_point.end());
        std::wstring w_target(target_profile.begin(), target_profile.end());

        LPCWSTR args[] = {
            L"-E", w_entry.c_str(),
            L"-T", w_target.c_str(),
            L"-O3",
            L"-enable-16bit-types"
        };

        ComPtr<IDxcResult> result;
        hr = compiler_->Compile(&src_buffer, args, 5, nullptr, IID_PPV_ARGS(&result));
        src_blob->Release();

        if (FAILED(hr) || !result) {
            throw GTurboFormatError("DXC Compile call failed");
        }

        ComPtr<IDxcBlobUtf8> errors = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        if (errors && errors->GetStringLength() > 0) {
            std::string err_str = errors->GetStringPointer();
            if (err_str.find("error:") != std::string::npos) {
                if (errors) errors->Release();
                throw GTurboFormatError("DXC HLSL Shader compilation error:\n" + err_str);
            }
        }

        HRESULT status;
        result->GetStatus(&status);
        if (FAILED(status)) {
            std::string err_str = (errors && errors->GetStringLength() > 0) ? errors->GetStringPointer() : "Unknown DXC failure";
            if (errors) errors->Release();
            throw GTurboFormatError("DXC Shader compilation failed:\n" + err_str);
        }
        if (errors) errors->Release();

        ComPtr<IDxcBlob> shader_dxc_blob;
        result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader_dxc_blob), nullptr);
        if (!shader_dxc_blob) {
            throw GTurboFormatError("DXC failed to produce shader object blob");
        }

        ComPtr<ID3DBlob> d3d_blob;
        hr = D3DCreateBlob(shader_dxc_blob->GetBufferSize(), &d3d_blob);
        if (FAILED(hr) || !d3d_blob) {
            throw GTurboFormatError("D3DCreateBlob failed to wrap DXC output blob");
        }
        std::memcpy(d3d_blob->GetBufferPointer(), shader_dxc_blob->GetBufferPointer(), shader_dxc_blob->GetBufferSize());
        return d3d_blob;
    }

private:
    DxcCompiler() {
        // Look for the DXC we vendor next to the executable first, then let the OS loader
        // search. Version-pinned browser install paths are deliberately not probed -- they
        // ship an unrelated DXC, drift on every browser update, and their absence used to
        // silently downgrade every shader to cs_5_0.
        fs::path exe_dir = get_executable_directory();
        std::vector<std::string> search_paths = {
            (exe_dir / "dxcompiler.dll").string(),
            (exe_dir / ".." / "dxcompiler.dll").lexically_normal().string(),
            "dxcompiler.dll"
        };

        for (const auto& p : search_paths) {
            h_dxc_ = LoadLibraryA(p.c_str());
            if (h_dxc_) break;
        }

        if (!h_dxc_) return;

        pfn_dxc_create_instance_ = (DxcCreateInstanceProc)GetProcAddress(h_dxc_, "DxcCreateInstance");
        if (!pfn_dxc_create_instance_) return;

        HRESULT hr1 = pfn_dxc_create_instance_(CLSID_DxcUtils, IID_PPV_ARGS(&utils_));
        HRESULT hr2 = pfn_dxc_create_instance_(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler_));

        if (SUCCEEDED(hr1) && SUCCEEDED(hr2) && utils_ && compiler_) {
            available_ = true;
        }
    }

    HMODULE h_dxc_ = nullptr;
    DxcCreateInstanceProc pfn_dxc_create_instance_ = nullptr;
    ComPtr<IDxcUtils> utils_;
    ComPtr<IDxcCompiler3> compiler_;
    bool available_ = false;
};

ComputePipelineManager::ComputePipelineManager(std::shared_ptr<D3D12Context> ctx)
    : ctx_(ctx) {}

ComputePipelineManager::~ComputePipelineManager() {}

// Distinct binding sets consumed by a full generation. See the comment on descriptors_for()
// in pipeline.hpp for where each term comes from. Kept as a separate helper so the test can
// assert the raw formula against the padded capacity.
// This is the COUNTED number of sets, carrying no safety margin -- all padding lives in
// descriptors_for(). Keeping the two separate is what lets max_slots_for_capacity() answer
// "what will actually run" without being needlessly pessimistic, and lets the test pin the
// historical 65536-descriptor boundary at exactly 44 slots.
static uint64_t binding_sets_for(size_t slots_per_layer, int num_layers) {
    const uint64_t L = static_cast<uint64_t>(num_layers < 0 ? 0 : num_layers);
    const uint64_t S = static_cast<uint64_t>(slots_per_layer);
    return 3ULL * L * S     // expert gate/up/down, one set per (layer, slot)
         + 3ULL * L         // k epilogue, v epilogue, Attention -- per-layer KV buffers
         + 20ULL;           // layer-invariant sets, counted from the dispatch graph
}

uint32_t ComputePipelineManager::descriptors_for(size_t slots_per_layer, int num_layers) {
    const uint64_t exact = binding_sets_for(slots_per_layer, num_layers) * 16ULL;
    // 25% headroom absorbs a modest amount of formula drift -- a new dispatch shape, an
    // extra scratch buffer -- without silently running out mid-generation.
    uint64_t padded = exact + exact / 4ULL;
    // Round up to a whole 4096 so the number reads sensibly in a diagnostic.
    padded = ((padded + 4095ULL) / 4096ULL) * 4096ULL;
    if (padded < kLegacyDescriptorCapacity) padded = kLegacyDescriptorCapacity;
    if (padded > kMaxShaderVisibleDescriptors) padded = kMaxShaderVisibleDescriptors;
    return static_cast<uint32_t>(padded);
}

size_t ComputePipelineManager::max_slots_for_capacity(uint32_t capacity, int num_layers) {
    if (num_layers <= 0) return 0;
    // Invert descriptors_for() by search rather than algebra: the padding, the 4096
    // rounding and the floor make a closed form easy to get subtly wrong, and this runs
    // once, only on the failure path.
    size_t best = 0;
    for (size_t s = 1; s <= 4096; ++s) {
        if (binding_sets_for(s, num_layers) * 16ULL <= static_cast<uint64_t>(capacity)) {
            best = s;
        } else {
            break;
        }
    }
    return best;
}

void ComputePipelineManager::create_root_signature(uint32_t descriptor_capacity) {
    heap_capacity_ = descriptor_capacity;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = heap_capacity_;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    HRESULT hr_heap = ctx_->device()->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap_));
    if (FAILED(hr_heap)) {
        // Report what the device would actually accept, and what that means in the units the
        // user controls (--slots), rather than a bare HRESULT. The engine never queried the
        // binding tier before, so the limit was pure guesswork.
        D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
        std::string tier = "unknown";
        if (SUCCEEDED(ctx_->device()->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts)))) {
            tier = std::to_string(static_cast<int>(opts.ResourceBindingTier));
        }
        throw GTurboFormatError(
            "Failed to create a shader-visible CBV/SRV/UAV descriptor heap of " +
            std::to_string(heap_capacity_) + " descriptors (resource binding tier " + tier +
            "). This capacity is derived from the expert slot count; lower --slots and "
            "retry.");
    }
    descriptor_handle_size_ = ctx_->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_ROOT_PARAMETER root_params[3]{};

    // Slot 0: Push Constants (up to 16 DWORDs = 64 bytes)
    root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_params[0].Constants.ShaderRegister = 0;
    root_params[0].Constants.RegisterSpace = 0;
    root_params[0].Constants.Num32BitValues = 16;
    root_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 1: SRVs (t0..t7)
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 8;
    srv_range.BaseShaderRegister = 0;
    srv_range.RegisterSpace = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[1].DescriptorTable.NumDescriptorRanges = 1;
    root_params[1].DescriptorTable.pDescriptorRanges = &srv_range;
    root_params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Slot 2: UAVs (u0..u7)
    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 8;
    uav_range.BaseShaderRegister = 0;
    uav_range.RegisterSpace = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    root_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[2].DescriptorTable.NumDescriptorRanges = 1;
    root_params[2].DescriptorTable.pDescriptorRanges = &uav_range;
    root_params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_sig_desc{};
    root_sig_desc.NumParameters = 3;
    root_sig_desc.pParameters = root_params;
    root_sig_desc.NumStaticSamplers = 0;
    root_sig_desc.pStaticSamplers = nullptr;
    root_sig_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized_sig;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized_sig, &error_blob);
    if (FAILED(hr)) {
        std::string err = error_blob ? (char*)error_blob->GetBufferPointer() : "Unknown error";
        throw GTurboFormatError("Failed to serialize root signature: " + err);
    }

    hr = ctx_->device()->CreateRootSignature(0, serialized_sig->GetBufferPointer(), serialized_sig->GetBufferSize(), IID_PPV_ARGS(&root_signature_));
    if (FAILED(hr)) {
        throw GTurboFormatError("Failed to create Root Signature");
    }
}

ComPtr<ID3DBlob> ComputePipelineManager::compile_hlsl(const std::string& raw_source, const std::string& entry_point) {
    // Shader Model 6.6 via DXC is a hard requirement. Every kernel here depends on wave
    // intrinsics (WaveActiveSum / WaveActiveMax / WaveReadLaneFirst) for its cross-lane
    // reductions. The old fallback compiled to cs_5_0 with those intrinsics #define'd to
    // identity, which silently reduced every wave reduction to a single lane's partial
    // value -- numerically wrong everywhere, with no diagnostic.
    if (!DxcCompiler::instance().is_available()) {
        throw GTurboFormatError(
            "DXC (dxcompiler.dll) not found. Shader Model 6.6 wave intrinsics are required; "
            "there is no cs_5_0 fallback. Run tools/download_toolchain.py to fetch "
            "dxcompiler.dll and dxil.dll next to the executable.");
    }

    std::string prepended_source =
        "#define float16_t min16float\n"
        "#define uint32_t uint\n"
        "#define uint32_t3 uint3\n"
        "#define uint4_t uint\n" + raw_source;

    return DxcCompiler::instance().compile(prepended_source, entry_point, "cs_6_6");
}

void ComputePipelineManager::initialize_pipelines(uint32_t descriptor_capacity) {
    create_root_signature(descriptor_capacity);

    // Only kernels that are actually implemented are listed. Missing entries surface as a
    // clear "no pipeline for kernel" error at dispatch rather than a silent no-op.
    std::map<KernelType, std::string> kernel_files = {
        {KernelType::EmbedLookup, "shaders/EmbedLookup.hlsl"},
        {KernelType::RMSNormK,    "shaders/RMSNormK.hlsl"},
        {KernelType::GemvInt4,    "shaders/GemvInt4.hlsl"},
        {KernelType::GemvInt8,    "shaders/GemvInt8.hlsl"},
        {KernelType::QKVEpilogue, "shaders/QKVEpilogue.hlsl"},
        {KernelType::Attention,   "shaders/Attention.hlsl"},
        {KernelType::RouterTopK,  "shaders/RouterTopK.hlsl"},
        {KernelType::GeGLU,       "shaders/GeGLU.hlsl"},
        {KernelType::PostAttn,    "shaders/PostAttn.hlsl"},
        {KernelType::LayerTail,   "shaders/LayerTail.hlsl"},
        {KernelType::Softcap,     "shaders/Softcap.hlsl"},
        {KernelType::ScaleAccum,  "shaders/ScaleAccum.hlsl"},
        {KernelType::MulBF16,     "shaders/MulBF16.hlsl"},
        {KernelType::LMHeadGreedy, "shaders/LMHeadGreedy.hlsl"},
        {KernelType::ArgmaxReduce, "shaders/ArgmaxReduce.hlsl"},
    };

    // HLSL #include is resolved here by textual substitution rather than through a DXC
    // include handler -- one shared header, one substitution, no search-path surprises.
    std::string common;
    {
        fs::path common_path = resolve_shader_path("shaders/Common.hlsli");
        if (common_path.empty()) {
            throw GTurboFormatError("Failed to find shaders/Common.hlsli");
        }
        std::ifstream cf(common_path, std::ios::binary);
        std::stringstream cs;
        cs << cf.rdbuf();
        common = cs.str();
    }

    for (const auto& [type, relative_file_path] : kernel_files) {
        fs::path resolved_path = resolve_shader_path(relative_file_path);
        if (resolved_path.empty()) {
            throw GTurboFormatError("Failed to open shader file: " + relative_file_path +
                                   " (searched executable and working directories)");
        }

        std::ifstream file(resolved_path, std::ios::binary);
        if (!file.is_open()) {
            throw GTurboFormatError("Failed to open shader file at resolved path: " + resolved_path.string());
        }

        std::stringstream ss;
        ss << file.rdbuf();
        std::string raw_source = ss.str();

        // Strip a UTF-8 BOM if present. DXC rejects it with a confusing "non-ASCII
        // characters are not allowed" error pointing at the wrong line, and Windows editors
        // (and PowerShell's Set-Content -Encoding utf8) add one silently.
        if (raw_source.size() >= 3 &&
            static_cast<unsigned char>(raw_source[0]) == 0xEF &&
            static_cast<unsigned char>(raw_source[1]) == 0xBB &&
            static_cast<unsigned char>(raw_source[2]) == 0xBF) {
            raw_source.erase(0, 3);
        }

        const std::string directive = "#include \"Common.hlsli\"";
        size_t at = raw_source.find(directive);
        if (at != std::string::npos) {
            raw_source.replace(at, directive.size(), common);
        }

        auto shader_blob = compile_hlsl(raw_source);
        if (!shader_blob) {
            throw GTurboFormatError("Failed to compile shader: " + relative_file_path);
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_signature_.Get();
        pso_desc.CS = {shader_blob->GetBufferPointer(), shader_blob->GetBufferSize()};

        ComPtr<ID3D12PipelineState> pipeline;
        HRESULT hr = ctx_->device()->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline));
        if (FAILED(hr)) {
            throw GTurboFormatError("Failed to create compute pipeline state for shader: " + relative_file_path);
        }

        pipelines_[type] = pipeline;
    }
}

ComPtr<ID3D12PipelineState> ComputePipelineManager::get_pipeline(KernelType type) const {
    auto it = pipelines_.find(type);
    if (it != pipelines_.end()) {
        return it->second;
    }
    return nullptr;
}

void ComputePipelineManager::dispatch(ID3D12GraphicsCommandList* cmd_list,
                                       KernelType type,
                                       const std::vector<ID3D12Resource*>& srvs,
                                       const std::vector<ID3D12Resource*>& uavs,
                                       const KernelDispatchParams& params) {
    auto pipeline = get_pipeline(type);
    if (!pipeline) {
        // Silently returning here used to make an unimplemented kernel look like a
        // successful no-op dispatch.
        throw GTurboFormatError("No compute pipeline registered for kernel type " +
                                std::to_string(static_cast<int>(type)));
    }

    cmd_list->SetComputeRootSignature(root_signature_.Get());
    cmd_list->SetPipelineState(pipeline.Get());
    cmd_list->SetComputeRoot32BitConstants(
        0, static_cast<UINT>(params.constants.size()), params.constants.data(), 0);

    // Each table is 8 descriptors wide in the root signature, so reserve the full width even
    // when fewer resources are bound. Leaving the tail uninitialized let shaders that
    // declared t4/t5 read whatever descriptor happened to be resident.
    constexpr uint32_t SRV_SLOTS = 8;
    constexpr uint32_t UAV_SLOTS = 8;

    ID3D12DescriptorHeap* heaps[] = {descriptor_heap_.Get()};
    cmd_list->SetDescriptorHeaps(1, heaps);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_start = descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_start = descriptor_heap_->GetGPUDescriptorHandleForHeapStart();

    TableKey key{};
    for (size_t i = 0; i < SRV_SLOTS && i < srvs.size(); ++i) key.res[i] = srvs[i];
    for (size_t i = 0; i < UAV_SLOTS && i < uavs.size(); ++i) key.res[8 + i] = uavs[i];

    uint32_t srv_base, uav_base;
    auto cached = table_cache_.find(key);
    if (cached != table_cache_.end()) {
        srv_base = cached->second.srv_base;
        uav_base = cached->second.uav_base;
        ++table_hits_;
    } else {
        if (current_heap_offset_ + SRV_SLOTS + UAV_SLOTS > heap_capacity_) {
            // Unreachable by construction: the capacity is derived from the resolved slot
            // count via descriptors_for(), with headroom. If it ever fires, the formula has
            // drifted from what the dispatch graph actually binds -- so say by how much,
            // rather than "raise heap_capacity_", which is no longer where the number
            // comes from.
            throw GTurboFormatError(
                "Descriptor heap exhausted after " + std::to_string(table_cache_.size()) +
                " distinct binding sets (" + std::to_string(current_heap_offset_) + " of " +
                std::to_string(heap_capacity_) + " descriptors used). The capacity is "
                "derived from the expert slot count by "
                "ComputePipelineManager::descriptors_for(); this means the dispatch graph "
                "now binds more distinct resource sets than that formula predicts.");
        }

        auto handle_at = [&](uint32_t slot) {
            D3D12_CPU_DESCRIPTOR_HANDLE h = {cpu_start.ptr +
                static_cast<SIZE_T>(slot) * descriptor_handle_size_};
            return h;
        };

        // --- SRVs: raw byte-address views ---------------------------------
        srv_base = current_heap_offset_;
        for (uint32_t i = 0; i < SRV_SLOTS; ++i) {
            ID3D12Resource* res = key.res[i];
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = res ? static_cast<UINT>(res->GetDesc().Width / 4) : 1;
            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            ctx_->device()->CreateShaderResourceView(res, &desc, handle_at(srv_base + i));
        }
        current_heap_offset_ += SRV_SLOTS;

        // --- UAVs ---------------------------------------------------------
        uav_base = current_heap_offset_;
        for (uint32_t i = 0; i < UAV_SLOTS; ++i) {
            ID3D12Resource* res = key.res[8 + i];
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = res ? static_cast<UINT>(res->GetDesc().Width / 4) : 1;
            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            ctx_->device()->CreateUnorderedAccessView(res, nullptr, &desc, handle_at(uav_base + i));
        }
        current_heap_offset_ += UAV_SLOTS;

        TableEntry entry;
        entry.srv_base = srv_base;
        entry.uav_base = uav_base;
        for (size_t i = 0; i < 16; ++i) entry.keep_alive[i] = key.res[i];
        table_cache_.emplace(key, std::move(entry));
        ++table_misses_;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = {gpu_start.ptr +
        static_cast<UINT64>(srv_base) * descriptor_handle_size_};
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = {gpu_start.ptr +
        static_cast<UINT64>(uav_base) * descriptor_handle_size_};
    cmd_list->SetComputeRootDescriptorTable(1, srv_gpu);
    cmd_list->SetComputeRootDescriptorTable(2, uav_gpu);

    cmd_list->Dispatch(params.grid_x, params.grid_y, params.grid_z);
}

void ComputePipelineManager::barrier(ID3D12GraphicsCommandList* cmd_list,
                                     const std::vector<ID3D12Resource*>& resources) {
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.reserve(resources.size());
    for (ID3D12Resource* r : resources) {
        if (!r) continue;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.UAV.pResource = r;
        barriers.push_back(b);
    }
    if (!barriers.empty()) {
        cmd_list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }
}

} // namespace gturbo
