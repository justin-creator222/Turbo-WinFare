#pragma once

#include "gturbo/d3d12_context.hpp"
#include <d3d12.h>
#include <wrl/client.h>
#include <array>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>

namespace gturbo {

// Kernels mirror the CPU reference one-for-one, so a GPU-vs-CPU diff localizes a failure to
// a single kernel instead of just "the logits are wrong".
enum class KernelType {
    EmbedLookup,   // dequant one embedding row, x sqrt(D)
    RMSNormK,      // x/rms, optional BF16 weight
    GemvInt4,      // affine INT4 matvec (q/k/v/o, gate/up/down, experts, LM head)
    GemvInt8,      // affine INT8 matvec (routers)
    QKVEpilogue,   // per-head q_norm/k_norm/V-no-scale + NeoX RoPE
    Attention,     // GQA with sliding-window masking
    RouterTopK,    // top-K select + softmax over the top-K only
    GeGLU,         // gelu_tanh(gate) * up
    PostAttn,      // residual + the three pre-FFN views
    LayerTail,     // the Gemma 4 decoder-layer tail
    Softcap,       // 30*tanh(z/30)
    ScaleAccum,    // weighted reduction of the routed experts into h2
    MulBF16,       // elementwise multiply by a BF16 vector and a constant
    LMHeadGreedy,  // fused tied-head GEMV + partial argmax (greedy path only)
    ArgmaxReduce   // reduces LMHeadGreedy summaries to one token id
};

struct KernelDispatchParams {
    uint32_t grid_x{1};
    uint32_t grid_y{1};
    uint32_t grid_z{1};
    // 16 root constants, read by the shaders as gp0..gp3.
    std::array<uint32_t, 16> constants{};

    // Sets the leading constants and zeroes the rest, so a kernel never reads a stale value
    // left over from a previous dispatch.
    void set(std::initializer_list<uint32_t> values) {
        constants.fill(0);
        size_t i = 0;
        for (uint32_t v : values) {
            if (i >= constants.size()) break;
            constants[i++] = v;
        }
    }
};

class ComputePipelineManager {
public:
    explicit ComputePipelineManager(std::shared_ptr<D3D12Context> ctx);
    ~ComputePipelineManager();

    // `descriptor_capacity` MUST come from descriptors_for() with the resolved expert slot
    // count. It is a required argument, not a default, because the capacity depends on a
    // value (expert_slots_per_layer_) that ForwardRunner::initialize() used to resolve
    // *after* building this object -- so the dependency is enforced by the compiler rather
    // than by a comment someone can reorder past.
    void initialize_pipelines(uint32_t descriptor_capacity);

    // Descriptor budget for a whole generation.
    //
    // Every dispatch whose exact resource set has not been seen before burns 16 descriptors
    // (8 SRV + 8 UAV, see dispatch()), and tables are NEVER recycled -- so this is a
    // cumulative total for the process, not a per-token working set. Distinct binding sets:
    //
    //   3 per (layer, expert slot)  -- expert_gemv gate/up/down, src/runner.cpp
    //   3 per layer                 -- k epilogue, v epilogue, Attention (per-layer KV bufs)
    //   <= 64 layer-invariant       -- embed, norms, router, GeGLU, PostAttn, LayerTail, head
    //
    // The expert term dominates: at 30 layers it is 90 sets per slot. A hardcoded 65536
    // capped `--slots` at 44, and because tables are created lazily as slots are touched,
    // breaching it threw MID-GENERATION rather than at startup.
    static uint32_t descriptors_for(size_t slots_per_layer, int num_layers);

    // Largest slot count whose descriptor budget fits in `capacity` -- the inverse of
    // descriptors_for(), used to tell the user what they can actually ask for.
    static size_t max_slots_for_capacity(uint32_t capacity, int num_layers);

    // A shader-visible CBV/SRV/UAV heap is capped at 1,000,000 descriptors on resource
    // binding tier 1 and 2; tier 3 is limited only by memory.
    static constexpr uint32_t kMaxShaderVisibleDescriptors = 1000000;
    // Kept as the floor so small configurations, and the tests that pass it explicitly,
    // allocate exactly what they always did.
    static constexpr uint32_t kLegacyDescriptorCapacity = 65536;

    ComPtr<ID3D12RootSignature> root_signature() const { return root_signature_; }
    ComPtr<ID3D12PipelineState> get_pipeline(KernelType type) const;

    // Binds every resource as a RAW byte-address buffer. The shaders index them by explicit
    // byte offset, so there is no structured-buffer stride to disagree about.
    void dispatch(ID3D12GraphicsCommandList* cmd_list,
                  KernelType type,
                  const std::vector<ID3D12Resource*>& srvs,
                  const std::vector<ID3D12Resource*>& uavs,
                  const KernelDispatchParams& params);

    // Inserts UAV barriers so a dependent dispatch cannot start before its inputs are
    // written. produce_token previously had none at all.
    static void barrier(ID3D12GraphicsCommandList* cmd_list,
                        const std::vector<ID3D12Resource*>& resources);

    // Descriptor heap occupancy. Tables are cached for the heap's lifetime and never
    // recycled, so `descriptors_used` only ever grows -- it is the number to watch if a
    // future change starts binding per-slot resources the descriptors_for() formula does
    // not account for. Surfaced in /api/telemetry so that drift is visible before it is
    // fatal.
    uint32_t descriptors_used() const { return current_heap_offset_; }
    uint32_t descriptor_capacity() const { return heap_capacity_; }
    size_t distinct_binding_sets() const { return table_cache_.size(); }

    double descriptor_cache_hit_rate() const {
        const uint64_t total = table_hits_ + table_misses_;
        return total ? 100.0 * static_cast<double>(table_hits_) / static_cast<double>(total) : 0.0;
    }

private:
    void create_root_signature(uint32_t descriptor_capacity);
    ComPtr<ID3DBlob> compile_hlsl(const std::string& hlsl_source,
                                  const std::string& entry_point = "main");

    std::shared_ptr<D3D12Context> ctx_;
    ComPtr<ID3D12RootSignature> root_signature_;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
    UINT descriptor_handle_size_{0};
    uint32_t current_heap_offset_{0};
    uint32_t heap_capacity_{0};
    std::map<KernelType, ComPtr<ID3D12PipelineState>> pipelines_;

    // Descriptor tables keyed by the exact set of bound resources.
    //
    // Every dispatch used to create 16 fresh views -- 28,448 driver calls per token -- always
    // describing the same long-lived buffers. The views are pure functions of the resource
    // set, so once written they stay valid for the heap's lifetime and are simply reused.
    struct TableKey {
        std::array<ID3D12Resource*, 16> res{};
        bool operator==(const TableKey& o) const { return res == o.res; }
    };
    struct TableKeyHash {
        size_t operator()(const TableKey& k) const {
            size_t h = 1469598103934665603ULL;
            for (auto* p : k.res) {
                h = (h ^ reinterpret_cast<uintptr_t>(p)) * 1099511628211ULL;
            }
            return h;
        }
    };
    struct TableEntry {
        uint32_t srv_base{0};
        uint32_t uav_base{0};
        // Strong references to every bound resource.
        //
        // Without these the cache is keyed on raw pointers that the allocator will happily
        // reuse: release a buffer, allocate another, and a stale descriptor -- still carrying
        // the old NumElements -- gets served for the new one. That showed up as reads
        // returning 0 past the old bounds. Holding a reference makes each pointer unique for
        // the cache's lifetime, which is what the key assumes.
        std::array<ComPtr<ID3D12Resource>, 16> keep_alive;
    };
    std::unordered_map<TableKey, TableEntry, TableKeyHash> table_cache_;
    uint64_t table_hits_{0};
    uint64_t table_misses_{0};
};

} // namespace gturbo
