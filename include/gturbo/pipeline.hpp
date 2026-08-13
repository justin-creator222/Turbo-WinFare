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

    void initialize_pipelines();

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

    // No-op now that descriptor tables are cached for the heap's lifetime; kept so callers
    // do not have to care. Tables are never recycled, so nothing can be overwritten while a
    // command list still references it.
    void reset_descriptor_ring() {}

    double descriptor_cache_hit_rate() const {
        const uint64_t total = table_hits_ + table_misses_;
        return total ? 100.0 * static_cast<double>(table_hits_) / static_cast<double>(total) : 0.0;
    }

private:
    void create_root_signature();
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
