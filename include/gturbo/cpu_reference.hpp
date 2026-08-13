#pragma once

// A plain scalar FP32 forward pass for Gemma 4 26B-A4B.
//
// This exists to be *correct*, not fast -- it is the ground truth the DirectCompute kernels
// get diffed against in Stage 3. It touches no D3D12 at all: resident weights are memory
// mapped, expert blocks are read from the layer files on demand, and every operation is a
// straightforward loop.
//
// Every constant here was verified against the reference implementation's Metal kernels
// (fused.metal, moe.metal) and the pinned checkpoint's config.json, not inferred. The ones
// that are easy to get subtly wrong are called out at their use sites.

#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/resident_index.hpp"
#include "gturbo/tokenizer.hpp"
#include "gturbo/sampling.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gturbo {

struct CpuGenerationOptions {
    int max_tokens{64};
    float temperature{0.0f};   // 0 = greedy
    float top_p{0.95f};
    int top_k{64};
    float repetition_penalty{1.0f};
    uint64_t seed{0};
    bool verbose{false};
    // When set, per-stage FP32 tensors for the first token are written here for the
    // Stage 3 GPU diff.
    std::string dump_dir;
};

// One logical weight from resident.bin: the quantized payload plus its scales/biases.
struct QuantTensor {
    const uint8_t* data{nullptr};      // packed weights (or BF16 payload when unquantized)
    const uint16_t* scales{nullptr};   // BF16, [rows][groups]
    const uint16_t* biases{nullptr};   // BF16, [rows][groups]
    uint32_t rows{0};
    uint32_t in_dim{0};                // unpacked input width
    uint32_t groups{0};                // in_dim / group_size
    int bits{4};
    bool quantized{false};

    bool valid() const { return data != nullptr; }
};

class CpuReference {
public:
    CpuReference(const GTurboManifestV1& manifest,
                 const PackedExpertsLayoutV1& layout,
                 const std::string& model_dir);
    ~CpuReference();

    void load();

    // Returns completion tokens only (never the prompt). Stops on any stop token; the stop
    // token itself is not emitted.
    std::vector<uint32_t> generate(const std::vector<uint32_t>& prompt_tokens,
                                   const CpuGenerationOptions& opts);

    const Tokenizer& tokenizer() const { return tokenizer_; }
    std::string generate_text(const std::string& prompt, const CpuGenerationOptions& opts);

private:
    GTurboManifestV1 manifest_;
    PackedExpertsLayoutV1 layout_;
    std::string model_dir_;
    Tokenizer tokenizer_;

    // Memory-mapped resident.bin.
    void* map_handle_{nullptr};
    void* file_handle_{nullptr};
    const uint8_t* resident_{nullptr};
    uint64_t resident_size_{0};
    std::unordered_map<std::string, ResidentIndexEntryV1> entries_;

    // Open handles to packed_experts/layer_NN.bin, plus a scratch block.
    std::vector<void*> layer_files_;
    std::vector<uint8_t> expert_block_;

    // KV cache. Stored FP32 at full history for every layer; sliding-window layers apply
    // their window as an attention mask rather than as a ring buffer. That costs memory but
    // removes a whole class of index bugs from the thing we are trusting as ground truth.
    struct LayerKV {
        std::vector<float> k;  // [pos][kv_heads * head_dim]
        std::vector<float> v;
    };
    std::vector<LayerKV> kv_;
    int position_{0};

    QuantTensor tensor(const std::string& name) const;
    QuantTensor layer_tensor(int layer, const std::string& suffix) const;
    const uint16_t* bf16_tensor(const std::string& name, uint32_t expect_len) const;

    void load_expert(int layer, int expert);
    QuantTensor expert_tensor(const std::string& name) const;

    void forward(uint32_t token, int pos, std::vector<float>& logits,
                 const std::string& dump_dir);
};

} // namespace gturbo
