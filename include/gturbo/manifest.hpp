#pragma once

#include "gturbo/format.hpp"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace gturbo {

struct ManifestFileV1 {
    uint64_t size{0};
    std::string sha256;
};

// Gemma 4 26B-A4B architecture. Every value here is verified against the pinned checkpoint's
// config.json (mlx-community/gemma-4-26b-a4b-it-4bit @ 0d77464e). These are defaults only --
// a loaded manifest always overrides them, and validate() rejects a manifest that disagrees.
//
// Note the asymmetry between sliding-window and full-attention layers, which is easy to miss:
// full layers use head_dim 512 with 2 KV heads (GQA 8:1) and rotate only 25% of their pairs,
// while SWA layers use head_dim 256 with 8 KV heads (GQA 2:1) and rotate everything.
struct ManifestArchV1 {
    int hidden_size{2816};
    int ffn_intermediate{2112};
    int moe_intermediate_size{704};
    int num_heads{16};
    int num_kv_heads{8};            // sliding-window layers
    int num_full_kv_heads{2};       // full-attention layers
    int head_dim{256};              // sliding-window layers
    int full_head_dim{512};         // full-attention layers ("global_head_dim")
    int vocab_size{262144};
    int sliding_window{1024};
    double final_logit_softcap{30.0};
    double rope_theta{10000.0};         // sliding-window layers
    double full_rope_theta{1000000.0};  // full-attention layers
    // Applies to full-attention layers only: 0.25 * 512 / 2 = 64 of 256 pairs are rotated.
    double partial_rotary_factor{0.25};
    double rms_norm_eps{1e-6};
    int num_layers{30};
    int num_experts{128};
    int top_k_experts{8};
    bool tie_word_embeddings{true};
    // Full-attention layers have no v_proj: V reuses the k_proj output, then takes a
    // no-scale norm and skips RoPE.
    bool attention_k_eq_v{true};
    std::string hidden_activation{"gelu_pytorch_tanh"};
    // 30 entries, 1 = full attention. For this model: layers 5, 11, 17, 23, 29.
    std::vector<int> full_attention_layer_mask;
};

// MLX affine quantization, as shipped in the checkpoint: 4-bit weights in group_size 64
// with BF16 scale and BF16 bias (routers are 8-bit). The repacker copies these values
// through unchanged -- there is no requantization step, and no "Q4_K_M" anywhere.
struct ManifestQuantSlotV1 {
    int weight_bits{4};
    std::string scheme{"mlx_affine"};
    std::string scale_type{"BF16"};
    std::string bias_type{"BF16"};
    int group_size{64};
};

struct ManifestQuantV1 {
    ManifestQuantSlotV1 embedding;
    ManifestQuantSlotV1 attention;
    ManifestQuantSlotV1 router;
    ManifestQuantSlotV1 shared_expert;
    ManifestQuantSlotV1 routed_expert;
};

struct GTurboManifestV1 {
    std::string magic{GTurboFormatV1::MAGIC};
    int version_major{GTurboFormatV1::VERSION_MAJOR};
    int version_minor{GTurboFormatV1::VERSION_MINOR};
    std::map<std::string, bool> flags;
    std::string model_id;
    std::optional<std::string> source_snapshot_hash;
    ManifestArchV1 arch;
    std::optional<ManifestQuantV1> quant;
    std::map<std::string, ManifestFileV1> files;
    int experts_per_layer{128};
    int num_layers{30};
    // One expert's nine sub-tensors (gate/up/down x weight/scales/biases) occupy
    // 3,345,408 B, rounded up to the 16 KB alignment. The previous default of 692,224 was
    // 4.85x too small, so any code computing a file offset from it read the wrong expert.
    uint64_t expert_stride{3358720};
    std::optional<int> bit_width_overrides_honored;

    void validate() const;
    std::string to_json_string() const;
    static GTurboManifestV1 from_json_string(const std::string& json_str);
};

// Reads <model_dir>/manifest.json and <model_dir>/packed_experts/layout.json.
// Throws GTurboFormatError if either is missing, malformed, or mutually inconsistent.
std::string read_text_file(const std::string& path);

// True when `dir` is a bundle that actually parses: manifest.json and
// packed_experts/layout.json both present, well-formed, and mutually consistent.
//
// Deliberately validates by PARSING rather than by checking that files exist. A stale
// placeholder bundle has both files and still cannot be loaded -- the retired repacker left
// one in build/, whose layout.json predates the expertBlock format -- so an existence check
// happily selects it over the real bundle.
bool bundle_loads(const std::string& dir);

// Resolves a bundle name or path to a directory that actually loads.
//
// An absolute path, or one containing a separator, is taken literally: second-guessing an
// explicit path would load a model the user never asked for. A bare name is searched for in
// the working directory, then next to the executable, then one level up -- that last one
// matters because the exe lives in build\ while the bundle sits at the repo root, and
// launching by double-click makes build\ the working directory.
//
// Returns the first candidate that parses. When none do, returns the input unchanged so the
// caller's error names the place the user most likely meant.
std::string resolve_bundle_path(const std::string& name_or_path);

// Every directory a bare bundle name is searched in, in priority order. Used by the model
// list so the GUI offers the same bundles the loader would actually resolve to.
std::vector<std::string> bundle_search_roots();

} // namespace gturbo
