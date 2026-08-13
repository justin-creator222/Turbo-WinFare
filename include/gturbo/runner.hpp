#pragma once

#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/resident_index.hpp"
#include "gturbo/d3d12_context.hpp"
#include "gturbo/pipeline.hpp"
#include "gturbo/streamer.hpp"
#include "gturbo/kv_cache.hpp"
#include "gturbo/tokenizer.hpp"
#include "gturbo/sampling.hpp"

#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <functional>

namespace gturbo {

// Sampling defaults match the reference (README.md:116-117): temperature 0.2, top-K 64,
// top-P 0.95. Truncation order there is Top-P over the full distribution, then Top-K over
// the survivors, then temperature on the final draw -- not the usual HuggingFace order.
struct GenerationOptions {
    int max_tokens{2000};
    float temperature{0.2f};
    float top_p{0.95f};
    int top_k{64};
    float repetition_penalty{1.0f};
    bool has_seed{false};
    uint64_t seed{0};
    bool verbose{true};
    std::string system_prompt;
    // Matched against the visible text as it streams, not against token ids. A stop string
    // spanning two tokens still matches, and is never partially emitted.
    std::vector<std::string> stop_strings;

    SamplingParams sampling() const {
        return SamplingParams{temperature, top_p, top_k, repetition_penalty, has_seed, seed};
    }
};

// Why generation ended. Previously this was inferred from the token list, which made a
// user-initiated stop indistinguishable from a broken forward pass -- see generate_text.
enum class StopReason {
    EndOfSequence,  // <eos>
    EndOfTurn,      // <turn|> or <|tool_response>
    StopString,     // reserved for the streaming stop matcher
    MaxTokens,      // ran out of budget
    Cancelled,      // stop_generation() was called
};

const char* stop_reason_name(StopReason reason);

// Progress from a running generation. One mechanism feeds the CLI, the HTTP server, and any
// future SSE stream, mirroring the reference's RawDecodeProgress.
struct StreamEvent {
    enum class Kind {
        Prefill,   // prompt token consumed; `done`/`total` are meaningful
        Token,     // a completion token; `delta` is the newly visible text (may be empty)
        Tail,      // final flush of anything the stop matcher was withholding
    };
    Kind kind{Kind::Token};
    int done{0};
    int total{0};
    int index{0};
    uint32_t id{0};
    std::string delta;
};

// Return false to cancel. Used by the HTTP layer to abandon a generation whose client has
// disconnected, so a dead connection does not keep the GPU busy.
using StreamCallback = std::function<bool(const StreamEvent&)>;

struct GenerationResult {
    std::string text;
    std::vector<uint32_t> tokens;
    StopReason reason{StopReason::MaxTokens};
    std::string matched_stop;       // set when reason == StopString
    int prompt_tokens{0};
    int completion_tokens{0};
    double ttft_ms{0.0};            // to the first completion token
};

struct PerformanceMetrics {
    double prefill_tokens_per_sec{0.0};
    double decode_tokens_per_sec{0.0};
    double total_time_ms{0.0};
    uint64_t total_io_bytes{0};
    uint64_t total_io_calls{0};

    // Per-phase breakdown, so an optimization can be attributed rather than guessed at.
    // Mirrors the reference's decode decomposition in docs/BENCHMARKS.md:38-52.
    double expert_io_ms{0.0};    // blocked in load_experts_batch
    double gpu_wait_ms{0.0};     // blocked in flush_gpu
    double lm_head_ms{0.0};      // final norm + head + softcap + argmax readback
    double cpu_other_ms{0.0};    // recording, readbacks, everything else
    uint64_t tokens_measured{0};
    uint64_t gpu_waits{0};       // fence stalls per token

    double per_token(double total) const {
        return tokens_measured ? total / static_cast<double>(tokens_measured) : 0.0;
    }
};

struct ModelMemoryUsage {
    uint64_t resident_weights_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t expert_cache_bytes{0};
    uint64_t total_model_bytes{0};
    double cache_hit_rate_pct{0.0};
};

class ForwardRunner {
public:
    ForwardRunner(std::shared_ptr<D3D12Context> ctx,
                  const GTurboManifestV1& manifest,
                  const PackedExpertsLayoutV1& layout,
                  const std::string& model_dir);
    ~ForwardRunner();

    void initialize();
    
    // Execute Forward Pass for Single Token.
    //
    // `history` is only consulted when a repetition penalty is active; pass the tokens
    // generated so far. The 3-argument form generates without one.
    uint32_t produce_token(uint32_t input_token, int position, const GenerationOptions& opts,
                           const std::vector<uint32_t>& history);
    uint32_t produce_token(uint32_t input_token, int position, const GenerationOptions& opts) {
        static const std::vector<uint32_t> kNoHistory;
        return produce_token(input_token, position, opts, kNoHistory);
    }

    // Run Full Generation Sequence for Tokens
    std::vector<uint32_t> generate(const std::vector<uint32_t>& prompt_tokens, const GenerationOptions& opts);

    // Streaming generation over already-encoded tokens. `on_event` may be empty.
    //
    // The pipeline order inside is detokenize -> stop-match, and it matters: a stop string is
    // matched against the visible text, so it cannot be triggered by text the caller never saw.
    GenerationResult generate_tokens(const std::vector<uint32_t>& prompt_tokens,
                                     const GenerationOptions& opts,
                                     const StreamCallback& on_event = {});

    // Run Full Text Generation (Tokenize -> Forward MoE Loop -> Detokenize)
    std::string generate_text(const std::string& prompt, const GenerationOptions& opts);
    GenerationResult generate_chat(const std::vector<Tokenizer::ChatMessage>& messages,
                                   const GenerationOptions& opts,
                                   const StreamCallback& on_event = {});

    PerformanceMetrics metrics() const { return metrics_; }
    ModelMemoryUsage get_memory_usage() const;

    void clear_expert_cache();
    // Applies to every streamer that is already open AND to every one opened later.
    // Streamers are created lazily in ensure_streamer_opened, so storing the policy is not
    // bookkeeping -- without it, any layer first touched after this call silently reverts.
    void set_eviction_policy(EvictionPolicy policy);
    EvictionPolicy eviction_policy() const { return eviction_policy_; }

    StopReason last_stop_reason() const { return last_stop_reason_; }

    // Slots per layer in the expert cache. 0 = auto-size from installed RAM at load time.
    // Must be set before initialize(); it decides the size of every streamer's slot pool.
    void set_expert_slots(size_t slots) { requested_slots_ = slots; }
    size_t expert_slots_per_layer() const { return expert_slots_per_layer_; }

    // Max context in tokens. 0 = auto-size from installed RAM. Must be set before
    // initialize(); capped by the attention kernel's staged-score span.
    void set_max_context(int tokens) { requested_context_ = tokens; }
    int max_context() const { return max_context_; }
    // Must match ATTN_MAX_SPAN in shaders/Attention.hlsl.
    static constexpr int kAttentionMaxSpan = 4096;
    void stop_generation() { stop_requested_ = true; }
    void reset_stop_flag() { stop_requested_ = false; }
    bool is_stop_requested() const { return stop_requested_.load(); }

    std::vector<int> last_active_experts() const;
    // Hits over hits+misses across every opened layer streamer. Weighted by actual request
    // counts rather than averaged per layer, so a rarely-used layer cannot skew it.
    double expert_cache_hit_rate() const;
    std::string model_dir() const { return model_dir_; }
    const GTurboManifestV1& manifest() const { return manifest_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }

    uint64_t get_tensor_offset(const std::string& name) const {
        auto it = resident_entries_.find(name);
        if (it != resident_entries_.end()) {
            return it->second.file_offset;
        }
        return 0;
    }

private:
    std::shared_ptr<D3D12Context> ctx_;
    GTurboManifestV1 manifest_;
    PackedExpertsLayoutV1 layout_;
    std::string model_dir_;
    Tokenizer tokenizer_;
    std::unordered_map<std::string, ResidentIndexEntryV1> resident_entries_;

    std::unique_ptr<ComputePipelineManager> pipeline_mgr_;
    std::unique_ptr<KVCacheManager> kv_cache_;
    std::vector<std::unique_ptr<ExpertStreamer>> streamers_;

    // Activation scratch. All FP32 -- the kernels are correctness-first; FP16 packing is a
    // later optimization that would make GPU-vs-CPU diffs ambiguous.
    ComPtr<ID3D12Resource> buf_hidden_;
    ComPtr<ID3D12Resource> buf_normed_;
    ComPtr<ID3D12Resource> buf_q_;          // q_proj output
    ComPtr<ID3D12Resource> buf_q2_;         // after q_norm + RoPE
    ComPtr<ID3D12Resource> buf_k_;
    ComPtr<ID3D12Resource> buf_v_;
    ComPtr<ID3D12Resource> buf_ctx_;        // attention output, pre o_proj
    ComPtr<ID3D12Resource> buf_attn_out_;
    ComPtr<ID3D12Resource> buf_dense_x_;    // -> shared expert
    ComPtr<ID3D12Resource> buf_routed_x_;   // -> routed experts
    ComPtr<ID3D12Resource> buf_router_x_;   // -> router (unscaled)
    ComPtr<ID3D12Resource> buf_router_in_;  // router_x * router.scale / sqrt(D)
    ComPtr<ID3D12Resource> buf_h1_;
    ComPtr<ID3D12Resource> buf_h2_;
    ComPtr<ID3D12Resource> buf_scratch_d_;  // LayerTail scratch / expert down output
    ComPtr<ID3D12Resource> buf_ffn_gate_;
    ComPtr<ID3D12Resource> buf_ffn_up_;
    ComPtr<ID3D12Resource> buf_ffn_act_;
    // Dedicated scratch for the shared expert. It is logically independent of the routed
    // experts, but sharing scratch created a false dependency that forced the GPU to run the
    // two branches strictly in sequence.
    ComPtr<ID3D12Resource> buf_shared_gate_;
    ComPtr<ID3D12Resource> buf_shared_up_;
    ComPtr<ID3D12Resource> buf_shared_act_;
    ComPtr<ID3D12Resource> buf_shared_out_;
    ComPtr<ID3D12Resource> buf_router_logits_;
    ComPtr<ID3D12Resource> buf_router_indices_;
    ComPtr<ID3D12Resource> buf_router_weights_;
    ComPtr<ID3D12Resource> buf_logits_;
    // Single token id written by ArgmaxReduce on the greedy path, so the only thing read
    // back per token is 4 bytes instead of a 1 MB logit vector.
    ComPtr<ID3D12Resource> buf_out_token_;
    ComPtr<ID3D12Resource> buf_resident_weights_;
    uint64_t resident_weights_bytes_{0};

    // KV cache lives in KVCacheManager, which owns the per-layer FP32 buffers and the ring
    // indexing (physical_slot). Sliding-window layers get exactly `sliding_window` slots
    // regardless of context length; only full-attention layers scale with max_context.
    int max_context_{0};                 // 0 = auto-size from RAM
    int requested_context_{0};           // 0 = auto
    size_t requested_slots_{0};          // 0 = auto
    size_t expert_slots_per_layer_{16};  // resolved during initialize()

    // Byte offset of a resident tensor's payload / scales / biases, and its row count.
    struct TensorRef {
        uint32_t data_off{0};
        uint32_t scale_off{0};
        uint32_t bias_off{0};
        uint32_t rows{0};
        uint32_t in_dim{0};
        bool quantized{false};
    };
    TensorRef resident_ref(const std::string& name) const;
    TensorRef layer_ref(int layer, const std::string& suffix) const;
    uint32_t bf16_off(const std::string& name) const;
    // Reads a single BF16 value (e.g. layer_scalar) on the CPU. The resident UMA buffer
    // stays mapped for its lifetime, so this is a plain pointer read.
    float read_bf16_scalar(const std::string& name) const;
    const uint8_t* resident_cpu_{nullptr};
    // Reused across tokens on the sampled path so the 1 MB logit vector is not reallocated
    // every step. Unused on the greedy path, which reads back 4 bytes.
    std::vector<float> logits_scratch_;

    PerformanceMetrics metrics_;
    std::atomic<bool> stop_requested_{false};
    StopReason last_stop_reason_{StopReason::MaxTokens};
    EvictionPolicy eviction_policy_{EvictionPolicy::LFU};
    std::vector<int> last_active_experts_;
    mutable std::mutex streamer_mutex_;

    ExpertStreamer* ensure_streamer_opened(int layer);
};

} // namespace gturbo
