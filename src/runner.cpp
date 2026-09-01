// Copyright 2026 justin-creator222. Licensed under the Apache License, Version 2.0.
//
// The dispatch graph and forward-pass ordering implemented here are derived from
// TurboFieldfare (https://github.com/drumih/turbo-fieldfare), copyright Andrey Mikhaylov,
// Apache-2.0. See NOTICE in the repository root.

#include "gturbo/runner.hpp"
#include "gturbo/detokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>

namespace gturbo {

ForwardRunner::ForwardRunner(std::shared_ptr<D3D12Context> ctx,
                             const GTurboManifestV1& manifest,
                             const PackedExpertsLayoutV1& layout,
                             const std::string& model_dir)
    : ctx_(ctx), manifest_(manifest), layout_(layout), model_dir_(model_dir) {}

ForwardRunner::~ForwardRunner() {
    // Every buffer this object owns -- resident weights, KV cache, activations, expert slots
    // -- may still be referenced by a command list the GPU has not finished executing. The
    // D3D12Context, and therefore the command queue, is SHARED and outlives the runner, so
    // releasing those resources without draining first is a device removal.
    //
    // This belongs in the destructor rather than at each call site: the server reload path,
    // /api/unload_model and the C ABI's load/unload all destroy runners, and remembering the
    // flush at three of those and forgetting the fourth is exactly how this goes wrong.
    if (ctx_) {
        try {
            ctx_->flush_gpu();
        } catch (...) {
            // A destructor must not throw. A failed flush here means the device is already
            // lost, in which case there is nothing left to drain.
        }
    }
}

// Size the expert cache. Pure logic, deliberately free of D3D12 and of the model bundle so
// the ladder and its bounds can be tested directly -- reaching this through initialize()
// would require a tokenizer, a device, resident.bin and 30 layer files.
//
// Bigger is NOT automatically better: the slot pool competes with the OS page cache for the
// same physical memory, which is why the ladder stops well short of the 128 experts a layer
// has. Raising it is an opt-in, not a default.
size_t ForwardRunner::resolve_slots(size_t requested, uint64_t total_ram_bytes,
                                    int top_k_experts, int num_experts) {
    size_t slots;
    if (requested > 0) {
        slots = requested;
    } else {
        // A zero here means "no device to ask"; assume the low tier rather than the high one.
        const double ram_gb = total_ram_bytes > 0
            ? static_cast<double>(total_ram_bytes) / (1024.0 * 1024.0 * 1024.0)
            : 16.0;
        if      (ram_gb >= 30.0) slots = 32;
        else if (ram_gb >= 22.0) slots = 24;
        else                     slots = 16;
    }

    // A pool larger than the expert count can never fill. Lossless to clamp.
    if (num_experts > 0 && slots > static_cast<size_t>(num_experts)) {
        slots = static_cast<size_t>(num_experts);
    }

    // Hard error, not a clamp: the pool must strictly exceed the routed batch so
    // plan_experts always has an unpinned slot to evict. Below that, eviction deadlocks.
    const size_t min_slots = static_cast<size_t>(top_k_experts) + 1;
    if (slots < min_slots) {
        throw GTurboFormatError(
            "Expert slot pool of " + std::to_string(slots) +
            " must exceed top_k_experts (" + std::to_string(top_k_experts) + "). Use at "
            "least --slots " + std::to_string(min_slots) + ".");
    }
    return slots;
}

void ForwardRunner::initialize() {
    // The tokenizer is a hard dependency: the reference bundle ships tokenizer/tokenizer.json
    // alongside the weights. There is no synthesized-vocabulary fallback.
    const std::string candidates[] = {
        model_dir_ + "/tokenizer/tokenizer.json",
        model_dir_ + "/tokenizer.json"
    };
    std::string vocab_path;
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            vocab_path = candidate;
            break;
        }
    }
    if (vocab_path.empty()) {
        throw GTurboFormatError(
            "No tokenizer found in model bundle '" + model_dir_ +
            "': expected tokenizer/tokenizer.json. Re-run tools/convert_hf_to_gturbo.py to "
            "produce a complete bundle.");
    }
    tokenizer_.load_vocabulary(vocab_path);

    const auto& arch = manifest_.arch;

    // Resolve the expert cache size FIRST -- before the pipeline manager exists.
    //
    // The descriptor heap capacity is a function of this number (see
    // ComputePipelineManager::descriptors_for), so resolving it afterwards is what capped
    // --slots at 44 and made the breach surface mid-generation. initialize_pipelines() now
    // takes the capacity as a required argument, so this ordering cannot be reversed
    // without a compile error.
    const D3D12Context::SystemMemoryInfo mem =
        ctx_ ? ctx_->query_memory_info() : D3D12Context::SystemMemoryInfo{};
    expert_slots_per_layer_ = resolve_slots(requested_slots_, mem.total_system_ram_bytes,
                                            arch.top_k_experts, arch.num_experts);

    if (requested_slots_ > static_cast<size_t>(arch.num_experts)) {
        // Clamped rather than rejected: a pool larger than the expert count can never fill,
        // so honouring it would only waste memory. That is lossless to correct on the user's
        // behalf, unlike the top_k floor below, which deadlocks eviction and stays an error.
        std::cout << "      Note: --slots " << requested_slots_ << " exceeds the "
                  << arch.num_experts << " experts per layer; using "
                  << expert_slots_per_layer_ << ".\n";
    }

    const uint64_t pool_bytes = static_cast<uint64_t>(expert_slots_per_layer_) *
                                manifest_.expert_stride *
                                static_cast<uint64_t>(arch.num_layers);
    std::cout << "      Expert cache: " << expert_slots_per_layer_ << " slots/layer ("
              << (pool_bytes / (1024 * 1024))
              << " MB if every layer opens)\n";

    // Warn, do not block. Streamers open lazily so the pool is never committed all at once,
    // and an explicit --slots is an opt-in -- but there are two distinct ceilings here and
    // neither announces itself clearly when breached.
    //
    // 1. Available system RAM. The slot pool competes with the OS page cache for the same
    //    physical memory, and the streamer leans on that cache deliberately. The reference
    //    measured 32 slots collapsing their 8 GB host from 5.6 to 1.578 tok/s with I/O
    //    rising to 318 ms/token -- so overshooting here makes things slower, not faster.
    //
    // 2. The adapter's shared-system-memory budget, which is the real cap on UMA
    //    allocation and is typically about half of installed RAM. Overshooting THIS does
    //    not degrade, it fails -- and it surfaces far downstream as an unrelated-looking
    //    "Failed to map ..." rather than as an allocation error.
    const uint64_t shared_budget = ctx_ ? ctx_->shared_system_memory() : 0;
    // Everything else the engine commits to UMA: resident weights, KV cache, activations.
    // Approximate, and deliberately on the low side so this warns rather than nags.
    constexpr uint64_t kOtherUmaBytes = 2ULL * 1024 * 1024 * 1024;
    if (shared_budget > 0 && pool_bytes + kOtherUmaBytes > shared_budget) {
        std::cout << "      WARNING: a " << (pool_bytes / (1024 * 1024))
                  << " MB pool plus resident weights and KV cache will not fit the adapter's "
                  << (shared_budget / (1024 * 1024))
                  << " MB shared-memory budget. Allocation is likely to fail partway through "
                     "loading. Lower --slots.\n";
    } else if (mem.avail_system_ram_bytes > 0 && pool_bytes > mem.avail_system_ram_bytes) {
        std::cout << "      WARNING: that pool exceeds available RAM ("
                  << (mem.avail_system_ram_bytes / (1024 * 1024))
                  << " MB free). Expect the OS page cache to be squeezed and expert I/O to "
                     "get slower, not faster. Lower --slots if throughput drops.\n";
    }

    pipeline_mgr_ = std::make_unique<ComputePipelineManager>(ctx_);
    pipeline_mgr_->initialize_pipelines(
        ComputePipelineManager::descriptors_for(expert_slots_per_layer_, arch.num_layers));

    // Lazy vector for per-layer Expert Streamers (slots allocated on demand)
    streamers_.resize(arch.num_layers);

    const uint64_t D = static_cast<uint64_t>(arch.hidden_size);
    const uint64_t d_bytes = D * 4;   // FP32 activations
    const uint64_t max_head_dim = static_cast<uint64_t>(
        std::max(arch.head_dim, arch.full_head_dim));
    const uint64_t max_kv_heads = static_cast<uint64_t>(
        std::max(arch.num_kv_heads, arch.num_full_kv_heads));
    const uint64_t q_bytes = static_cast<uint64_t>(arch.num_heads) * max_head_dim * 4;
    const uint64_t kv_bytes = max_kv_heads * max_head_dim * 4;
    const uint64_t ffn_bytes = static_cast<uint64_t>(
        std::max(arch.ffn_intermediate, arch.moe_intermediate_size)) * 4;

    buf_hidden_        = ctx_->create_uma_buffer(d_bytes, "hidden");
    buf_normed_        = ctx_->create_uma_buffer(d_bytes, "normed");
    buf_q_             = ctx_->create_uma_buffer(q_bytes, "q");
    buf_q2_            = ctx_->create_uma_buffer(q_bytes, "q_roped");
    buf_k_             = ctx_->create_uma_buffer(kv_bytes, "k");
    buf_v_             = ctx_->create_uma_buffer(kv_bytes, "v");
    buf_ctx_           = ctx_->create_uma_buffer(q_bytes, "ctx");
    buf_attn_out_      = ctx_->create_uma_buffer(d_bytes, "attn_out");
    buf_dense_x_       = ctx_->create_uma_buffer(d_bytes, "dense_x");
    buf_routed_x_      = ctx_->create_uma_buffer(d_bytes, "routed_x");
    buf_router_x_      = ctx_->create_uma_buffer(d_bytes, "router_x");
    buf_router_in_     = ctx_->create_uma_buffer(d_bytes, "router_in");
    buf_h1_            = ctx_->create_uma_buffer(d_bytes, "h1_dense");
    buf_h2_            = ctx_->create_uma_buffer(d_bytes, "h2_routed");
    buf_scratch_d_     = ctx_->create_uma_buffer(d_bytes, "scratch_d");
    buf_ffn_gate_      = ctx_->create_uma_buffer(ffn_bytes, "ffn_gate");
    buf_ffn_up_        = ctx_->create_uma_buffer(ffn_bytes, "ffn_up");
    buf_ffn_act_       = ctx_->create_uma_buffer(ffn_bytes, "ffn_act");
    buf_shared_gate_   = ctx_->create_uma_buffer(ffn_bytes, "shared_gate");
    buf_shared_up_     = ctx_->create_uma_buffer(ffn_bytes, "shared_up");
    buf_shared_act_    = ctx_->create_uma_buffer(ffn_bytes, "shared_act");
    buf_shared_out_    = ctx_->create_uma_buffer(d_bytes, "shared_out");
    buf_router_logits_ = ctx_->create_uma_buffer(
        static_cast<uint64_t>(arch.num_experts) * 4, "router_logits");
    buf_router_indices_ = ctx_->create_uma_buffer(
        static_cast<uint64_t>(arch.top_k_experts) * 4, "router_indices");
    buf_router_weights_ = ctx_->create_uma_buffer(
        static_cast<uint64_t>(arch.top_k_experts) * 4, "router_weights");
    // Doubles as the LMHeadGreedy summaries buffer (8 bytes per threadgroup, well under
    // vocab_size * 4) and as the full logit vector on the sampled path.
    buf_logits_        = ctx_->create_uma_buffer(
        static_cast<uint64_t>(arch.vocab_size) * 4, "logits");
    buf_out_token_     = ctx_->create_uma_buffer(4, "out_token");

    // Context length. Only the 5 full-attention layers scale with it -- the 25 sliding-window
    // layers are capped at `sliding_window` slots by the ring buffer -- so it costs about
    // 41 MB per 1024 tokens rather than the ~340 MB it would without the ring.
    if (requested_context_ > 0) {
        max_context_ = requested_context_;
    } else {
        const double ram_gb = ctx_
            ? static_cast<double>(ctx_->query_memory_info().total_system_ram_bytes) / (1024.0 * 1024.0 * 1024.0)
            : 16.0;
        max_context_ = (ram_gb >= 22.0) ? 4096 : 2048;
    }
    if (max_context_ > kAttentionMaxSpan) {
        throw GTurboFormatError(
            "Requested context " + std::to_string(max_context_) + " exceeds the attention "
            "kernel's staged-score span (" + std::to_string(kAttentionMaxSpan) + "). Raise "
            "ATTN_MAX_SPAN in shaders/Attention.hlsl, mindful of the 32 KB groupshared limit.");
    }

    kv_cache_ = std::make_unique<KVCacheManager>(
        ctx_, arch.num_layers, arch.num_kv_heads, arch.num_full_kv_heads,
        arch.head_dim, arch.full_head_dim, max_context_, arch.sliding_window,
        arch.full_attention_layer_mask);
    kv_cache_->initialize();

    std::cout << "      Context:      " << max_context_ << " tokens, KV cache "
              << (kv_cache_->total_memory_bytes() / (1024 * 1024)) << " MB"
              << " (sliding-window layers ring at " << arch.sliding_window << ")\n";

    // Resident weights are mandatory. A missing, truncated, or undecodable index makes the
    // model unloadable -- substituting an empty buffer here yields all-zero logits, which is
    // indistinguishable from a working engine right up until the output is garbage.
    std::string resident_bin_path = model_dir_ + "/resident.bin";
    if (!std::filesystem::exists(resident_bin_path)) {
        throw GTurboFormatError("Missing resident.bin in model bundle '" + model_dir_ + "'");
    }
    uint64_t file_size = std::filesystem::file_size(resident_bin_path);
    if (file_size < GTurboFormatV1::RESIDENT_HEADER_BYTES) {
        throw GTurboFormatError("resident.bin is truncated (" + std::to_string(file_size) +
                                " bytes, need at least " +
                                std::to_string(GTurboFormatV1::RESIDENT_HEADER_BYTES) + ")");
    }

    // Read-only: resident weights are always bound as an SRV, never written by a shader.
    buf_resident_weights_ = ctx_->create_uma_buffer(file_size, "resident_weights",
                                                    /*needs_uav=*/false);
    std::ifstream file(resident_bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw GTurboFormatError("Failed to open " + resident_bin_path);
    }

    void* ptr = nullptr;
    D3D12_RANGE r{0, file_size};
    if (FAILED(buf_resident_weights_->Map(0, &r, &ptr)) || !ptr) {
        throw GTurboFormatError("Failed to map UMA buffer for resident.bin");
    }
    file.read(reinterpret_cast<char*>(ptr), file_size);
    if (static_cast<uint64_t>(file.gcount()) != file_size) {
        buf_resident_weights_->Unmap(0, nullptr);
        throw GTurboFormatError("Short read on resident.bin: got " +
                                std::to_string(file.gcount()) + " of " +
                                std::to_string(file_size) + " bytes");
    }

    auto header = ResidentIndexCodec::decode_header(reinterpret_cast<const uint8_t*>(ptr), file_size);
    auto entries = ResidentIndexCodec::decode_region(reinterpret_cast<const uint8_t*>(ptr), file_size, header);
    // Deliberately left mapped for the runner's lifetime: it is host-coherent UMA memory,
    // and a few scalars (layer_scalar) are read on the CPU each token.
    resident_cpu_ = reinterpret_cast<const uint8_t*>(ptr);

    for (const auto& entry : entries) {
        resident_entries_[entry.name] = entry;
    }
    if (resident_entries_.empty()) {
        throw GTurboFormatError("resident.bin declares no tensors; the bundle is a placeholder");
    }
    resident_weights_bytes_ = file_size;

    // Probe the device once, here, now that the largest single allocation this engine makes
    // has landed. An over-greedy UMA commitment is the most plausible way to lose the device
    // on an APU, and a lost device does not announce itself -- fence waits start returning
    // instantly and readbacks hand back stale bytes, so generation continues at an
    // impressive tokens/sec producing nonsense. Failing at load is the difference between a
    // clear error and a benchmark result nobody can explain.
    if (ctx_ && !ctx_->device_ok()) {
        throw GTurboFormatError(
            "The D3D12 device was lost while loading resident weights: " +
            ctx_->device_removed_reason() +
            ". Lower --slots or --context; " +
            std::to_string(resident_weights_bytes_ / (1024 * 1024)) +
            " MB of resident weights had just been committed.");
    }
}

ExpertStreamer* ForwardRunner::ensure_streamer_opened(int L) {
    std::lock_guard<std::mutex> guard(streamer_mutex_);
    if (L < 0 || L >= static_cast<int>(streamers_.size())) return nullptr;
    if (!streamers_[L]) {
        // Built as a std::string, not into a fixed buffer. This was a char[64], which fits
        // "gemma-4-26b-a4b.gturbo/packed_experts/layer_00.bin" and nothing longer -- an
        // absolute model path silently truncated to ".../packed_expert" and the open failed
        // with a Win32 "file not found" naming a path the user never supplied.
        char suffix[32];
        snprintf(suffix, sizeof(suffix), "/packed_experts/layer_%02d.bin", L);
        const std::string file_path = model_dir_ + suffix;
        auto streamer = std::make_unique<ExpertStreamer>(
            ctx_,
            file_path,
            expert_slots_per_layer_,
            manifest_.expert_stride,
            eviction_policy_
        );
        streamer->initialize();
        streamers_[L] = std::move(streamer);
    }
    return streamers_[L].get();
}

// Resolves a resident tensor to byte offsets into buf_resident_weights_.
//
// Names are the HuggingFace names with "language_model." stripped, exactly as the converter
// writes them. The previous code asked for GGUF-style "blk.0.attn_norm.weight", which is in
// no index this project ever produced, so every lookup silently returned offset 0.
ForwardRunner::TensorRef ForwardRunner::resident_ref(const std::string& name) const {
    auto it = resident_entries_.find(name);
    if (it == resident_entries_.end()) {
        throw GTurboFormatError("resident index has no tensor '" + name + "'");
    }
    const auto& e = it->second;
    if (e.file_offset > 0xFFFFFFFFull) {
        throw GTurboFormatError(name + ": offset exceeds the 4 GB addressable by a "
                                "ByteAddressBuffer");
    }

    TensorRef r;
    r.data_off = static_cast<uint32_t>(e.file_offset);
    r.rows = e.shape.empty() ? 1u : e.shape[0];
    r.quantized = (e.scale_size > 0);
    if (r.quantized) {
        r.scale_off = static_cast<uint32_t>(e.scale_offset);
        r.bias_off = static_cast<uint32_t>(e.bias_offset);
        // The scales table has in_dim/64 entries per row, which is what tells us in_dim.
        const uint32_t groups = static_cast<uint32_t>(e.scale_size / 2 / r.rows);
        r.in_dim = groups * 64u;
    } else {
        r.in_dim = r.rows;
    }
    return r;
}

ForwardRunner::TensorRef ForwardRunner::layer_ref(int layer, const std::string& suffix) const {
    return resident_ref("model.layers." + std::to_string(layer) + "." + suffix);
}

uint32_t ForwardRunner::bf16_off(const std::string& name) const {
    return resident_ref(name).data_off;
}

float ForwardRunner::read_bf16_scalar(const std::string& name) const {
    if (!resident_cpu_) {
        throw GTurboFormatError("resident weights are not mapped");
    }
    const uint32_t off = resident_ref(name).data_off;
    uint16_t half;
    std::memcpy(&half, resident_cpu_ + off, sizeof(half));
    const uint32_t w = static_cast<uint32_t>(half) << 16;
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

uint32_t ForwardRunner::produce_token(uint32_t input_token, int position,
                                      const GenerationOptions& opts,
                                      const std::vector<uint32_t>& history) {
    const auto& arch = manifest_.arch;
    const uint32_t D = static_cast<uint32_t>(arch.hidden_size);
    const uint32_t K = static_cast<uint32_t>(arch.top_k_experts);
    const float eps = static_cast<float>(arch.rms_norm_eps);

    if (position >= max_context_) {
        // Sliding-window layers could run indefinitely -- their ring never fills. The limit
        // comes from the full-attention layers, which keep the whole history, and from the
        // attention kernel's staged-score span.
        throw GTurboFormatError("Context limit reached (" + std::to_string(max_context_) +
                                " tokens). Raise it with --context, up to " +
                                std::to_string(kAttentionMaxSpan) + ".");
    }

    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    ID3D12Resource* RES = buf_resident_weights_.Get();

    // The router's top-K must be read on the CPU before the experts it selects can be
    // streamed, so a layer cannot be a single command list. Each `submit` below is a
    // fence point. This is correctness-first: 60+ submissions per token is slow, and
    // collapsing them is the main Stage 4 optimization.
    using clock = std::chrono::high_resolution_clock;
    auto ms_since = [](clock::time_point t) {
        return std::chrono::duration<double, std::milli>(clock::now() - t).count();
    };

    // Descriptor tables are cached for the process lifetime and never recycled -- a table
    // must not be overwritten while a command list referencing it is still in flight, and
    // there is no per-descriptor fence tracking to make that safe. The heap is sized up
    // front for the whole run by ComputePipelineManager::descriptors_for(), so nothing is
    // reset per token.

    ComPtr<ID3D12GraphicsCommandList> cl;
    auto begin = [&]() { cl = ctx_->reset_command_list(); };

    // Submit without blocking. Work on a single queue executes in submission order, so a
    // dependent list does not need a CPU round trip -- only an actual CPU-side *read* of GPU
    // output does. That is why only the router top-K and the final logits force a wait.
    auto submit = [&]() { ctx_->submit_command_list(cl.Get()); };

    auto submit_and_wait = [&]() {
        const uint64_t v = ctx_->submit_command_list(cl.Get());
        const auto t0 = clock::now();
        ctx_->wait_for_fence(v);
        metrics_.gpu_wait_ms += ms_since(t0);
        metrics_.gpu_waits++;
    };
    auto dispatch = [&](KernelType k, const std::vector<ID3D12Resource*>& srvs,
                        const std::vector<ID3D12Resource*>& uavs,
                        const KernelDispatchParams& p) {
        pipeline_mgr_->dispatch(cl.Get(), k, srvs, uavs, p);
        ComputePipelineManager::barrier(cl.Get(), uavs);
    };

    // out = W * x, chunked because a dispatch dimension caps at 65535 threadgroups and the
    // tied LM head has 262,144 rows.
    auto gemv = [&](const TensorRef& w, ID3D12Resource* x, uint32_t x_off,
                    ID3D12Resource* out, uint32_t out_off, bool int8 = false) {
        constexpr uint32_t MAX_GROUPS = 65535;
        for (uint32_t base = 0; base < w.rows; base += MAX_GROUPS) {
            KernelDispatchParams p{};
            p.grid_x = std::min(MAX_GROUPS, w.rows - base);
            p.set({w.rows, w.in_dim, w.data_off, w.scale_off,
                           w.bias_off, x_off, out_off, base});
            dispatch(int8 ? KernelType::GemvInt8 : KernelType::GemvInt4,
                     {RES, RES, RES, x}, {out}, p);
        }
    };

    auto rmsnorm = [&](ID3D12Resource* x, uint32_t x_off, uint32_t w_off, bool has_w,
                       ID3D12Resource* out, uint32_t out_off, uint32_t n) {
        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({n, x_off, w_off, out_off, has_w ? 1u : 0u, bits(eps), 0, 0});
        dispatch(KernelType::RMSNormK, {x, RES}, {out}, p);
    };

    // ---- Embedding ------------------------------------------------------
    const TensorRef embed = resident_ref("model.embed_tokens.weight");
    if (input_token >= embed.rows) {
        throw GTurboFormatError("token id " + std::to_string(input_token) + " out of vocabulary");
    }
    begin();
    {
        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({input_token, embed.in_dim, embed.data_off, embed.scale_off,
                       embed.bias_off, 0, bits(std::sqrt(static_cast<float>(D))), 0});
        dispatch(KernelType::EmbedLookup, {RES, RES, RES}, {buf_hidden_.Get()}, p);
    }
    submit();

    std::vector<uint32_t> top_idx(K);

    for (int L = 0; L < arch.num_layers; ++L) {
        const bool is_full = (L < static_cast<int>(arch.full_attention_layer_mask.size())) &&
                             arch.full_attention_layer_mask[static_cast<size_t>(L)] != 0;
        const uint32_t head_dim = static_cast<uint32_t>(is_full ? arch.full_head_dim : arch.head_dim);
        const uint32_t kv_heads = static_cast<uint32_t>(is_full ? arch.num_full_kv_heads : arch.num_kv_heads);
        const uint32_t q_heads = static_cast<uint32_t>(arch.num_heads);
        const float theta = static_cast<float>(is_full ? arch.full_rope_theta : arch.rope_theta);
        // Partial rotary is a full-attention-layer property only.
        const uint32_t rotated = is_full
            ? static_cast<uint32_t>(head_dim * arch.partial_rotary_factor / 2.0)
            : head_dim / 2;
        const std::string P = "model.layers." + std::to_string(L) + ".";
        const uint32_t kv_row = kv_heads * head_dim * 4;
        const uint32_t kv_capacity = static_cast<uint32_t>(kv_cache_->layer_capacity(L));
        ID3D12Resource* KC = kv_cache_->key_buffer(L);
        ID3D12Resource* VC = kv_cache_->value_buffer(L);

        // ---- Part 1: attention through the router ------------------------
        begin();
        rmsnorm(buf_hidden_.Get(), 0, bf16_off(P + "input_layernorm.weight"), true,
                buf_normed_.Get(), 0, D);

        const TensorRef wq = layer_ref(L, "self_attn.q_proj.weight");
        const TensorRef wk = layer_ref(L, "self_attn.k_proj.weight");
        gemv(wq, buf_normed_.Get(), 0, buf_q_.Get(), 0);
        gemv(wk, buf_normed_.Get(), 0, buf_k_.Get(), 0);
        if (arch.attention_k_eq_v && is_full) {
            // Full-attention layers have no v_proj: V reuses the raw k_proj output.
            gemv(wk, buf_normed_.Get(), 0, buf_v_.Get(), 0);
        } else {
            gemv(layer_ref(L, "self_attn.v_proj.weight"), buf_normed_.Get(), 0, buf_v_.Get(), 0);
        }

        auto epilogue = [&](ID3D12Resource* in, ID3D12Resource* out, uint32_t out_off,
                            uint32_t heads, uint32_t w_off, bool has_w, bool do_rope) {
            KernelDispatchParams p{};
            p.grid_x = heads;
            p.set({head_dim, heads, 0, out_off,
                           bits(eps), w_off, has_w ? 1u : 0u, do_rope ? 1u : 0u,
                           rotated, static_cast<uint32_t>(position), bits(theta), 0});
            dispatch(KernelType::QKVEpilogue, {in, RES}, {out}, p);
        };
        // Ring: position p lands on slot p % capacity, overwriting p - capacity, which for a
        // sliding-window layer is exactly the entry leaving the window.
        const uint32_t kv_slot =
            static_cast<uint32_t>(kv_cache_->physical_slot(L, position)) * kv_row;
        epilogue(buf_q_.Get(), buf_q2_.Get(), 0, q_heads,
                 bf16_off(P + "self_attn.q_norm.weight"), true, true);
        epilogue(buf_k_.Get(), KC, kv_slot, kv_heads,
                 bf16_off(P + "self_attn.k_norm.weight"), true, true);
        // V takes a no-scale RMSNorm and skips RoPE entirely.
        epilogue(buf_v_.Get(), VC, kv_slot, kv_heads, 0, false, false);

        {
            const uint32_t n_pos = static_cast<uint32_t>(position) + 1;
            const uint32_t first = is_full ? 0u
                : (n_pos > static_cast<uint32_t>(arch.sliding_window)
                   ? n_pos - static_cast<uint32_t>(arch.sliding_window) : 0u);
            KernelDispatchParams p{};
            p.grid_x = q_heads;
            p.set({q_heads, kv_heads, head_dim, n_pos, first, 0, 0, 0, 0, kv_capacity, 0, 0});
            dispatch(KernelType::Attention, {buf_q2_.Get(), KC, VC}, {buf_ctx_.Get()}, p);
        }

        gemv(layer_ref(L, "self_attn.o_proj.weight"), buf_ctx_.Get(), 0, buf_attn_out_.Get(), 0);

        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({D, 0, 0, 0,
                           bits(eps), bf16_off(P + "post_attention_layernorm.weight"),
                           bf16_off(P + "pre_feedforward_layernorm.weight"),
                           bf16_off(P + "pre_feedforward_layernorm_2.weight"),
                           0, 0, 0, 0});
            dispatch(KernelType::PostAttn,
                     {buf_attn_out_.Get(), buf_hidden_.Get(), RES, RES, RES},
                     {buf_hidden_.Get(), buf_dense_x_.Get(), buf_routed_x_.Get(),
                      buf_router_x_.Get()}, p);
        }

        // Router: input is rmsnorm_no_scale(hidden) * router.scale / sqrt(D).
        {
            KernelDispatchParams p{};
            p.grid_x = (D + 255) / 256;
            p.set({D, 0, bf16_off(P + "router.scale"), 0,
                           bits(1.0f / std::sqrt(static_cast<float>(D))), 0, 0, 0});
            dispatch(KernelType::MulBF16, {buf_router_x_.Get(), RES}, {buf_router_in_.Get()}, p);
        }
        gemv(layer_ref(L, "router.proj.weight"), buf_router_in_.Get(), 0,
             buf_router_logits_.Get(), 0, /*int8=*/true);
        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({static_cast<uint32_t>(arch.num_experts), K, 0,
                           bf16_off(P + "router.per_expert_scale"), 0, 0, 0, 0});
            dispatch(KernelType::RouterTopK, {buf_router_logits_.Get(), RES},
                     {buf_router_indices_.Get(), buf_router_weights_.Get()}, p);
        }
        // The only unavoidable stall in a layer: the top-K has to reach the CPU before we
        // know which expert bytes to fetch.
        submit_and_wait();

        // ---- Fence point: read the routing decision, then stream those experts ----
        {
            void* p = nullptr;
            D3D12_RANGE r{0, K * sizeof(uint32_t)};
            if (FAILED(buf_router_indices_->Map(0, &r, &p)) || !p) {
                throw GTurboFormatError(
                    "Failed to map router indices for readback."
                    " A Map failure here almost always means the UMA/shared-memory budget is"
                    " exhausted rather than anything wrong with the buffer itself -- lower"
                    " --slots or --context.");
            }
            std::memcpy(top_idx.data(), p, K * sizeof(uint32_t));
            buf_router_indices_->Unmap(0, nullptr);
        }
        if (L == 0) {
            // Telemetry reads this from another thread; a vector::assign racing a
            // copy-construct is a genuine data race, not merely a torn integer.
            std::lock_guard<std::mutex> guard(active_experts_mutex_);
            last_active_experts_.assign(top_idx.begin(), top_idx.end());
        }

        ExpertStreamer* streamer = ensure_streamer_opened(L);
        if (!streamer) {
            throw GTurboFormatError("No expert streamer for layer " + std::to_string(L));
        }
        std::vector<int> ids(top_idx.begin(), top_idx.end());

        // Resolve the cache without doing any I/O yet, so the work that does not depend on
        // the missing bytes can be queued first.
        auto plan = streamer->plan_experts(L, ids);
        if (plan.slots.size() != K) {
            throw GTurboFormatError("Expert streamer returned " +
                                    std::to_string(plan.slots.size()) +
                                    " slots, expected " + std::to_string(K));
        }

        // Runs one expert's gate/up -> GeGLU -> down -> weighted accumulate into h2.
        // `write_index` orders the accumulation (the first write overwrites, which also
        // clears the buffer); `request_index` selects the routing weight and must stay the
        // router's own ordering.
        uint32_t writes = 0;
        auto encode_expert = [&](size_t request_index) {
            ID3D12Resource* slot = plan.slots[request_index]->buffer.Get();
            auto sub = [&](const char* role, const char* part) {
                auto it = layout_.expert_block.find(std::string(role) + "." + part);
                if (it == layout_.expert_block.end()) {
                    throw GTurboFormatError(std::string("layout.expertBlock missing ") + role);
                }
                return it->second;
            };
            auto expert_gemv = [&](const char* role, ID3D12Resource* x, ID3D12Resource* out) {
                const auto& w = sub(role, "weight");
                const auto& s = sub(role, "scales");
                const auto& b = sub(role, "biases");
                const uint32_t rows = w.shape.empty() ? 0u : w.shape[0];
                const uint32_t in_dim = static_cast<uint32_t>(s.size / 2 / rows) * 64u;
                KernelDispatchParams p{};
                p.grid_x = rows;
                p.set({rows, in_dim,
                       static_cast<uint32_t>(w.offset),
                       static_cast<uint32_t>(s.offset),
                       static_cast<uint32_t>(b.offset), 0, 0, 0});
                dispatch(KernelType::GemvInt4, {slot, slot, slot, x}, {out}, p);
                return rows;
            };

            const uint32_t rows = expert_gemv("gate_proj", buf_routed_x_.Get(), buf_ffn_gate_.Get());
            expert_gemv("up_proj", buf_routed_x_.Get(), buf_ffn_up_.Get());
            {
                KernelDispatchParams p{};
                p.grid_x = (rows + 255) / 256;
                p.set({rows, 0, 0, 0});
                dispatch(KernelType::GeGLU, {buf_ffn_gate_.Get(), buf_ffn_up_.Get()},
                         {buf_ffn_act_.Get()}, p);
            }
            expert_gemv("down_proj", buf_ffn_act_.Get(), buf_scratch_d_.Get());
            {
                KernelDispatchParams p{};
                p.grid_x = (D + 255) / 256;
                p.set({D, 0, 0, 0, (writes == 0) ? 0u : 1u,
                       static_cast<uint32_t>(request_index), 0, 0});
                dispatch(KernelType::ScaleAccum,
                         {buf_scratch_d_.Get(), buf_router_weights_.Get()},
                         {buf_h2_.Get()}, p);
            }
            ++writes;
        };

        // ---- Part 2a: work that needs no missing bytes ------------------
        // The shared expert depends only on dense_x, and cache-hit experts already have
        // their weights resident. Queueing both before the reads keeps the GPU busy for
        // what used to be a pure stall.
        begin();
        {
            // Shared expert reads dense_x (pre_feedforward_layernorm).
            const TensorRef g = layer_ref(L, "mlp.gate_proj.weight");
            const TensorRef u = layer_ref(L, "mlp.up_proj.weight");
            const TensorRef d = layer_ref(L, "mlp.down_proj.weight");
            // Its own scratch, so nothing here aliases the routed-expert chain and the GPU is
            // free to run the two branches concurrently.
            gemv(g, buf_dense_x_.Get(), 0, buf_shared_gate_.Get(), 0);
            gemv(u, buf_dense_x_.Get(), 0, buf_shared_up_.Get(), 0);
            KernelDispatchParams p{};
            p.grid_x = (g.rows + 255) / 256;
            p.set({g.rows, 0, 0, 0});
            dispatch(KernelType::GeGLU, {buf_shared_gate_.Get(), buf_shared_up_.Get()},
                     {buf_shared_act_.Get()}, p);
            gemv(d, buf_shared_act_.Get(), 0, buf_shared_out_.Get(), 0);
            // The shared branch is added with NO routing weight.
            rmsnorm(buf_shared_out_.Get(), 0,
                    bf16_off(P + "post_feedforward_layernorm_1.weight"), true,
                    buf_h1_.Get(), 0, D);
        }

        // Cache-hit experts: their weights are already resident, so they need no I/O.
        for (size_t i : plan.hits) {
            encode_expert(i);
        }
        submit();

        // ---- The reads, with the GPU already working on part 2a ----------
        const auto t_io = clock::now();
        streamer->fetch_misses(plan);
        metrics_.expert_io_ms += ms_since(t_io);

        // ---- Part 2b: the experts that needed bytes, then the layer tail -
        begin();
        for (size_t i : plan.misses) {
            encode_expert(i);
        }

        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({D, 0, 0, 0,
                           bits(eps),
                           bf16_off(P + "post_feedforward_layernorm_2.weight"),
                           bf16_off(P + "post_feedforward_layernorm.weight"),
                           0,
                           bits(read_bf16_scalar(P + "layer_scalar")), 0, 0, 0});
            dispatch(KernelType::LayerTail,
                     {buf_h2_.Get(), buf_h1_.Get(), buf_hidden_.Get(), RES, RES},
                     {buf_hidden_.Get(), buf_scratch_d_.Get()}, p);
        }
        submit();
        streamer->release_plan(plan);
    }

    // ---- Final norm, tied LM head, softcap -------------------------------
    const auto t_head = clock::now();
    begin();
    rmsnorm(buf_hidden_.Get(), 0, bf16_off("model.norm.weight"), true,
            buf_normed_.Get(), 0, D);

    const uint32_t V = static_cast<uint32_t>(arch.vocab_size);
    uint32_t best_id = 0;

    // Sampling and the repetition penalty both need the whole distribution on the host, so
    // they force the materializing path. Greedy stays fused -- it is the default and the fast
    // path, and keeping it bit-identical is what preserves the token-for-token gate against
    // the CPU reference.
    const SamplingParams sampling = opts.sampling();
    const bool needs_logits = !sampling.is_greedy();

    if (!needs_logits) {
        // Greedy: reduce straight to a token id on the GPU. The 262,144-wide logit vector is
        // never materialized and only 4 bytes come back. Softcap is skipped because
        // 30*tanh(z/30) is monotonic and cannot move the argmax.
        constexpr uint32_t LMH_THREADS = 512;
        // Rows per group is the wave count, which is 8 on RDNA 3 (Wave64) and 16 on Wave32.
        // Assume the smaller so the grid is always large enough; surplus groups exit early.
        constexpr uint32_t MIN_ROWS_PER_GROUP = LMH_THREADS / 64;
        const uint32_t groups = (V + MIN_ROWS_PER_GROUP - 1) / MIN_ROWS_PER_GROUP;
        {
            KernelDispatchParams p{};
            p.grid_x = groups;
            p.set({V, embed.in_dim, embed.data_off, embed.scale_off,
                   embed.bias_off, 0, 0, 0});
            dispatch(KernelType::LMHeadGreedy, {RES, RES, RES, buf_normed_.Get()},
                     {buf_logits_.Get()}, p);
        }
        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({groups, 0, 0, 0});
            dispatch(KernelType::ArgmaxReduce, {buf_logits_.Get()}, {buf_out_token_.Get()}, p);
        }
        submit_and_wait();

        void* p = nullptr;
        D3D12_RANGE r{0, sizeof(uint32_t)};
        if (FAILED(buf_out_token_->Map(0, &r, &p)) || !p) {
            throw GTurboFormatError(
                    "Failed to map output token for readback."
                    " A Map failure here almost always means the UMA/shared-memory budget is"
                    " exhausted rather than anything wrong with the buffer itself -- lower"
                    " --slots or --context.");
        }
        best_id = *static_cast<const uint32_t*>(p);
        buf_out_token_->Unmap(0, nullptr);
    } else {
        // Sampled decode needs the whole distribution, so keep the materializing path.
        gemv(embed, buf_normed_.Get(), 0, buf_logits_.Get(), 0);   // tie_word_embeddings
        {
            KernelDispatchParams p{};
            p.grid_x = (V + 255) / 256;
            p.set({V, 0, 0, bits(static_cast<float>(arch.final_logit_softcap))});
            dispatch(KernelType::Softcap, {buf_logits_.Get()}, {buf_logits_.Get()}, p);
        }
        submit_and_wait();

        // Copy out and unmap before doing CPU work, rather than holding a mapping across it.
        // On UMA the Map is a pointer hand-back, so this is a memcpy of 1 MB, not a transfer.
        if (logits_scratch_.size() != V) logits_scratch_.resize(V);
        {
            void* p = nullptr;
            D3D12_RANGE r{0, static_cast<size_t>(V) * 4};
            if (FAILED(buf_logits_->Map(0, &r, &p)) || !p) {
                throw GTurboFormatError(
                    "Failed to map logits for readback."
                    " A Map failure here almost always means the UMA/shared-memory budget is"
                    " exhausted rather than anything wrong with the buffer itself -- lower"
                    " --slots or --context.");
            }
            std::memcpy(logits_scratch_.data(), p, static_cast<size_t>(V) * 4);
            buf_logits_->Unmap(0, nullptr);
        }

        // The penalty is applied to the already-softcapped values -- see the note in
        // sampling.hpp. Applying it before the softcap would be very nearly a no-op.
        apply_repetition_penalty(logits_scratch_.data(), V, history,
                                 sampling.repetition_penalty,
                                 static_cast<float>(arch.final_logit_softcap));

        best_id = sample_token(logits_scratch_.data(), V, sampling,
                               seed_for(sampling, position));
    }
    metrics_.lm_head_ms += ms_since(t_head);
    metrics_.tokens_measured++;
    // Track how far the ring has been written. Attention indexes off `position` directly, so
    // this is not load-bearing for a single request -- it is what lets a later request know
    // where a reusable prefix ends.
    if (kv_cache_) kv_cache_->set_position(position + 1);
    return best_id;
}


GenerationResult ForwardRunner::generate_tokens(const std::vector<uint32_t>& prompt_tokens,
                                                const GenerationOptions& opts,
                                                const StreamCallback& on_event) {
    validate_sampling(opts.sampling());
    reset_stop_flag();

    GenerationResult result;
    result.prompt_tokens = static_cast<int>(prompt_tokens.size());
    result.reason = StopReason::MaxTokens;

    metrics_ = PerformanceMetrics{};
    last_stop_reason_ = StopReason::MaxTokens;
    if (kv_cache_) kv_cache_->reset();

    using clock = std::chrono::high_resolution_clock;
    const auto start_time = clock::now();

    IncrementalDetokenizer detok(tokenizer_, /*skip_special=*/true);
    StreamingStopMatcher stop_matcher(opts.stop_strings);

    bool cancelled = false;
    auto emit = [&](const StreamEvent& ev) {
        if (on_event && !on_event(ev)) cancelled = true;
    };

    // Prefill. Prompt tokens build KV state and are not part of the output.
    uint32_t current_token = 0;
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (is_stop_requested() || cancelled) { result.reason = StopReason::Cancelled; break; }
        current_token = produce_token(prompt_tokens[i], static_cast<int>(i), opts, prompt_tokens);

        StreamEvent ev;
        ev.kind = StreamEvent::Kind::Prefill;
        ev.done = static_cast<int>(i + 1);
        ev.total = static_cast<int>(prompt_tokens.size());
        emit(ev);
    }

    const auto prefill_done_time = clock::now();

    if (result.reason != StopReason::Cancelled) {
        const int start_pos = static_cast<int>(prompt_tokens.size());
        const auto& stop_ids = tokenizer_.stop_token_ids();
        std::vector<uint32_t> history;
        if (opts.repetition_penalty != 1.0f) history = prompt_tokens;

        for (int t = 0; t < opts.max_tokens; ++t) {
            if (is_stop_requested() || cancelled) { result.reason = StopReason::Cancelled; break; }

            if (std::find(stop_ids.begin(), stop_ids.end(), current_token) != stop_ids.end()) {
                result.reason = (current_token == tokenizer_.eos_id())
                    ? StopReason::EndOfSequence : StopReason::EndOfTurn;
                break;
            }

            if (result.tokens.empty()) {
                result.ttft_ms =
                    std::chrono::duration<double, std::milli>(clock::now() - start_time).count();
            }
            result.tokens.push_back(current_token);
            if (opts.repetition_penalty != 1.0f) history.push_back(current_token);

            // Detokenize first, then stop-match. Reversing these would let a stop string be
            // triggered by text the caller never receives.
            const std::string visible = stop_matcher.push(detok.push(current_token));
            result.text += visible;

            StreamEvent ev;
            ev.kind = StreamEvent::Kind::Token;
            ev.index = t;
            ev.id = current_token;
            ev.delta = visible;
            emit(ev);

            if (stop_matcher.stopped()) {
                result.reason = StopReason::StopString;
                result.matched_stop = stop_matcher.matched();
                break;
            }
            if (t + 1 >= opts.max_tokens) break;
            current_token = produce_token(current_token, start_pos + t, opts, history);
        }
    }

    // Release whatever the detokenizer and stop matcher were still holding.
    if (result.reason != StopReason::StopString) {
        std::string tail = stop_matcher.push(detok.finish());
        tail += stop_matcher.finish();
        if (!tail.empty()) {
            result.text += tail;
            StreamEvent ev;
            ev.kind = StreamEvent::Kind::Tail;
            ev.delta = tail;
            emit(ev);
        }
    }

    const auto total_done_time = clock::now();
    result.completion_tokens = static_cast<int>(result.tokens.size());
    last_stop_reason_ = result.reason;

    const double prefill_ms =
        std::chrono::duration<double, std::milli>(prefill_done_time - start_time).count();
    const double decode_ms =
        std::chrono::duration<double, std::milli>(total_done_time - prefill_done_time).count();

    metrics_.total_time_ms =
        std::chrono::duration<double, std::milli>(total_done_time - start_time).count();
    metrics_.prefill_tokens_per_sec = (prompt_tokens.empty() || prefill_ms <= 0.0)
        ? 0.0 : (prompt_tokens.size() / (prefill_ms / 1000.0));
    metrics_.decode_tokens_per_sec = (result.tokens.empty() || decode_ms <= 0.0)
        ? 0.0 : (result.tokens.size() / (decode_ms / 1000.0));
    // Expert I/O, GPU wait and this residual are DISJOINT and sum to the token.
    //
    // lm_head_ms is deliberately not subtracted here, because it is not a fourth peer bucket:
    // the head is one submit_and_wait, so its cost already sits inside gpu_wait_ms (the fence)
    // and inside this residual (the recording). Subtracting it as though it were disjoint
    // double-counted the fence, which made the four figures sum to more than 100% and pushed
    // this residual negative -- the CLI printed "CPU other: -2.16 ms (-1.93%)" once the
    // optimized build shrank the real CPU work below the size of the overlap. The head is
    // reported separately as an overlapping annotation instead.
    metrics_.cpu_other_ms = metrics_.total_time_ms - metrics_.expert_io_ms -
                            metrics_.gpu_wait_ms;

    uint64_t io_bytes = 0, io_calls = 0;
    {
        std::lock_guard<std::mutex> guard(streamer_mutex_);
        for (const auto& streamer : streamers_) {
            if (streamer) {
                io_bytes += streamer->total_bytes_read();
                io_calls += streamer->total_io_calls();
            }
        }
    }
    metrics_.total_io_bytes = io_bytes;
    metrics_.total_io_calls = io_calls;

    return result;
}

// Retained for callers that only want token ids. The streaming core is generate_tokens.
std::vector<uint32_t> ForwardRunner::generate(const std::vector<uint32_t>& prompt_tokens,
                                             const GenerationOptions& opts) {
    return generate_tokens(prompt_tokens, opts).tokens;
}
GenerationResult ForwardRunner::generate_chat(const std::vector<Tokenizer::ChatMessage>& messages,
                                              const GenerationOptions& opts,
                                              const StreamCallback& on_event) {
    if (messages.empty()) {
        throw GTurboFormatError("generate_chat: no messages");
    }
    // apply_chat_template already accepts arbitrary history and enforces system-must-be-first;
    // add_bos is false because the template already begins with <bos>.
    auto prompt_tokens = tokenizer_.encode(tokenizer_.apply_chat_template(messages), false);
    auto result = generate_tokens(prompt_tokens, opts, on_event);

    // A deliberate stop is not a failure. These guards exist to catch a broken forward pass,
    // but they used to fire on cancellation too: pressing Stop in the GUI truncates the
    // output, which then looks exactly like "produced no tokens" or "emitted one token
    // repeatedly", and the user got an engine error for doing what the button says.
    const bool cancelled = (result.reason == StopReason::Cancelled);

    if (result.tokens.empty()) {
        if (cancelled) return result;
        throw GTurboFormatError(
            "Model produced no tokens. This indicates a broken forward pass or placeholder "
            "weights, not an empty answer.");
    }

    // A forward pass that is not actually computing anything emits the same token forever --
    // typically id 0 (<pad>), because an argmax over all-zero logits always picks index 0.
    // That decodes to an empty string, which must not be reported as a successful reply.
    //
    // The threshold is deliberately above 1: a legitimate short answer like "Yes." is a
    // couple of tokens and can repeat one of them, so a low bar reports real replies as
    // corruption. Genuine degeneracy runs to the token budget, not to three tokens.
    constexpr size_t kDegenerateMinRun = 4;
    bool degenerate = true;
    for (size_t i = 1; i < result.tokens.size(); ++i) {
        if (result.tokens[i] != result.tokens[0]) { degenerate = false; break; }
    }
    if (degenerate && !cancelled && result.tokens.size() >= kDegenerateMinRun) {
        throw GTurboFormatError(
            "Model emitted token " + std::to_string(result.tokens[0]) + " " +
            std::to_string(result.tokens.size()) +
            " times in a row. The forward pass is not computing logits.");
    }

    if (result.text.empty() && !cancelled) {
        throw GTurboFormatError(
            "Model produced " + std::to_string(result.tokens.size()) +
            " tokens that decode to an empty string. This is a broken forward pass, not an "
            "empty answer.");
    }
    return result;
}

std::string ForwardRunner::generate_text(const std::string& prompt, const GenerationOptions& opts) {
    // Wrap the prompt in the Gemma 4 chat template. Feeding raw text to an instruction-tuned
    // model produces continuation, not an answer -- previously the prompt went in bare and
    // the GUI's system prompt was dropped entirely.
    std::vector<Tokenizer::ChatMessage> messages;
    if (!opts.system_prompt.empty()) {
        messages.push_back({"system", opts.system_prompt});
    }
    messages.push_back({"user", prompt});
    return generate_chat(messages, opts).text;
}

const char* stop_reason_name(StopReason reason) {
    switch (reason) {
        case StopReason::EndOfSequence: return "end_of_sequence";
        case StopReason::EndOfTurn:     return "end_of_turn";
        case StopReason::StopString:    return "stop_string";
        case StopReason::MaxTokens:     return "max_tokens";
        case StopReason::Cancelled:     return "cancelled";
    }
    return "unknown";
}

ModelMemoryUsage ForwardRunner::get_memory_usage() const {
    ModelMemoryUsage mem{};
    mem.resident_weights_bytes = resident_weights_bytes_;
    if (kv_cache_) {
        mem.kv_cache_bytes = kv_cache_->total_memory_bytes();
    }
    uint64_t exp_cache = 0;
    {
        std::lock_guard<std::mutex> guard(streamer_mutex_);
        for (const auto& s : streamers_) {
            if (s) exp_cache += s->total_cache_memory_bytes();
        }
    }
    mem.expert_cache_bytes = exp_cache;
    mem.total_model_bytes = mem.resident_weights_bytes + mem.kv_cache_bytes + mem.expert_cache_bytes;
    // One definition of the hit rate. This used to average hit_rate_pct() per layer without
    // weighting, so a layer that served three requests counted as much as one that served
    // three thousand -- and it disagreed with the number the CLI printed.
    mem.cache_hit_rate_pct = expert_cache_hit_rate();
    return mem;
}

void ForwardRunner::clear_expert_cache() {
    std::lock_guard<std::mutex> guard(streamer_mutex_);
    for (auto& s : streamers_) {
        if (s) s->clear_cache();
    }
}

void ForwardRunner::set_eviction_policy(EvictionPolicy policy) {
    std::lock_guard<std::mutex> guard(streamer_mutex_);
    // Store it as well as pushing it down. Streamers open lazily, so a policy that was only
    // pushed to the currently-open ones silently reverted to the default on every layer that
    // had not been touched yet.
    eviction_policy_ = policy;
    for (auto& s : streamers_) {
        if (s) s->set_eviction_policy(policy);
    }
}

std::vector<int> ForwardRunner::last_active_experts() const {
    std::lock_guard<std::mutex> guard(active_experts_mutex_);
    return last_active_experts_;
}

double ForwardRunner::expert_cache_hit_rate() const {
    uint64_t hits = 0, misses = 0;
    std::lock_guard<std::mutex> guard(streamer_mutex_);
    for (const auto& s : streamers_) {
        if (!s) continue;
        hits += s->total_hits();
        misses += s->total_misses();
    }
    return (hits + misses == 0) ? 0.0
        : 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses);
}

} // namespace gturbo
