#define TURBO_BUILD_DLL
#include "gturbo/c_api.h"
#include "gturbo/runner.hpp"
#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/d3d12_context.hpp"
#include "gturbo/json.hpp"
#include <algorithm>
#include <cctype>
#include <vector>

#include <string>
#include <memory>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

struct TurboEngineContext {
    std::shared_ptr<gturbo::D3D12Context> ctx;
    std::shared_ptr<gturbo::ForwardRunner> runner;
    std::string model_dir;
    std::string last_response;
    std::string last_telemetry;
};

namespace {
thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }
void clear_error() { g_last_error.clear(); }
} // namespace

extern "C" {

TURBO_API const char* turbo_engine_last_error(void) {
    return g_last_error.c_str();
}

TURBO_API void* turbo_engine_create(const char* model_dir) {
    clear_error();
    try {
        auto e_ctx = std::make_unique<TurboEngineContext>();
        e_ctx->ctx = std::make_shared<gturbo::D3D12Context>();
        e_ctx->ctx->initialize(false);

        std::string m_dir = (model_dir && strlen(model_dir) > 0) ? model_dir : "gemma-4-26b-a4b.gturbo";
        e_ctx->model_dir = m_dir;

        auto manifest = gturbo::GTurboManifestV1::from_json_string(
            gturbo::read_text_file(m_dir + "/manifest.json"));
        auto layout = gturbo::PackedExpertsLayoutV1::from_json_string(
            gturbo::read_text_file(m_dir + "/packed_experts/layout.json"));
        layout.cross_validate(manifest);

        e_ctx->runner = std::make_shared<gturbo::ForwardRunner>(e_ctx->ctx, manifest, layout, m_dir);
        e_ctx->runner->initialize();

        return static_cast<void*>(e_ctx.release());
    } catch (const std::exception& ex) {
        set_error(ex.what());
        std::cerr << "[C_API] Create failed: " << ex.what() << "\n";
        return nullptr;
    }
}

TURBO_API void turbo_engine_destroy(void* handle) {
    if (handle) {
        auto e_ctx = static_cast<TurboEngineContext*>(handle);
        delete e_ctx;
    }
}

TURBO_API const char* turbo_engine_generate(void* handle, const char* prompt, int max_tokens) {
    clear_error();
    if (!handle) {
        set_error("Null engine handle");
        return nullptr;
    }
    auto e_ctx = static_cast<TurboEngineContext*>(handle);
    if (!e_ctx->runner) {
        set_error("No model loaded");
        return nullptr;
    }
    if (!prompt || *prompt == '\0') {
        set_error("Empty prompt");
        return nullptr;
    }

    // Sampling is left at GenerationOptions' defaults. This used to force temperature 0.7,
    // which not only ignored the caller but pushed every request onto the materializing
    // logits path -- slower, for a value nobody asked for. Per-request sampling arrives with
    // turbo_engine_generate_ex.
    gturbo::GenerationOptions opts{};
    opts.max_tokens = (max_tokens > 0) ? max_tokens : 64;

    try {
        e_ctx->last_response = e_ctx->runner->generate_text(prompt, opts);
        return e_ctx->last_response.c_str();
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return nullptr;
    }
}

TURBO_API void turbo_options_default(TurboGenerationOptions* out) {
    if (!out) return;
    const gturbo::GenerationOptions d{};
    out->max_tokens = 512;
    out->temperature = d.temperature;
    out->top_p = d.top_p;
    out->top_k = d.top_k;
    out->repetition_penalty = d.repetition_penalty;
    out->has_seed = 0;
    out->seed = 0;
    out->system_prompt = nullptr;
    out->stop_strings = nullptr;
    out->stop_count = 0;
}

TURBO_API const char* turbo_engine_generate_ex(void* handle, const char* messages_json,
                                               const TurboGenerationOptions* opts,
                                               TurboTokenCallback on_token, void* user) {
    clear_error();
    if (!handle) { set_error("Null engine handle"); return nullptr; }
    auto e_ctx = static_cast<TurboEngineContext*>(handle);
    if (!e_ctx->runner) { set_error("No model loaded"); return nullptr; }
    if (!messages_json || *messages_json == '\0') { set_error("Empty messages"); return nullptr; }

    gturbo::GenerationOptions g{};
    if (opts) {
        g.max_tokens = (opts->max_tokens > 0) ? opts->max_tokens : 512;
        g.temperature = opts->temperature;
        g.top_p = opts->top_p;
        g.top_k = opts->top_k;
        g.repetition_penalty = opts->repetition_penalty;
        g.has_seed = (opts->has_seed != 0);
        g.seed = opts->seed;
        if (opts->system_prompt) g.system_prompt = opts->system_prompt;
        for (int i = 0; i < opts->stop_count; ++i) {
            if (opts->stop_strings && opts->stop_strings[i]) {
                g.stop_strings.emplace_back(opts->stop_strings[i]);
            }
        }
    }

    try {
        // OpenAI-shaped array in, ChatMessage vector out. Parsing here rather than exposing a
        // struct array keeps the ABI stable as roles are added.
        std::vector<gturbo::Tokenizer::ChatMessage> messages;
        gturbo::JsonValue arr = gturbo::JsonValue::parse(messages_json);
        if (!arr.is_array() || arr.array_value.empty()) {
            set_error("messages_json must be a non-empty JSON array");
            return nullptr;
        }
        for (const auto& m : arr.array_value) {
            if (!m.is_object()) { set_error("Each message must be an object"); return nullptr; }
            const std::string role = m.string_or("role", "");
            if (role.empty()) { set_error("Each message needs a 'role'"); return nullptr; }
            messages.push_back({role, m.string_or("content", "")});
        }

        gturbo::StreamCallback cb;
        if (on_token) {
            cb = [on_token, user](const gturbo::StreamEvent& ev) {
                if (ev.kind == gturbo::StreamEvent::Kind::Prefill) return true;
                return on_token(user, ev.index, ev.id, ev.delta.c_str()) != 0;
            };
        }

        auto result = e_ctx->runner->generate_chat(messages, g, cb);
        e_ctx->last_response = result.text;
        return e_ctx->last_response.c_str();
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return nullptr;
    }
}

TURBO_API int turbo_engine_last_stop_reason(void* handle) {
    if (!handle) return 3;
    auto e_ctx = static_cast<TurboEngineContext*>(handle);
    if (!e_ctx->runner) return 3;
    switch (e_ctx->runner->last_stop_reason()) {
        case gturbo::StopReason::EndOfSequence: return 0;
        case gturbo::StopReason::EndOfTurn:     return 1;
        case gturbo::StopReason::StopString:    return 2;
        case gturbo::StopReason::MaxTokens:     return 3;
        case gturbo::StopReason::Cancelled:     return 4;
    }
    return 3;
}

TURBO_API void turbo_engine_set_eviction_policy(void* handle, const char* policy) {
    if (!handle || !policy) return;
    auto e_ctx = static_cast<TurboEngineContext*>(handle);
    if (!e_ctx->runner) return;
    std::string p(policy);
    std::transform(p.begin(), p.end(), p.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (p == "lru")      e_ctx->runner->set_eviction_policy(gturbo::EvictionPolicy::LRU);
    else if (p == "lfu") e_ctx->runner->set_eviction_policy(gturbo::EvictionPolicy::LFU);
}

TURBO_API const char* turbo_engine_get_telemetry(void* handle) {
    if (!handle) return "{}";
    auto e_ctx = static_cast<TurboEngineContext*>(handle);

    gturbo::D3D12Context::SystemMemoryInfo sys_mem{};
    if (e_ctx->ctx) {
        sys_mem = e_ctx->ctx->query_memory_info();
    }

    gturbo::ModelMemoryUsage mod_mem{};
    gturbo::PerformanceMetrics perf{};
    std::string model_dir = "None";
    bool active = false;
    std::vector<int> active_experts;
    gturbo::EvictionPolicy policy = gturbo::EvictionPolicy::LFU;

    if (e_ctx->runner) {
        mod_mem = e_ctx->runner->get_memory_usage();
        perf = e_ctx->runner->metrics();
        model_dir = e_ctx->runner->model_dir();
        active_experts = e_ctx->runner->last_active_experts();
        policy = e_ctx->runner->eviction_policy();
        active = true;
    }

    std::ostringstream json;
    json << "{"
         << "\"status\":\"OK\","
         // Quoted, not interpolated: model_dir is an absolute Windows path, and raw
         // backslashes make the document invalid JSON for every consumer.
         << "\"gpu_name\":" << gturbo::JsonValue::quote(
                e_ctx->ctx ? e_ctx->ctx->adapter_name() : "No D3D12 device") << ","
         << "\"model_active\":" << (active ? "true" : "false") << ","
         << "\"model_dir\":" << gturbo::JsonValue::quote(model_dir) << ","
         << "\"memory\":{"
             << "\"resident_weights_mb\":" << (mod_mem.resident_weights_bytes / (1024.0 * 1024.0)) << ","
             << "\"kv_cache_mb\":" << (mod_mem.kv_cache_bytes / (1024.0 * 1024.0)) << ","
             << "\"expert_cache_mb\":" << (mod_mem.expert_cache_bytes / (1024.0 * 1024.0)) << ","
             << "\"total_model_ram_mb\":" << (mod_mem.total_model_bytes / (1024.0 * 1024.0)) << ","
             << "\"process_working_set_mb\":" << (sys_mem.process_working_set_bytes / (1024.0 * 1024.0)) << ","
             << "\"process_private_mb\":" << (sys_mem.process_private_bytes / (1024.0 * 1024.0)) << ","
             << "\"system_total_ram_gb\":" << (sys_mem.total_system_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << ","
             << "\"system_avail_ram_gb\":" << (sys_mem.avail_system_ram_bytes / (1024.0 * 1024.0 * 1024.0)) << ","
             << "\"gpu_vram_used_mb\":" << (sys_mem.gpu_dedicated_vram_used / (1024.0 * 1024.0)) << ","
             << "\"gpu_shared_used_mb\":" << (sys_mem.gpu_shared_ram_used / (1024.0 * 1024.0))
         << "},"
         << "\"performance\":{"
             << "\"prefill_toks_sec\":" << perf.prefill_tokens_per_sec << ","
             << "\"decode_toks_sec\":" << perf.decode_tokens_per_sec << ","
             << "\"total_time_ms\":" << perf.total_time_ms << ","
             << "\"total_io_mbs\":" << ((perf.total_time_ms > 0) ? (perf.total_io_bytes / (1024.0 * 1024.0) / (perf.total_time_ms / 1000.0)) : 0.0) << ","
             << "\"total_io_calls\":" << perf.total_io_calls
         << "},"
         << "\"cache\":{"
             << "\"hit_rate_pct\":" << mod_mem.cache_hit_rate_pct << ","
             << "\"eviction_policy\":\"" << (policy == gturbo::EvictionPolicy::LRU ? "LRU" : "LFU") << "\""
         << "},"
         << "\"active_experts\":[";
    for (size_t i = 0; i < active_experts.size(); ++i) {
        if (i > 0) json << ",";
        json << active_experts[i];
    }
    json << "]}";

    e_ctx->last_telemetry = json.str();
    return e_ctx->last_telemetry.c_str();
}

TURBO_API int turbo_engine_load_model(void* handle, const char* model_dir) {
    clear_error();
    if (!handle) {
        set_error("Null engine handle");
        return 0;
    }
    auto e_ctx = static_cast<TurboEngineContext*>(handle);
    try {
        std::string m_dir = (model_dir && strlen(model_dir) > 0) ? model_dir : "gemma-4-26b-a4b.gturbo";
        auto manifest = gturbo::GTurboManifestV1::from_json_string(
            gturbo::read_text_file(m_dir + "/manifest.json"));
        auto layout = gturbo::PackedExpertsLayoutV1::from_json_string(
            gturbo::read_text_file(m_dir + "/packed_experts/layout.json"));
        layout.cross_validate(manifest);

        auto runner = std::make_shared<gturbo::ForwardRunner>(e_ctx->ctx, manifest, layout, m_dir);
        runner->initialize();
        // Only swap in the new runner once it has loaded, so a failed load leaves the
        // previously working model in place rather than a half-initialized one.
        e_ctx->runner = std::move(runner);
        e_ctx->model_dir = m_dir;
        return 1;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return 0;
    }
}

TURBO_API void turbo_engine_unload_model(void* handle) {
    if (handle) {
        auto e_ctx = static_cast<TurboEngineContext*>(handle);
        e_ctx->runner = nullptr;
    }
}

TURBO_API void turbo_engine_clear_cache(void* handle) {
    if (handle) {
        auto e_ctx = static_cast<TurboEngineContext*>(handle);
        if (e_ctx->runner) e_ctx->runner->clear_expert_cache();
    }
}

TURBO_API void turbo_engine_stop(void* handle) {
    if (handle) {
        auto e_ctx = static_cast<TurboEngineContext*>(handle);
        if (e_ctx->runner) e_ctx->runner->stop_generation();
    }
}

// turbo_engine_repack() was removed. It overwrote the target directory in place with
// zero-filled placeholder files whose resident header claimed 2.85 GB of weights, which is
// how the checked-in placeholder bundle came to exist. Real repacking streams ~14.6 GB from
// the pinned HuggingFace checkpoint and lives in tools/convert_hf_to_gturbo.py.

} // extern "C"
