#pragma once

#include "gturbo/runner.hpp"
#include "gturbo/openai_api.hpp"
#include "gturbo/http.hpp"
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>

namespace gturbo {

// Generation knobs the GUI can change at runtime. Anything that only takes effect on a
// reload is reported back as such rather than silently accepted.
struct ServerConfig {
    float temperature{0.2f};
    float top_p{0.95f};
    int top_k{64};
    int max_tokens{512};
    // Applied immediately.
    EvictionPolicy eviction_policy{EvictionPolicy::LFU};
    // Require re-initialization; stored so the GUI can show what a reload would use.
    int context_len{0};
    int slots{0};
};

class HTTPServer {
public:
    HTTPServer();
    ~HTTPServer();

    void start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<D3D12Context> ctx = nullptr);
    void stop();
    bool is_running() const { return is_running_; }

    void set_context(std::shared_ptr<D3D12Context> ctx);
    // Returns a COPY. Callers must hold their own reference for the whole time they use the
    // runner -- a reload can republish runner_ at any moment, and dereferencing the member
    // directly is what let a reload destroy a ForwardRunner out from under a generation.
    std::shared_ptr<ForwardRunner> runner() const { return current_runner(); }

    // Reason the model failed to load, surfaced through /api/telemetry so the GUI can
    // explain why it shows NO MODEL instead of leaving the user guessing.
    void set_load_error(const std::string& message);

    void set_openai_config(const OpenAIServerConfig& cfg);
    // By value: the reload path rewrites max_context, so handing out a reference would let
    // a caller read it while it is being written.
    OpenAIServerConfig openai_config() const;

    // Startup configuration the GUI's reload path should inherit. Without this, launching
    // with `--slots 44 --gui` and then reloading from the GUI silently drops back to
    // RAM auto-sizing, because ServerConfig starts at the 0 = auto sentinel.
    void set_initial_engine_config(int context_len, int slots);

private:
    void listen_loop(uint16_t port);
    void handle_client(uintptr_t client_socket);

    // OpenAI-compatible endpoints. Return true when the path was theirs.
    bool handle_openai(uintptr_t client, const HttpRequest& req);
    void handle_chat_completion(uintptr_t client, const HttpRequest& req);

    // Snapshot accessors. Every handler must go through these rather than touching runner_
    // or ctx_ directly -- see the locking rules on runner_mutex_ below.
    std::shared_ptr<ForwardRunner> current_runner() const;
    std::shared_ptr<D3D12Context> current_ctx() const;

    // Shared implementation of /api/load_model and /api/unload_model. Returns false and
    // fills `error` when a generation is in flight, so both endpoints answer 409 rather
    // than tearing the runner down underneath it.
    bool swap_runner(const std::string& model_dir, bool load, std::string& error);

    std::shared_ptr<ForwardRunner> runner_;
    std::shared_ptr<D3D12Context> ctx_;

    ServerConfig config_;
    mutable std::mutex config_mutex_;

    // One generation at a time. The runner owns a single set of GPU scratch buffers and a
    // single KV cache, so two concurrent generations interleave inside produce_token and
    // corrupt each other's activations -- the output stays fluent, it is just wrong.
    //
    // EVERY generation entry point must hold this: /api/generate AND /v1/chat/completions.
    // The /v1 path used to rely on RequestCoordinator alone, which serializes only among
    // /v1 callers -- so a GUI request and an OpenAI request could run concurrently on the
    // same runner. Both now funnel through with_generation_lock() so the rule is structural
    // rather than a convention each new endpoint has to remember.
    std::mutex generate_mutex_;

    // Guards runner_, ctx_, load_error_ and openai_cfg_.
    //
    // Never held across a call INTO ForwardRunner: handlers snapshot the shared_ptr and
    // release the lock first. In particular a reload runs initialize() -- multi-GB
    // allocation and file I/O -- while holding only generate_mutex_, so /api/telemetry
    // keeps answering during a load.
    //
    // Lock order, never reversed:
    //     coordinator lease -> generate_mutex_ -> runner_mutex_ -> config_mutex_
    //
    // INVARIANT: runner_ is reassigned ONLY while generate_mutex_ is held. That is what
    // makes /api/stop correct without taking generate_mutex_ itself (which would deadlock
    // against the very generation it is cancelling): there can never be an in-flight
    // generation on a runner that is no longer the published one.
    mutable std::mutex runner_mutex_;

    OpenAIServerConfig openai_cfg_;
    // Bounded admission for the /v1 endpoints, so a burst queues fairly and then 429s
    // instead of piling up without limit.
    std::unique_ptr<RequestCoordinator> coordinator_;

    std::string load_error_;
    std::atomic<bool> is_running_{false};
    std::thread server_thread_;
    uintptr_t listen_socket_{0};
};

} // namespace gturbo
