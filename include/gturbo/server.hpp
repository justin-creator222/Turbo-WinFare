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

    void set_context(std::shared_ptr<D3D12Context> ctx) { ctx_ = ctx; }
    std::shared_ptr<ForwardRunner> runner() const { return runner_; }

    // Reason the model failed to load at startup, surfaced through /api/telemetry so the
    // GUI can explain why it shows NO MODEL instead of leaving the user guessing.
    void set_load_error(const std::string& message) { load_error_ = message; }

    void set_openai_config(const OpenAIServerConfig& cfg) { openai_cfg_ = cfg; }
    const OpenAIServerConfig& openai_config() const { return openai_cfg_; }

private:
    void listen_loop(uint16_t port);
    void handle_client(uintptr_t client_socket);

    // OpenAI-compatible endpoints. Return true when the path was theirs.
    bool handle_openai(uintptr_t client, const HttpRequest& req);
    void handle_chat_completion(uintptr_t client, const HttpRequest& req);

    std::shared_ptr<ForwardRunner> runner_;
    std::shared_ptr<D3D12Context> ctx_;

    ServerConfig config_;
    mutable std::mutex config_mutex_;

    // One generation at a time. The runner owns a single set of GPU scratch buffers and a
    // single KV cache, so two concurrent /api/generate calls interleave inside produce_token
    // and corrupt each other's activations -- the output stays fluent, it is just wrong.
    // A queue with a bounded backlog replaces this when the OpenAI server lands.
    std::mutex generate_mutex_;

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
