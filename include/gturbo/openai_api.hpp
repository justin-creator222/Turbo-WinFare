#pragma once

// OpenAI-compatible chat completions, matching the subset the Swift reference implements
// (docs/OPENAI_SERVER.md). This is what lets external clients -- the `openai` Python SDK,
// OpenCode, anything speaking the API -- talk to the engine.
//
// Deliberately NOT supported, and rejected explicitly rather than silently ignored: the
// Responses API, legacy completions, embeddings, multimodal input, logprobs, `n != 1`,
// presence/frequency penalties, and `tool_choice: required`. Silently accepting a parameter
// the engine does not honour is worse than a 400, because the caller believes it applied.

#include "gturbo/json.hpp"
#include "gturbo/runner.hpp"
#include "gturbo/tokenizer.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gturbo {

struct ApiError {
    int status{400};
    std::string message;
    std::string type{"invalid_request_error"};
    std::string param;
    std::string code;
};

struct ValidatedChatRequest {
    std::vector<Tokenizer::ChatMessage> messages;
    GenerationOptions options;
    bool stream{false};
    bool include_usage{false};
    std::string model;
};

// Server-wide settings for the OpenAI endpoints.
struct OpenAIServerConfig {
    std::string model_id{"gemma-4-26b-a4b-it"};
    int max_context{4096};
    int queue_limit{4};
};

// Returns false and fills `err` on any rejection. `manifest_vocab` is unused today but keeps
// the signature stable for token-budget checks.
bool validate_chat_request(const JsonValue& body, const OpenAIServerConfig& cfg,
                           ValidatedChatRequest& out, ApiError& err);

std::string render_error(const ApiError& err);
std::string render_completion(const std::string& id, const std::string& model,
                              const GenerationResult& result, int64_t created);
// One SSE data payload. `role_only` emits the opening delta that carries the assistant role.
std::string render_chunk(const std::string& id, const std::string& model, int64_t created,
                         const std::string& delta, bool role_only);
std::string render_final_chunk(const std::string& id, const std::string& model, int64_t created,
                               StopReason reason);
std::string render_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                               const GenerationResult& result);
std::string render_models_list(const std::string& model_id);

const char* finish_reason_for(StopReason reason);

// Serializes generation and bounds the backlog.
//
// The engine has one set of GPU scratch buffers and one KV cache, so exactly one generation
// can be in flight. Without admission control a burst of requests would either corrupt each
// other or queue without limit; the reference bounds the wait list and returns 429 beyond it.
class RequestCoordinator {
public:
    explicit RequestCoordinator(int queue_limit) : queue_limit_(queue_limit) {}

    class Lease {
    public:
        Lease() = default;
        explicit Lease(RequestCoordinator* c) : owner_(c) {}
        ~Lease() { if (owner_) owner_->release(); }
        Lease(Lease&& o) noexcept : owner_(o.owner_) { o.owner_ = nullptr; }
        Lease& operator=(Lease&& o) noexcept {
            if (this != &o) { if (owner_) owner_->release(); owner_ = o.owner_; o.owner_ = nullptr; }
            return *this;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        RequestCoordinator* owner_{nullptr};
    };

    // Blocks until it is this caller's turn. Returns nullopt immediately when the wait list
    // is already at `queue_limit`, which the caller turns into 429 queue_full.
    std::optional<Lease> acquire();

    int waiting() const {
        std::lock_guard<std::mutex> g(mutex_);
        return static_cast<int>(queue_.size());
    }

private:
    void release();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    // Explicit FIFO. condition_variable wake order is unspecified, so relying on notify_one
    // to be fair would make service order depend on the scheduler.
    std::deque<uint64_t> queue_;
    uint64_t next_ticket_{0};
    bool busy_{false};
    int queue_limit_{4};
};

} // namespace gturbo
