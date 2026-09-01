#include "gturbo/openai_api.hpp"

#include <algorithm>
#include <sstream>

namespace gturbo {

namespace {

// Mirrors OpenAIRequestValidator in the reference. Defaults match its documented values.
constexpr float kDefaultTemperature = 0.2f;
constexpr float kDefaultTopP = 0.95f;
constexpr int   kDefaultTopK = 64;
constexpr int   kDefaultMaxTokens = 4096;
constexpr size_t kMaxStopStrings = 4;

bool fail(ApiError& err, int status, std::string message,
          std::string param = {}, std::string code = {}, std::string type = "invalid_request_error") {
    err.status = status;
    err.message = std::move(message);
    err.param = std::move(param);
    err.code = std::move(code);
    err.type = std::move(type);
    return false;
}

// Content is either a plain string or an array of {type:"text", text:"..."} parts.
bool extract_content(const JsonValue& msg, size_t index, std::string& out, ApiError& err) {
    if (!msg.has("content")) { out.clear(); return true; }
    const JsonValue& c = msg.object_value.at("content");

    if (c.type == JsonValue::Type::String) { out = c.string_value; return true; }
    if (c.is_null()) { out.clear(); return true; }

    if (c.is_array()) {
        std::string joined;
        for (const auto& part : c.array_value) {
            if (!part.is_object()) {
                return fail(err, 400, "Message content parts must be objects.",
                            "messages[" + std::to_string(index) + "].content");
            }
            const std::string ptype = part.string_or("type", "");
            if (ptype != "text") {
                // Images and audio are out of scope; saying so beats silently dropping them.
                return fail(err, 400,
                            "Only text content parts are supported; got '" + ptype + "'.",
                            "messages[" + std::to_string(index) + "].content",
                            "unsupported_value");
            }
            joined += part.string_or("text", "");
        }
        out = std::move(joined);
        return true;
    }
    return fail(err, 400, "Message content must be a string or an array of text parts.",
                "messages[" + std::to_string(index) + "].content");
}

} // namespace

const char* finish_reason_for(StopReason reason) {
    switch (reason) {
        case StopReason::MaxTokens: return "length";
        // A stop string, an end-of-turn marker, EOS, and a client disconnect all present to
        // the caller as a normally-finished completion.
        case StopReason::EndOfSequence:
        case StopReason::EndOfTurn:
        case StopReason::StopString:
        case StopReason::Cancelled:
        default: return "stop";
    }
}

bool validate_chat_request(const JsonValue& body, const OpenAIServerConfig& cfg,
                           ValidatedChatRequest& out, ApiError& err) {
    // Start clean. This function appends messages and stop strings, so a caller reusing the
    // same struct across requests would otherwise accumulate the previous conversation into
    // the next one -- which would look like the model spontaneously remembering things.
    out = ValidatedChatRequest{};

    if (!body.is_object()) {
        return fail(err, 400, "Request body must be a JSON object.");
    }

    // --- model ---------------------------------------------------------------------
    out.model = body.string_or("model", "");
    if (out.model.empty()) {
        return fail(err, 400, "Missing required parameter: 'model'.", "model");
    }
    if (out.model != cfg.model_id) {
        // This server hosts exactly one model; there is no remote switching.
        return fail(err, 404, "The model '" + out.model + "' does not exist.", "model",
                    "model_not_found");
    }

    // --- unsupported parameters, rejected rather than ignored -----------------------
    if (body.has("n") && body.int_or("n", 1) != 1) {
        return fail(err, 400, "Only n=1 is supported.", "n", "unsupported_value");
    }
    if (body.bool_or("logprobs", false)) {
        return fail(err, 400, "logprobs is not supported.", "logprobs", "unsupported_value");
    }
    if (body.double_or("presence_penalty", 0.0) != 0.0) {
        return fail(err, 400, "presence_penalty must be 0.", "presence_penalty",
                    "unsupported_value");
    }
    if (body.double_or("frequency_penalty", 0.0) != 0.0) {
        return fail(err, 400, "frequency_penalty must be 0.", "frequency_penalty",
                    "unsupported_value");
    }
    if (body.has("parallel_tool_calls") && !body.bool_or("parallel_tool_calls", true)) {
        return fail(err, 400, "parallel_tool_calls=false is not supported.",
                    "parallel_tool_calls", "unsupported_value");
    }
    if (body.has("tool_choice")) {
        const JsonValue& tc = body.object_value.at("tool_choice");
        const std::string mode = (tc.type == JsonValue::Type::String) ? tc.string_value : "named";
        if (mode != "auto" && mode != "none") {
            return fail(err, 400, "tool_choice must be 'auto' or 'none'.", "tool_choice",
                        "unsupported_value");
        }
    }

    // --- messages -------------------------------------------------------------------
    if (!body.has("messages")) {
        return fail(err, 400, "Missing required parameter: 'messages'.", "messages");
    }
    const JsonValue& msgs = body.object_value.at("messages");
    if (!msgs.is_array() || msgs.array_value.empty()) {
        return fail(err, 400, "'messages' must be a non-empty array.", "messages");
    }

    bool seen_conversation = false;
    for (size_t i = 0; i < msgs.array_value.size(); ++i) {
        const JsonValue& m = msgs.array_value[i];
        if (!m.is_object()) {
            return fail(err, 400, "Each message must be an object.",
                        "messages[" + std::to_string(i) + "]");
        }
        std::string role = m.string_or("role", "");
        if (role.empty()) {
            return fail(err, 400, "Each message needs a 'role'.",
                        "messages[" + std::to_string(i) + "].role");
        }
        if (role != "system" && role != "developer" && role != "user" &&
            role != "assistant" && role != "tool") {
            return fail(err, 400, "Unknown role '" + role + "'.",
                        "messages[" + std::to_string(i) + "].role", "unsupported_value");
        }
        // System instructions must precede the conversation; a system turn in the middle
        // renders into a template position the model was never trained on.
        if (role == "system" || role == "developer") {
            if (seen_conversation) {
                return fail(err, 400,
                            "System and developer messages must precede the conversation.",
                            "messages[" + std::to_string(i) + "].role");
            }
            role = "system";
        } else {
            seen_conversation = true;
        }

        std::string content;
        if (!extract_content(m, i, content, err)) return false;
        out.messages.push_back({role, content});
    }

    // --- sampling -------------------------------------------------------------------
    GenerationOptions& o = out.options;
    o.temperature = static_cast<float>(body.double_or("temperature", kDefaultTemperature));
    if (o.temperature < 0.0f || o.temperature > 2.0f) {
        return fail(err, 400, "temperature must be in 0..2.", "temperature");
    }
    o.top_p = static_cast<float>(body.double_or("top_p", kDefaultTopP));
    if (o.top_p <= 0.0f || o.top_p > 1.0f) {
        return fail(err, 400, "top_p must be in (0, 1].", "top_p");
    }
    o.top_k = static_cast<int>(body.int_or("top_k", kDefaultTopK));
    if (o.top_k < 1 || o.top_k > 256) {
        return fail(err, 400, "top_k must be in 1..256.", "top_k");
    }
    o.repetition_penalty = static_cast<float>(body.double_or("repetition_penalty", 1.0));
    if (o.repetition_penalty <= 0.0f) {
        return fail(err, 400, "repetition_penalty must be > 0.", "repetition_penalty");
    }
    if (body.has("seed")) {
        o.has_seed = true;
        o.seed = static_cast<uint64_t>(body.int_or("seed", 0));
    }

    // max_completion_tokens is the current spelling; max_tokens is the legacy one.
    int64_t max_tokens = body.int_or("max_completion_tokens",
                                     body.int_or("max_tokens", kDefaultMaxTokens));
    if (max_tokens <= 0) {
        return fail(err, 400, "max_tokens must be greater than 0.", "max_tokens");
    }
    o.max_tokens = static_cast<int>(std::min<int64_t>(max_tokens, cfg.max_context));

    // --- stop: a string or an array of at most four ----------------------------------
    if (const std::string stop_err = parse_stop_field(body, kMaxStopStrings, o.stop_strings);
        !stop_err.empty()) {
        return fail(err, 400, stop_err, "stop");
    }

    out.stream = body.bool_or("stream", false);
    if (body.has("stream_options")) {
        const JsonValue& so = body.object_value.at("stream_options");
        if (so.is_object()) out.include_usage = so.bool_or("include_usage", false);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Rendering

std::string parse_stop_field(const JsonValue& body, size_t max_stops,
                             std::vector<std::string>& out) {
    if (!body.has("stop")) return "";
    const JsonValue& s = body.object_value.at("stop");
    if (s.type == JsonValue::Type::String) {
        out.push_back(s.string_value);
        return "";
    }
    if (s.is_array()) {
        if (s.array_value.size() > max_stops) {
            return "At most " + std::to_string(max_stops) + " stop sequences are supported.";
        }
        for (const auto& e : s.array_value) {
            if (e.type != JsonValue::Type::String) {
                return "Stop sequences must be strings.";
            }
            out.push_back(e.string_value);
        }
        return "";
    }
    if (s.is_null()) return "";
    return "'stop' must be a string or an array of strings.";
}

std::string render_error(const ApiError& err) {
    std::ostringstream o;
    o << "{\"error\":{"
      << "\"message\":" << JsonValue::quote(err.message) << ","
      << "\"type\":" << JsonValue::quote(err.type) << ",";
    if (err.param.empty()) o << "\"param\":null,";
    else                   o << "\"param\":" << JsonValue::quote(err.param) << ",";
    if (err.code.empty())  o << "\"code\":null";
    else                   o << "\"code\":" << JsonValue::quote(err.code);
    o << "}}";
    return o.str();
}

std::string render_completion(const std::string& id, const std::string& model,
                              const GenerationResult& result, int64_t created) {
    std::ostringstream o;
    o << "{\"id\":" << JsonValue::quote(id) << ","
      << "\"object\":\"chat.completion\","
      << "\"created\":" << created << ","
      << "\"model\":" << JsonValue::quote(model) << ","
      << "\"choices\":[{"
      << "\"index\":0,"
      << "\"message\":{\"role\":\"assistant\",\"content\":" << JsonValue::quote(result.text) << "},"
      << "\"finish_reason\":\"" << finish_reason_for(result.reason) << "\""
      << "}],"
      << "\"usage\":{"
      << "\"prompt_tokens\":" << result.prompt_tokens << ","
      << "\"completion_tokens\":" << result.completion_tokens << ","
      << "\"total_tokens\":" << (result.prompt_tokens + result.completion_tokens) << ","
      // Always 0 until the prompt cache lands; reporting the field keeps clients that read
      // it from having to special-case its absence.
      << "\"prompt_tokens_details\":{\"cached_tokens\":0}"
      << "}}";
    return o.str();
}

std::string render_chunk(const std::string& id, const std::string& model, int64_t created,
                         const std::string& delta, bool role_only) {
    std::ostringstream o;
    o << "{\"id\":" << JsonValue::quote(id) << ","
      << "\"object\":\"chat.completion.chunk\","
      << "\"created\":" << created << ","
      << "\"model\":" << JsonValue::quote(model) << ","
      << "\"choices\":[{\"index\":0,\"delta\":";
    if (role_only) o << "{\"role\":\"assistant\"}";
    else           o << "{\"content\":" << JsonValue::quote(delta) << "}";
    o << ",\"finish_reason\":null}]}";
    return o.str();
}

std::string render_final_chunk(const std::string& id, const std::string& model, int64_t created,
                               const GenerationResult& result) {
    std::ostringstream o;
    o << "{\"id\":" << JsonValue::quote(id) << ","
      << "\"object\":\"chat.completion.chunk\","
      << "\"created\":" << created << ","
      << "\"model\":" << JsonValue::quote(model) << ","
      << "\"choices\":[{\"index\":0,\"delta\":{},"
      << "\"finish_reason\":\"" << finish_reason_for(result.reason) << "\"}],"
      << "\"x_turbo\":{"
      << "\"stop_reason\":" << JsonValue::quote(stop_reason_name(result.reason)) << ","
      << "\"matched_stop\":"
      << (result.matched_stop.empty() ? "null" : JsonValue::quote(result.matched_stop)) << ","
      << "\"ttft_ms\":" << result.ttft_ms
      << "}}";
    return o.str();
}

std::string render_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                               const GenerationResult& result) {
    std::ostringstream o;
    o << "{\"id\":" << JsonValue::quote(id) << ","
      << "\"object\":\"chat.completion.chunk\","
      << "\"created\":" << created << ","
      << "\"model\":" << JsonValue::quote(model) << ","
      << "\"choices\":[],"
      << "\"usage\":{"
      << "\"prompt_tokens\":" << result.prompt_tokens << ","
      << "\"completion_tokens\":" << result.completion_tokens << ","
      << "\"total_tokens\":" << (result.prompt_tokens + result.completion_tokens) << ","
      << "\"prompt_tokens_details\":{\"cached_tokens\":0}"
      << "}}";
    return o.str();
}

std::string render_models_list(const std::string& model_id) {
    std::ostringstream o;
    o << "{\"object\":\"list\",\"data\":[{"
      << "\"id\":" << JsonValue::quote(model_id) << ","
      << "\"object\":\"model\","
      << "\"created\":0,"
      << "\"owned_by\":\"turbowinfare\"}]}";
    return o.str();
}

// ---------------------------------------------------------------------------

std::optional<RequestCoordinator::Lease> RequestCoordinator::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);

    // Admission is checked before the caller does any work, so an over-capacity request is
    // rejected before its prompt is tokenized rather than after.
    if (busy_ && static_cast<int>(queue_.size()) >= queue_limit_) {
        return std::nullopt;
    }

    const uint64_t ticket = next_ticket_++;
    queue_.push_back(ticket);
    // Strict FIFO: wait until nothing is running AND this ticket is at the head.
    cv_.wait(lock, [&] { return !busy_ && !queue_.empty() && queue_.front() == ticket; });
    queue_.pop_front();
    busy_ = true;
    return Lease(this);
}

void RequestCoordinator::release() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
    }
    // Every waiter must re-check, because only the ticket at the head may proceed.
    cv_.notify_all();
}

} // namespace gturbo
