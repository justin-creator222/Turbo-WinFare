// OpenAI-compatible request validation, response rendering, and request admission.
//
// The validator's job is to reject rather than ignore. Silently accepting `n: 3` or
// `presence_penalty: 0.5` is worse than a 400, because the caller believes the parameter
// applied and the difference only shows up as unexplained output quality.

#include "gturbo/openai_api.hpp"
#include "gturbo/json.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace gturbo;

namespace {

OpenAIServerConfig cfg() {
    OpenAIServerConfig c;
    c.model_id = "gemma-4-26b-a4b-it";
    c.max_context = 4096;
    c.queue_limit = 4;
    return c;
}

bool accepts(const std::string& json, ValidatedChatRequest& out) {
    ApiError err;
    return validate_chat_request(JsonValue::parse(json), cfg(), out, err);
}

ApiError rejection(const std::string& json) {
    ValidatedChatRequest out;
    ApiError err;
    const bool ok = validate_chat_request(JsonValue::parse(json), cfg(), out, err);
    assert(!ok && "expected this request to be rejected");
    return err;
}

const char* kMinimal =
    R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}]})";

} // namespace

int main() {
    std::cout << "[TEST] Running OpenAI API tests...\n";

    // ---- A minimal valid request, and the documented defaults ----------------------
    {
        ValidatedChatRequest r;
        assert(accepts(kMinimal, r));
        assert(r.messages.size() == 1);
        assert(r.messages[0].role == "user" && r.messages[0].content == "hi");
        assert(!r.stream);
        // Defaults must match the reference: temperature 0.2, top-p 0.95, top-k 64.
        assert(r.options.temperature == 0.2f);
        assert(r.options.top_p == 0.95f);
        assert(r.options.top_k == 64);
        assert(r.options.repetition_penalty == 1.0f);
        std::cout << "  [PASS] Minimal request accepted with the reference's defaults.\n";
    }

    // ---- Model routing ---------------------------------------------------------------
    {
        auto e = rejection(R"({"model":"gpt-4","messages":[{"role":"user","content":"hi"}]})");
        assert(e.status == 404 && e.code == "model_not_found" && e.param == "model");

        e = rejection(R"({"messages":[{"role":"user","content":"hi"}]})");
        assert(e.status == 400 && e.param == "model");
        std::cout << "  [PASS] Unknown model -> 404 model_not_found; missing model -> 400.\n";
    }

    // ---- Messages --------------------------------------------------------------------
    {
        auto e = rejection(R"({"model":"gemma-4-26b-a4b-it","messages":[]})");
        assert(e.status == 400 && e.param == "messages");

        e = rejection(R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"wizard","content":"x"}]})");
        assert(e.status == 400 && e.code == "unsupported_value");

        // A system message after the conversation has begun renders into a template position
        // the model was never trained on, so it is rejected rather than silently reordered.
        e = rejection(R"({"model":"gemma-4-26b-a4b-it","messages":[)"
                      R"({"role":"user","content":"hi"},{"role":"system","content":"be brief"}]})");
        assert(e.status == 400);
        std::cout << "  [PASS] Empty/unknown/misplaced messages rejected.\n";
    }

    // ---- Content as an array of text parts --------------------------------------------
    {
        ValidatedChatRequest r;
        assert(accepts(R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user",)"
                       R"("content":[{"type":"text","text":"one "},{"type":"text","text":"two"}]}]})", r));
        assert(r.messages[0].content == "one two");

        // Images are out of scope and must say so.
        auto e = rejection(R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user",)"
                           R"("content":[{"type":"image_url","image_url":{"url":"x"}}]}]})");
        assert(e.status == 400 && e.code == "unsupported_value");
        std::cout << "  [PASS] Text content parts joined; image parts rejected.\n";
    }

    // ---- Unsupported parameters are rejected, not ignored -----------------------------
    {
        const std::string base = R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}],)";
        for (const auto& [frag, param] : std::vector<std::pair<std::string, std::string>>{
                 {R"("n":2})", "n"},
                 {R"("logprobs":true})", "logprobs"},
                 {R"("presence_penalty":0.5})", "presence_penalty"},
                 {R"("frequency_penalty":0.5})", "frequency_penalty"},
                 {R"("parallel_tool_calls":false})", "parallel_tool_calls"},
                 {R"("tool_choice":"required"})", "tool_choice"}}) {
            auto e = rejection(base + frag);
            assert(e.status == 400 && e.code == "unsupported_value");
            assert(e.param == param);
        }
        // But n=1 and tool_choice auto/none are fine.
        ValidatedChatRequest r;
        assert(accepts(base + R"("n":1,"tool_choice":"auto"})", r));
        std::cout << "  [PASS] n!=1, logprobs, penalties, tool_choice:required all rejected.\n";
    }

    // ---- Sampling ranges ---------------------------------------------------------------
    {
        const std::string base = R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}],)";
        assert(rejection(base + R"("temperature":-1})").param == "temperature");
        assert(rejection(base + R"("temperature":2.5})").param == "temperature");
        assert(rejection(base + R"("top_p":0})").param == "top_p");
        assert(rejection(base + R"("top_p":1.5})").param == "top_p");
        assert(rejection(base + R"("top_k":0})").param == "top_k");
        assert(rejection(base + R"("top_k":257})").param == "top_k");
        assert(rejection(base + R"("max_tokens":0})").param == "max_tokens");
        assert(rejection(base + R"("repetition_penalty":0})").param == "repetition_penalty");

        // max_tokens is clamped to the context rather than rejected.
        ValidatedChatRequest r;
        assert(accepts(base + R"("max_tokens":999999})", r));
        assert(r.options.max_tokens == 4096);

        // max_completion_tokens is the current spelling and wins over the legacy one.
        assert(accepts(base + R"("max_completion_tokens":11,"max_tokens":99})", r));
        assert(r.options.max_tokens == 11);
        std::cout << "  [PASS] Sampling ranges enforced; max_tokens clamped to context.\n";
    }

    // ---- Stop sequences -----------------------------------------------------------------
    {
        ValidatedChatRequest r;
        const std::string base = R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}],)";
        assert(accepts(base + R"("stop":"END"})", r));
        assert(r.options.stop_strings.size() == 1 && r.options.stop_strings[0] == "END");

        assert(accepts(base + R"("stop":["a","b","c","d"]})", r));
        assert(r.options.stop_strings.size() == 4);

        assert(rejection(base + R"("stop":["a","b","c","d","e"]})").param == "stop");
        std::cout << "  [PASS] Stop as string or <=4 array; a 5th is rejected.\n";
    }

    // ---- Streaming flags -----------------------------------------------------------------
    {
        ValidatedChatRequest r;
        const std::string base = R"({"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"hi"}],)";
        assert(accepts(base + R"("stream":true,"stream_options":{"include_usage":true}})", r));
        assert(r.stream && r.include_usage);

        assert(accepts(base + R"("stream":true})", r));
        assert(r.stream && !r.include_usage);
        std::cout << "  [PASS] stream and stream_options.include_usage parsed.\n";
    }

    // ---- Rendering re-parses as valid JSON -------------------------------------------
    {
        // A delta containing quotes and newlines must not break the SSE payload -- this is
        // exactly what the old hand-rolled string building got wrong.
        const std::string nasty = "he said \"hi\"\nthen left";
        JsonValue chunk = JsonValue::parse(render_chunk("id-1", "m", 42, nasty, false));
        const JsonValue& delta = chunk.at("choices", "c").at(0, "c0").at("delta", "d");
        assert(delta.string_or("content", "") == nasty);
        assert(chunk.string_or("object", "") == "chat.completion.chunk");

        JsonValue role = JsonValue::parse(render_chunk("id-1", "m", 42, "", true));
        assert(role.at("choices", "c").at(0, "c0").at("delta", "d").string_or("role", "") ==
               "assistant");

        GenerationResult res;
        res.text = nasty;
        res.prompt_tokens = 7;
        res.completion_tokens = 5;
        res.reason = StopReason::EndOfTurn;
        JsonValue done = JsonValue::parse(render_completion("id-1", "m", res, 42));
        assert(done.at("choices", "c").at(0, "c0").string_or("finish_reason", "") == "stop");
        const JsonValue& usage = done.at("usage", "u");
        assert(usage.int_or("prompt_tokens", -1) == 7);
        assert(usage.int_or("completion_tokens", -1) == 5);
        assert(usage.int_or("total_tokens", -1) == 12);

        res.reason = StopReason::MaxTokens;
        JsonValue cut = JsonValue::parse(render_completion("id-1", "m", res, 42));
        assert(cut.at("choices", "c").at(0, "c0").string_or("finish_reason", "") == "length");

        JsonValue e = JsonValue::parse(render_error({404, "nope", "invalid_request_error",
                                                     "model", "model_not_found"}));
        assert(e.at("error", "e").string_or("code", "") == "model_not_found");

        JsonValue models = JsonValue::parse(render_models_list("gemma-4-26b-a4b-it"));
        assert(models.at("data", "d").at(0, "d0").string_or("owned_by", "") == "turbowinfare");
        std::cout << "  [PASS] Rendered completions/chunks/errors re-parse correctly.\n";
    }

    // ---- finish_reason mapping ---------------------------------------------------------
    {
        assert(std::string(finish_reason_for(StopReason::MaxTokens)) == "length");
        for (auto r : {StopReason::EndOfSequence, StopReason::EndOfTurn,
                       StopReason::StopString, StopReason::Cancelled}) {
            assert(std::string(finish_reason_for(r)) == "stop");
        }
        std::cout << "  [PASS] Only MaxTokens maps to 'length'.\n";
    }

    // ---- Coordinator: one at a time, bounded backlog, FIFO -----------------------------
    {
        RequestCoordinator c(2);
        auto first = c.acquire();
        assert(first.has_value());

        // Two may wait; the third is refused rather than queued without limit.
        std::atomic<int> admitted{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < 2; ++i) {
            waiters.emplace_back([&] {
                auto lease = c.acquire();
                if (lease) admitted.fetch_add(1);
            });
        }
        // Let both register as waiting before testing the limit.
        for (int spin = 0; spin < 200 && c.waiting() < 2; ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(c.waiting() == 2);
        assert(!c.acquire().has_value() && "third waiter should have been refused");

        first.reset();                       // release, letting the queue drain
        for (auto& t : waiters) t.join();
        assert(admitted.load() == 2);
        std::cout << "  [PASS] Coordinator admits one, queues to the limit, then refuses.\n";
    }

    {
        // Strict FIFO. condition_variable wake order is unspecified, so this asserts the
        // explicit ticket queue is doing the ordering rather than the scheduler.
        RequestCoordinator c(16);
        auto held = c.acquire();
        assert(held.has_value());

        std::mutex order_mutex;
        std::vector<int> order;
        std::vector<std::thread> threads;
        for (int i = 0; i < 8; ++i) {
            // Start each thread and wait for it to be queued, so arrival order is known.
            threads.emplace_back([&, i] {
                auto lease = c.acquire();
                std::lock_guard<std::mutex> g(order_mutex);
                order.push_back(i);
            });
            for (int spin = 0; spin < 200 && c.waiting() < i + 1; ++spin) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        held.reset();
        for (auto& t : threads) t.join();

        assert(order.size() == 8);
        for (int i = 0; i < 8; ++i) {
            assert(order[static_cast<size_t>(i)] == i && "coordinator did not serve in FIFO order");
        }
        std::cout << "  [PASS] Coordinator serves 8 waiters in strict FIFO order.\n";
    }

    std::cout << "[TEST] All OpenAI API tests passed.\n";
    return 0;
}
