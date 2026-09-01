#include <winsock2.h>
#include <ws2tcpip.h>
#include "gturbo/server.hpp"
#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/json.hpp"
#include "gturbo/sampling.hpp"
#include "gturbo/http.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <algorithm>

#pragma comment(lib, "ws2_32.lib")
namespace fs = std::filesystem;

namespace gturbo {

HTTPServer::HTTPServer() {}

HTTPServer::~HTTPServer() {
    stop();
}

std::shared_ptr<ForwardRunner> HTTPServer::current_runner() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return runner_;
}

std::shared_ptr<D3D12Context> HTTPServer::current_ctx() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return ctx_;
}

void HTTPServer::set_context(std::shared_ptr<D3D12Context> ctx) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    ctx_ = ctx;
}

void HTTPServer::set_load_error(const std::string& message) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    load_error_ = message;
}

const HostEnvironment& HTTPServer::host_environment() const {
    std::lock_guard<std::mutex> guard(env_mutex_);
    if (!env_probed_) {
        host_env_ = probe_host_environment();
        env_probed_ = true;
    }
    return host_env_;
}

std::string HTTPServer::load_error() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return load_error_;
}

void HTTPServer::set_startup_info(uint16_t port, const std::string& host, bool serve_mode) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    startup_port_ = port;
    bind_address_ = host;
    serve_mode_ = serve_mode;
}

void HTTPServer::set_openai_config(const OpenAIServerConfig& cfg) {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    openai_cfg_ = cfg;
}

OpenAIServerConfig HTTPServer::openai_config() const {
    std::lock_guard<std::mutex> guard(runner_mutex_);
    return openai_cfg_;
}

void HTTPServer::set_initial_engine_config(int context_len, int slots) {
    std::lock_guard<std::mutex> guard(config_mutex_);
    config_.context_len = context_len;
    config_.slots = slots;
}

// Replaces the live runner, or tears it down when `load` is false.
//
// The ordering here is load-bearing:
//
//  1. try_lock generate_mutex_ and hold it for the whole swap. This is what upholds the
//     "runner_ is only republished while generate_mutex_ is held" invariant, and it is why
//     /api/stop can skip the lock safely. try_lock rather than a blocking lock because a
//     generation can run for minutes and the GUI polls with no timeout handling.
//  2. Release the OLD runner and flush the GPU BEFORE building the new one. This gives up
//     the previous "a failed load leaves the old model in place" behaviour on purpose:
//     holding both alive means two resident-weight buffers, two KV caches and two slot
//     pools at once, and at 128 slots that is 2 x 12.9 GB. Not double-committing matters
//     more, and the GUI already surfaces load_error_ when the new load fails.
//  3. initialize() runs WITHOUT runner_mutex_ held, so /api/telemetry keeps answering
//     during a multi-GB load instead of freezing the UI exactly when it should show progress.
bool HTTPServer::swap_runner(const std::string& model_dir, bool load, std::string& error) {
    std::unique_lock<std::mutex> gen(generate_mutex_, std::try_to_lock);
    if (!gen.owns_lock()) {
        error = "BUSY:A generation is in progress. POST /api/stop first, or retry.";
        return false;
    }

    // Take the config the user actually asked for. THIS is the fix for the reload path
    // silently discarding it: the runner was previously constructed and initialized with
    // neither setter called, so every reload fell back to RAM auto-sizing and the GUI's
    // "click Load Model to reinitialize" instruction was false.
    ServerConfig cfg;
    {
        std::lock_guard<std::mutex> guard(config_mutex_);
        cfg = config_;
    }

    std::shared_ptr<D3D12Context> ctx = current_ctx();

    // Resolve and VALIDATE before touching the live runner.
    //
    // Both halves of this matter. Resolution: a bare bundle name is resolved against the
    // working directory first, and running from build/ that is where the retired repacker's
    // stale placeholder lives -- so an unresolved name loaded a bundle whose layout.json
    // predates the expertBlock format, while startup (which resolves properly) had loaded
    // the real one. Validation: parsing costs nothing, and doing it here means a bad path
    // cannot cost the user a working model, which is the one real downside of releasing the
    // old runner first.
    std::string resolved = model_dir;
    GTurboManifestV1 manifest;
    PackedExpertsLayoutV1 layout;
    if (load) {
        resolved = resolve_bundle_path(model_dir);
        try {
            manifest = GTurboManifestV1::from_json_string(
                read_text_file(resolved + "/manifest.json"));
            layout = PackedExpertsLayoutV1::from_json_string(
                read_text_file(resolved + "/packed_experts/layout.json"));
            layout.cross_validate(manifest);
        } catch (const std::exception& ex) {
            // The previously loaded model is untouched.
            error = std::string(ex.what()) + " (resolved '" + model_dir + "' to '" +
                    resolved + "')";
            std::lock_guard<std::mutex> guard(runner_mutex_);
            load_error_ = error;
            return false;
        }
    }

    // Drop the old runner first -- see (2) above.
    {
        std::shared_ptr<ForwardRunner> old;
        {
            std::lock_guard<std::mutex> guard(runner_mutex_);
            old = std::move(runner_);
            runner_ = nullptr;
        }
        if (old && ctx) {
            // The runner's buffers may still be referenced by command lists in flight.
            ctx->flush_gpu();
        }
        old.reset();
    }

    if (!load) {
        return true;
    }

    try {
        if (!ctx) {
            ctx = std::make_shared<D3D12Context>();
            ctx->initialize(false);
            std::lock_guard<std::mutex> guard(runner_mutex_);
            ctx_ = ctx;
        }

        auto next = std::make_shared<ForwardRunner>(ctx, manifest, layout, resolved);
        // 0 means "auto-size from RAM" in both the config and the runner, so the sentinel
        // passes straight through.
        next->set_expert_slots(static_cast<size_t>(cfg.slots > 0 ? cfg.slots : 0));
        next->set_max_context(cfg.context_len);
        next->initialize();

        {
            std::lock_guard<std::mutex> guard(runner_mutex_);
            runner_ = std::move(next);
            load_error_.clear();
            // main.cpp snapshots max_context once at startup, so without this refresh the
            // /v1 max_tokens clamp keeps using the pre-reload value forever.
            openai_cfg_.max_context = runner_->max_context();
        }
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        std::lock_guard<std::mutex> guard(runner_mutex_);
        load_error_ = error;
        return false;
    }
}

void HTTPServer::start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<D3D12Context> ctx) {
    {
        std::lock_guard<std::mutex> guard(runner_mutex_);
        runner_ = runner;
        ctx_ = ctx;
    }
    // Constructed here rather than lazily on the first /v1 request. That lazy path ran from
    // a detached handler thread with no synchronization, so two concurrent first requests
    // both saw null, both constructed one, and one unique_ptr assignment destroyed the
    // other's object while a Lease still pointed at it.
    if (!coordinator_) {
        coordinator_ = std::make_unique<RequestCoordinator>(openai_cfg_.queue_limit);
    }
    is_running_ = true;
    server_thread_ = std::thread(&HTTPServer::listen_loop, this, port);
}

void HTTPServer::stop() {
    if (is_running_) {
        is_running_ = false;
        if (listen_socket_ != 0 && listen_socket_ != INVALID_SOCKET) {
            closesocket(static_cast<SOCKET>(listen_socket_));
            listen_socket_ = 0;
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }
}

void HTTPServer::listen_loop(uint16_t port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[SERVER] WSAStartup failed\n";
        return;
    }

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "[SERVER] Socket creation failed\n";
        WSACleanup();
        return;
    }

    listen_socket_ = static_cast<uintptr_t>(server_fd);

    BOOL opt = TRUE;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    // Loopback by default. This used to be an unconditional INADDR_ANY, which combined with
    // no authentication and a permissive CORS header to expose model loading -- and the
    // static file handler -- to anything that could reach the port.
    if (bind_address_ == "0.0.0.0") {
        address.sin_addr.s_addr = INADDR_ANY;
    } else {
        address.sin_addr.s_addr = inet_addr(bind_address_.c_str());
        if (address.sin_addr.s_addr == INADDR_NONE) {
            std::cerr << "[SERVER] Invalid bind address '" << bind_address_ << "'\n";
            closesocket(server_fd);
            WSACleanup();
            return;
        }
    }
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[SERVER] Bind failed on port " << port << "\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    if (listen(server_fd, 10) == SOCKET_ERROR) {
        std::cerr << "[SERVER] Listen failed\n";
        closesocket(server_fd);
        WSACleanup();
        return;
    }

    std::cout << "[SERVER] GUI Web Server active at http://localhost:" << port << "\n";

    while (is_running_) {
        SOCKET client_socket = accept(server_fd, NULL, NULL);
        if (client_socket != INVALID_SOCKET) {
            std::thread(&HTTPServer::handle_client, this, static_cast<uintptr_t>(client_socket)).detach();
        }
    }

    closesocket(server_fd);
    WSACleanup();
}

// Rejects a sampling combination at accept time, using the engine's own rule set, so the
// GUI learns about it now rather than on the next generation. validate_sampling throws, and
// its message is the one worth showing.
static bool sampling_is_valid(const ServerConfig& cfg, std::string& why) {
    SamplingParams p;
    p.temperature = cfg.temperature;
    p.top_p = cfg.top_p;
    p.top_k = cfg.top_k;
    p.repetition_penalty = cfg.repetition_penalty;
    p.has_seed = cfg.has_seed;
    p.seed = cfg.seed;
    try {
        validate_sampling(p);
    } catch (const std::exception& ex) {
        why = ex.what();
        return false;
    }
    if (cfg.max_tokens <= 0) {
        why = "max_tokens must be greater than 0.";
        return false;
    }
    return true;
}

static std::string json_string_array(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ",";
        out += JsonValue::quote(items[i]);
    }
    out += "]";
    return out;
}

static void send_json_response(SOCKET client, const std::string& json_body, int status_code = 200) {
    // Reason phrases and framing now come from src/http.cpp. They used to be hand-built here
    // with the literal "OK" for every status, so errors went out as "HTTP/1.1 500 OK".
    send_http_response(static_cast<uintptr_t>(client), status_code, "application/json",
                       json_body, /*keep_alive=*/false,
                       {{"Access-Control-Allow-Headers", "Content-Type"}});
}

static std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

void HTTPServer::handle_client(uintptr_t client_socket_ptr) {
    SOCKET client = static_cast<SOCKET>(client_socket_ptr);

    // One request per connection for now: generation can occupy the socket for many seconds,
    // and the GUI does not pipeline. The reader itself supports keep-alive, so the OpenAI
    // endpoints can opt into it.
    HttpRequest req;
    constexpr size_t kMaxBody = 1024 * 1024;   // 1 MiB, matching the reference
    const HttpReadResult rr = read_http_request(client_socket_ptr, req, kMaxBody, 30000);

    switch (rr) {
        case HttpReadResult::Ok:
            break;
        case HttpReadResult::TooLarge:
            send_json_response(client,
                "{\"status\":\"ERROR\",\"message\":\"Request body exceeds 1 MiB.\"}", 413);
            closesocket(client);
            return;
        case HttpReadResult::UnsupportedMediaType:
            send_json_response(client,
                "{\"status\":\"ERROR\",\"message\":\"Content-Type must be application/json.\"}", 415);
            closesocket(client);
            return;
        case HttpReadResult::Malformed:
            send_json_response(client,
                "{\"status\":\"ERROR\",\"message\":\"Malformed HTTP request.\"}", 400);
            closesocket(client);
            return;
        default:   // Closed, Timeout -- nothing useful to reply to
            closesocket(client);
            return;
    }

    const std::string& method = req.method;
    const std::string& path = req.path;

    // Reject traversal before anything can act on the path.
    //
    // The static handler builds a filesystem path as fs::path("gui" + path), and nothing
    // upstream inspected the request target, so `GET /../../../../../Windows/win.ini`
    // returned that file with a 200. Percent-encoding is decoded here first, because
    // `%2e%2e%2f` reaches the same place without a literal "..".
    if (!request_path_is_safe(path)) {
        send_json_response(client_socket_ptr,
                           "{\"status\":\"ERROR\",\"message\":\"Invalid request path.\"}",
                           400);
        closesocket(client);
        return;
    }

    // OpenAI-compatible surface first: it owns /health and everything under /v1.
    if (path == "/health" || path.rfind("/v1/", 0) == 0) {
        if (handle_openai(client_socket_ptr, req)) {
            closesocket(client);
            return;
        }
    }

    if (method == "OPTIONS") {
        std::string options_resp = "HTTP/1.1 204 No Content\r\n"
                                   "Access-Control-Allow-Origin: *\r\n"
                                   "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                                   "Access-Control-Allow-Headers: Content-Type\r\n\r\n";
        send(client, options_resp.c_str(), static_cast<int>(options_resp.length()), 0);
        closesocket(client);
        return;
    }

    if (method == "GET") {
        if (path == "/api/models") {
            // Enumerate only bundles that actually exist on disk. Seeding this list with a
            // literal "gemma-4-26b-a4b.gturbo" made a missing model look installed.
            std::ostringstream json;
            json << "{\"models\":[";
            bool first = true;
            std::vector<std::string> model_paths;

            // Search every root the loader would search, not just the working directory,
            // and list only bundles that actually parse. Scanning "." alone offered the
            // stale placeholder in build/ as if it were the real model -- it has both
            // manifest.json and layout.json, so nothing short of parsing rejects it.
            std::error_code scan_ec;
            for (const auto& root : bundle_search_roots()) {
                for (const auto& entry : fs::directory_iterator(root, scan_ec)) {
                    if (scan_ec) break;
                    if (!entry.is_directory(scan_ec)) continue;
                    const std::string name = entry.path().filename().string();
                    if (name.size() <= 7 || name.compare(name.size() - 7, 7, ".gturbo") != 0) {
                        continue;
                    }
                    if (!bundle_loads(entry.path().string())) continue;
                    // Offer the bare name when it resolves back to this same directory, so
                    // the GUI keeps showing a readable label; otherwise the full path, so a
                    // shadowed bundle is still selectable and unambiguous.
                    const std::string label =
                        (resolve_bundle_path(name) == entry.path().string())
                            ? name : entry.path().string();
                    bool seen = false;
                    for (const auto& m : model_paths) seen = seen || (m == label);
                    if (!seen) model_paths.push_back(label);
                }
            }

            auto runner = current_runner();
            for (const auto& mpath : model_paths) {
                if (!first) json << ",";
                first = false;
                // Compare resolved locations, not the raw strings. The runner stores the
                // path it was loaded from (absolute, once resolution has run) while this
                // list emits a bare bundle name, so a plain == never matched -- which left
                // every entry reporting is_active:false and null architecture, and in turn
                // left the GUI unable to bound the slot slider by the model's real expert
                // count.
                bool is_active = false;
                if (runner) {
                    std::error_code eq_ec;
                    const fs::path a = fs::weakly_canonical(resolve_bundle_path(mpath), eq_ec);
                    const fs::path b = fs::weakly_canonical(runner->model_dir(), eq_ec);
                    is_active = !eq_ec && !a.empty() && a == b;
                }
                json << "{"
                     << "\"path\":" << JsonValue::quote(mpath) << ","
                     << "\"id\":" << JsonValue::quote(mpath) << ","
                     << "\"name\":" << JsonValue::quote(mpath) << ",";
                // Architecture is only known for a loaded bundle; reporting it for an
                // unopened directory would be a guess.
                if (is_active) {
                    const auto& manifest = runner->manifest();
                    const auto& arch = manifest.arch;
                    json << "\"layers\":" << arch.num_layers << ","
                         << "\"experts\":" << arch.num_experts << ","
                         << "\"top_k\":" << arch.top_k_experts << ","
                         << "\"context_window\":" << runner->max_context() << ","
                         // The GUI sized its expert-pool estimate from a stride hardcoded in
                         // JavaScript. Report the real one so the label cannot drift from the
                         // bundle the engine actually opened.
                         << "\"expert_stride\":" << runner->layout().expert_stride << ",";
                    if (!manifest.model_id.empty()) {
                        json << "\"model_id\":" << JsonValue::quote(manifest.model_id) << ",";
                    } else {
                        json << "\"model_id\":null,";
                    }
                    // Quantization is optional in the manifest; a bundle without it gets
                    // null and the GUI drops the row rather than showing a stale literal.
                    if (manifest.quant) {
                        const auto& q = manifest.quant->routed_expert;
                        std::ostringstream desc;
                        desc << q.scheme << " " << q.weight_bits << "-bit, group "
                             << q.group_size << ", " << q.scale_type << " scale/bias";
                        json << "\"quantization\":" << JsonValue::quote(desc.str()) << ",";
                    } else {
                        json << "\"quantization\":null,";
                    }
                } else {
                    json << "\"layers\":null,\"experts\":null,\"top_k\":null,"
                         << "\"context_window\":null,\"expert_stride\":null,"
                         << "\"model_id\":null,\"quantization\":null,";
                }
                json << "\"is_active\":" << (is_active ? "true" : "false")
                     << "}";
            }
            json << "]}";
            send_json_response(client, json.str());

        } else if (path == "/api/config") {
            // The GUI seeds its controls from here on load. Without it every slider showed a
            // value hardcoded in index.html -- the context dropdown read "62000 Tokens" while
            // the engine was auto-sized to 4096, and nothing on screen was the truth until the
            // user happened to drag something.
            //
            // context_len/slots report the *resolved* values off the runner when the
            // configured value is 0 (auto-size from installed RAM), so "auto" never displays
            // as a literal zero.
            ServerConfig cfg;
            {
                std::lock_guard<std::mutex> lock(config_mutex_);
                cfg = config_;
            }
            auto runner = current_runner();
            int live_ctx   = runner ? runner->max_context() : cfg.context_len;
            int live_slots = runner ? static_cast<int>(runner->expert_slots_per_layer())
                                    : cfg.slots;

            std::ostringstream json;
            json << "{\"status\":\"OK\",\"config\":{"
                 << "\"temperature\":" << cfg.temperature << ","
                 << "\"top_p\":" << cfg.top_p << ","
                 << "\"top_k\":" << cfg.top_k << ","
                 << "\"max_tokens\":" << cfg.max_tokens << ","
                 << "\"repetition_penalty\":" << cfg.repetition_penalty << ","
                 // A null seed means "draw a fresh one per request"; 0 is a legitimate seed
                 // value, so the two must not share a representation.
                 << "\"seed\":" << (cfg.has_seed ? std::to_string(cfg.seed) : "null") << ","
                 << "\"stop\":" << json_string_array(cfg.stop_strings) << ","
                 << "\"max_stop\":" << kMaxStopSequences << ","
                 << "\"eviction_policy\":\""
                 << ((runner && runner->eviction_policy() == EvictionPolicy::LRU) ? "LRU" : "LFU")
                 << "\","
                 << "\"context_len\":" << (cfg.context_len ? cfg.context_len : live_ctx) << ","
                 // The hard ceiling on context, so the GUI can build its dropdown from the
                 // engine rather than hardcoding a ladder that drifts out of sync with
                 // ATTN_MAX_SPAN. Requesting more than this throws at load.
                 << "\"context_max\":" << ForwardRunner::kAttentionMaxSpan << ","
                 << "\"slots\":" << (cfg.slots ? cfg.slots : live_slots) << ","
                 // The *_active pair is what the engine is actually running, as distinct
                 // from what has been requested for the next load. Reporting only the
                 // requested value meant that once the GUI had POSTed anything, /api/config
                 // echoed that number forever even though nothing had been applied -- so a
                 // pending change was indistinguishable from a live one.
                 << "\"context_len_active\":" << live_ctx << ","
                 << "\"slots_active\":" << live_slots << ","
                 << "\"reload_pending\":"
                 << (((cfg.context_len && cfg.context_len != live_ctx) ||
                      (cfg.slots && cfg.slots != live_slots)) ? "true" : "false")
                 << "}}";
            send_json_response(client, json.str());

        } else if (path == "/api/server_info") {
            // Read-only home for everything fixed at process launch. These are real settings
            // that change how the server behaves -- the model id the OpenAI route demands,
            // the queue depth before a 429, the interface it listens on -- and none of them
            // was visible anywhere in the GUI, so a user could not tell why a request was
            // being refused.
            OpenAIServerConfig ocfg = openai_config();
            std::string host;
            uint16_t port = 0;
            bool serve_mode = false;
            {
                std::lock_guard<std::mutex> guard(runner_mutex_);
                host = bind_address_;
                port = startup_port_;
                serve_mode = serve_mode_;
            }
            auto info_ctx = current_ctx();
            const HostEnvironment& env = host_environment();

            std::ostringstream json;
            json << "{\"status\":\"OK\",\"server\":{"
                 << "\"version\":" << JsonValue::quote(GTURBO_VERSION_STRING) << ","
                 << "\"host\":" << JsonValue::quote(host) << ","
                 << "\"port\":" << port << ","
                 << "\"lan_accessible\":" << ((host == "0.0.0.0") ? "true" : "false") << ","
                 << "\"serve_mode\":" << (serve_mode ? "true" : "false") << ","
                 << "\"model_id\":" << JsonValue::quote(ocfg.model_id) << ","
                 << "\"queue_limit\":" << ocfg.queue_limit << ","
                 << "\"context_max\":" << ForwardRunner::kAttentionMaxSpan << ","
                 << "\"max_stop_sequences\":" << kMaxStopSequences << ","
                 << "\"gpu_name\":"
                 << JsonValue::quote(info_ctx ? info_ctx->adapter_name() : "No D3D12 device")
                 << ","
                 // What the model-download flow needs in order to run. The GUI disables the
                 // button with the specific unmet reason rather than letting the user press
                 // it and discover the problem several seconds later.
                 << "\"python_available\":" << (env.python_available ? "true" : "false") << ","
                 << "\"python_version\":"
                 << (env.python_version.empty() ? "null" : JsonValue::quote(env.python_version))
                 << ","
                 << "\"converter_present\":" << (env.converter_present ? "true" : "false") << ","
                 << "\"free_disk_gb\":" << env.free_disk_gb << ","
                 // Peak transient cost of a conversion, from the README: ~14 GB for the
                 // bundle plus ~15 GB of streamed input.
                 << "\"download_needs_gb\":29"
                 << "}}";
            send_json_response(client, json.str());

        } else if (path == "/api/download/status") {
            const FetchProgress fp = fetcher_.snapshot();
            std::ostringstream json;
            json << "{\"status\":\"OK\",\"download\":{"
                 << "\"state\":" << JsonValue::quote(fetch_state_name(fp.state)) << ","
                 << "\"output\":" << JsonValue::quote(fp.output) << ","
                 << "\"stage\":" << JsonValue::quote(fp.stage) << ","
                 << "\"step\":" << fp.step << ","
                 << "\"steps\":" << fp.steps << ","
                 << "\"label\":" << JsonValue::quote(fp.label) << ","
                 << "\"bytes_done\":" << fp.bytes_done << ","
                 << "\"bytes_total\":" << fp.bytes_total << ","
                 << "\"pct\":" << fp.pct << ","
                 << "\"rate_mbs\":" << fp.rate_mbs << ","
                 << "\"eta_s\":" << fp.eta_s << ","
                 << "\"exit_code\":" << fp.exit_code << ","
                 << "\"message\":" << JsonValue::quote(fp.message) << ","
                 << "\"log\":" << json_string_array(fetcher_.recent_output())
                 << "}}";
            send_json_response(client, json.str());

        } else if (path == "/api/telemetry") {
            D3D12Context::SystemMemoryInfo sys_mem{};
            auto runner = current_runner();
            auto tctx = current_ctx();
            if (tctx) {
                sys_mem = tctx->query_memory_info();
            }

            ModelMemoryUsage mod_mem{};
            PerformanceMetrics perf{};
            std::string model_dir = "None (No Model Loaded)";
            bool active = false;
            std::vector<int> active_experts;
            // Copied under runner_mutex_ rather than read bare off the member.
            const std::string load_err = load_error();

            if (runner) {
                mod_mem = runner->get_memory_usage();
                perf = runner->metrics();
                model_dir = runner->model_dir();
                active_experts = runner->last_active_experts();
                active = true;
            }

            std::ostringstream json;
            json << "{"
                 << "\"status\":\"OK\","
                 // Every string here goes through quote(). model_dir is an absolute Windows
                 // path, so interpolating it raw emitted "C:\Users\..." -- an invalid escape
                 // that made the whole telemetry document unparseable, silently breaking the
                 // GUI's 1.5 s poll.
                 << "\"gpu_name\":" << JsonValue::quote(tctx ? tctx->adapter_name() : "No D3D12 device") << ","
                 << "\"model_active\":" << (active ? "true" : "false") << ","
                 << "\"model_dir\":" << JsonValue::quote(model_dir) << ","
                 << "\"load_error\":"
                 << (load_err.empty() ? "null" : JsonValue::quote(load_err)) << ","
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
                     << "\"gpu_shared_used_mb\":" << (sys_mem.gpu_shared_ram_used / (1024.0 * 1024.0)) << ","
                     // Buffers that could not get host-coherent CUSTOM/L0 memory and fell
                     // back to a write-combined UPLOAD heap. Non-zero means the adapter's
                     // shared-memory budget is exhausted, which is the condition that has
                     // been suspected behind the intermittent mid-run crash and could never
                     // be checked while the fallback was silent. Any value above 0 is worth
                     // acting on -- lower --slots or --context.
                     << "\"uma_upload_fallbacks\":" << (tctx ? tctx->uma_fallback_count() : 0u)
                 << "},"
                 << "\"performance\":{"
                     << "\"prefill_toks_sec\":" << perf.prefill_tokens_per_sec << ","
                     << "\"decode_toks_sec\":" << perf.decode_tokens_per_sec << ","
                     << "\"total_time_ms\":" << perf.total_time_ms << ","
                     << "\"total_io_mbs\":" << ((perf.total_time_ms > 0) ? (perf.total_io_bytes / (1024.0 * 1024.0) / (perf.total_time_ms / 1000.0)) : 0.0) << ","
                     << "\"total_io_calls\":" << perf.total_io_calls << ","
                     // The per-phase breakdown. These six already existed on
                     // PerformanceMetrics and were printed by the CLI footer, but no
                     // endpoint reported them -- so the GUI could show a decode rate with no
                     // way to say where the time went. lm_head_ms deliberately overlaps the
                     // other buckets; it is a slice of the same token, not a fifth phase.
                     << "\"expert_io_ms\":" << perf.expert_io_ms << ","
                     << "\"gpu_wait_ms\":" << perf.gpu_wait_ms << ","
                     << "\"lm_head_ms\":" << perf.lm_head_ms << ","
                     << "\"cpu_other_ms\":" << perf.cpu_other_ms << ","
                     << "\"gpu_waits\":" << perf.gpu_waits << ","
                     << "\"tokens_measured\":" << perf.tokens_measured
                 << "},"
                 << "\"cache\":{"
                     << "\"hit_rate_pct\":" << mod_mem.cache_hit_rate_pct << ","
                     << "\"eviction_policy\":\""
                     // Read the live policy off the runner rather than a server-side mirror
                     // that nothing ever wrote, which always reported LFU.
                     << ((runner && runner->eviction_policy() == EvictionPolicy::LRU) ? "LRU" : "LFU")
                     << "\""
                 << "},"
                 << "\"active_experts\":[";
            for (size_t i = 0; i < active_experts.size(); ++i) {
                if (i > 0) json << ",";
                json << active_experts[i];
            }
            json << "]}";
            send_json_response(client, json.str());

        } else {
            // Serve web GUI static assets
            std::string rel_path = "gui" + path;
            if (path == "/" || path == "/index.html") rel_path = "gui/index.html";

            wchar_t exe_buf[MAX_PATH]{};
            GetModuleFileNameW(NULL, exe_buf, MAX_PATH);
            fs::path exe_dir = fs::path(exe_buf).parent_path();

            std::vector<fs::path> candidates = {
                fs::path(rel_path),
                exe_dir / rel_path,
                exe_dir / ".." / rel_path,
                exe_dir / "gui" / (path == "/" ? "index.html" : path.substr(1))
            };

            std::ifstream file;
            std::string file_path_str = rel_path;
            for (const auto& cand : candidates) {
                file.open(cand, std::ios::binary);
                if (file.is_open()) {
                    file_path_str = cand.string();
                    break;
                }
            }

            if (file.is_open()) {
                std::stringstream ss;
                ss << file.rdbuf();
                std::string content = ss.str();

                // Always declare the charset. The GUI assets are UTF-8; without an explicit
                // charset a browser falls back to guessing (windows-1252 in practice) and
                // every emoji in index.html renders as mojibake.
                std::string ext = fs::path(file_path_str).extension().string();
                std::string mime = "text/html; charset=utf-8";
                if (ext == ".css") mime = "text/css; charset=utf-8";
                else if (ext == ".js") mime = "application/javascript; charset=utf-8";
                // Binary/vector assets must NOT inherit the text/html fallback. logo.svg was
                // served as text/html, and a browser refuses to render an <img> whose type is
                // a document type -- the header showed the alt text ("Turbo-WinFare Logo",
                // clipped to "Turb") in place of the icon, which looked like a missing file
                // rather than a wrong header.
                else if (ext == ".svg") mime = "image/svg+xml; charset=utf-8";
                else if (ext == ".png") mime = "image/png";
                else if (ext == ".jpg" || ext == ".jpeg") mime = "image/jpeg";
                else if (ext == ".webp") mime = "image/webp";
                else if (ext == ".gif") mime = "image/gif";
                else if (ext == ".ico") mime = "image/x-icon";
                else if (ext == ".woff2") mime = "font/woff2";
                else if (ext == ".json") mime = "application/json; charset=utf-8";

                std::ostringstream resp;
                resp << "HTTP/1.1 200 OK\r\n"
                     << "Content-Type: " << mime << "\r\n"
                     << "Content-Length: " << content.length() << "\r\n"
                     << "Access-Control-Allow-Origin: *\r\n\r\n"
                     << content;
                std::string resp_str = resp.str();
                send(client, resp_str.c_str(), static_cast<int>(resp_str.length()), 0);
            } else {
                std::string not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
                send(client, not_found.c_str(), static_cast<int>(not_found.length()), 0);
            }
        }
    } else if (method == "POST") {
        JsonValue req_json;
        bool body_ok = false;
        if (!req.body.empty()) {
            try {
                req_json = JsonValue::parse(req.body);
                body_ok = req_json.is_object();
            } catch (const std::exception&) {
                body_ok = false;
            }
        }

        if (path == "/api/generate") {
            const std::string prompt = body_ok ? req_json.string_or("prompt", "") : std::string();

            // Multi-turn: a `messages` array wins over `prompt`. apply_chat_template already
            // accepts arbitrary history, so this only had to be plumbed through -- previously
            // only the latest prompt was ever sent and every turn started a fresh
            // conversation.
            std::vector<Tokenizer::ChatMessage> messages;
            std::string message_error;
            if (body_ok && req_json.has("messages")) {
                const JsonValue& arr = req_json.at("messages", "request");
                if (!arr.is_array() || arr.array_value.empty()) {
                    message_error = "'messages' must be a non-empty array.";
                } else {
                    for (size_t i = 0; i < arr.array_value.size(); ++i) {
                        const JsonValue& m = arr.array_value[i];
                        if (!m.is_object()) {
                            message_error = "Each message must be an object.";
                            break;
                        }
                        const std::string role = m.string_or("role", "");
                        const std::string content = m.string_or("content", "");
                        if (role.empty()) {
                            message_error = "Each message needs a 'role'.";
                            break;
                        }
                        messages.push_back({role, content});
                    }
                }
            }

            GenerationOptions opts{};
            if (body_ok) {
                // Request values win; otherwise fall back to whatever /api/config last set.
                // temperature used to be pinned to 0.7 here, which ignored the GUI slider and
                // forced every request onto the slow materializing-logits path.
                std::lock_guard<std::mutex> cfg(config_mutex_);
                opts.max_tokens  = static_cast<int>(req_json.int_or("max_tokens", config_.max_tokens));
                opts.temperature = static_cast<float>(req_json.double_or("temperature", config_.temperature));
                opts.top_p       = static_cast<float>(req_json.double_or("top_p", config_.top_p));
                opts.top_k       = static_cast<int>(req_json.int_or("top_k", config_.top_k));
                opts.repetition_penalty = static_cast<float>(
                    req_json.double_or("repetition_penalty", config_.repetition_penalty));
                if (req_json.has("seed")) {
                    opts.has_seed = true;
                    opts.seed = static_cast<uint64_t>(req_json.int_or("seed", 0));
                } else {
                    opts.has_seed = config_.has_seed;
                    opts.seed = config_.seed;
                }
                // The GUI has always sent this; the server used to drop it on the floor.
                opts.system_prompt = req_json.string_or("system_prompt", "");

                // Stop sequences. This route did not parse `stop` at all, so a feature the
                // engine, the C ABI and /v1 all support was unreachable here -- and a caller
                // that sent one got a silent no-op rather than an error.
                if (req_json.has("stop")) {
                    std::vector<std::string> stops;
                    const std::string stop_err =
                        parse_stop_field(req_json, kMaxStopSequences, stops);
                    if (!stop_err.empty()) {
                        message_error = stop_err;
                    } else {
                        opts.stop_strings = std::move(stops);
                    }
                } else {
                    opts.stop_strings = config_.stop_strings;
                }
            }

            if (!message_error.empty()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":" + JsonValue::quote(message_error) + "}", 400);
            } else if (prompt.empty() && messages.empty()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":\"Request needs either 'prompt' or 'messages'.\"}", 400);
            } else if (!current_runner()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":\"No model loaded. Load a .gturbo bundle first.\"}", 409);
            } else {
                // Serialize generation. Concurrent callers previously ran produce_token on
                // the same runner and the same GPU scratch, silently corrupting both replies.
                std::unique_lock<std::mutex> gen(generate_mutex_, std::try_to_lock);
                if (!gen.owns_lock()) {
                    send_json_response(client,
                        "{\"status\":\"ERROR\", \"message\":\"A generation is already in progress. "
                        "Retry once it completes.\"}", 429);
                } else {
                    // Engine failures are reported as errors, never as assistant text -- otherwise
                    // a broken forward pass is indistinguishable from a model reply.
                    try {
                        if (messages.empty()) {
                            if (!opts.system_prompt.empty()) {
                                messages.push_back({"system", opts.system_prompt});
                            }
                            messages.push_back({"user", prompt});
                        }
                        // Snapshot under the generation lock: the null check above raced a
                        // concurrent reload, and dereferencing the member directly let that
                        // reload destroy the runner mid-call.
                        auto runner = current_runner();
                        if (!runner) {
                            throw GTurboFormatError("Model was unloaded before generation started.");
                        }
                        auto result = runner->generate_chat(messages, opts);

                        std::ostringstream json_resp;
                        json_resp << "{\"status\":\"SUCCESS\","
                                  << "\"output_text\":" << JsonValue::quote(result.text) << ","
                                  << "\"stop_reason\":\"" << stop_reason_name(result.reason) << "\","
                                  << "\"prompt_tokens\":" << result.prompt_tokens << ","
                                  << "\"completion_tokens\":" << result.completion_tokens << ","
                                  << "\"ttft_ms\":" << result.ttft_ms << "}";
                        send_json_response(client, json_resp.str());
                    } catch (const std::exception& ex) {
                        send_json_response(client,
                            "{\"status\":\"ERROR\", \"message\":" +
                            JsonValue::quote(ex.what()) + "}", 500);
                    }
                }
            }

        } else if (path == "/api/config") {
            // This endpoint did not exist. app.js POSTs to it on every slider move, so the
            // request fell through every branch below and reached closesocket() with no
            // response at all -- the fetch hung, the catch swallowed it, and every knob in
            // the GUI was inert.
            if (!body_ok) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":\"Request body must be a JSON object.\"}", 400);
            } else {
                bool requires_reload = false;
                ServerConfig applied;
                std::string reject;
                // Snapshot before taking config_mutex_: the lock order is
                // generate_mutex_ -> runner_mutex_ -> config_mutex_, so acquiring
                // runner_mutex_ while holding config_mutex_ would reverse it.
                auto cfg_runner = current_runner();
                {
                    std::lock_guard<std::mutex> cfg(config_mutex_);

                    // Work on a copy and commit only once every field has passed. Applying
                    // straight to config_ meant a rejected request still stored its bad
                    // value: POSTing top_k 999 answered 400 and left 999 in place, so the
                    // *next* request -- which does not resend top_k, because the GUI only
                    // sends what it holds -- was rejected too, for a value the user never
                    // successfully set. A 400 must leave the configuration untouched.
                    ServerConfig next = config_;

                    next.temperature = static_cast<float>(req_json.double_or("temperature", next.temperature));
                    next.top_p       = static_cast<float>(req_json.double_or("top_p", next.top_p));
                    next.top_k       = static_cast<int>(req_json.int_or("top_k", next.top_k));
                    next.max_tokens  = static_cast<int>(req_json.int_or("max_tokens", next.max_tokens));
                    next.repetition_penalty = static_cast<float>(
                        req_json.double_or("repetition_penalty", next.repetition_penalty));

                    // An explicit null clears the seed back to "fresh each request"; 0 is a
                    // real seed and must not mean the same thing.
                    if (req_json.has("seed")) {
                        const JsonValue& sv = req_json.object_value.at("seed");
                        if (sv.is_null()) {
                            next.has_seed = false;
                            next.seed = 0;
                        } else {
                            next.has_seed = true;
                            next.seed = static_cast<uint64_t>(req_json.int_or("seed", 0));
                        }
                    }

                    if (req_json.has("stop")) {
                        std::vector<std::string> stops;
                        const std::string stop_err =
                            parse_stop_field(req_json, kMaxStopSequences, stops);
                        if (!stop_err.empty()) {
                            reject = stop_err;
                        } else {
                            next.stop_strings = std::move(stops);
                        }
                    }

                    if (req_json.has("eviction_policy")) {
                        std::string pol = req_json.string_or("eviction_policy", "lfu");
                        std::transform(pol.begin(), pol.end(), pol.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                        next.eviction_policy = (pol == "lru") ? EvictionPolicy::LRU
                                                              : EvictionPolicy::LFU;
                    }

                    // Slot count and context size take effect on the next model load, not
                    // immediately -- both are fixed at initialize(). Accepting them silently
                    // would report success for a change that had not happened.
                    //
                    // Compare against the *resolved* values, because a stored 0 means
                    // "auto-size from RAM" rather than a literal zero. The GUI seeds its
                    // controls from the resolved numbers, so comparing against the raw 0 made
                    // the very first slider touch claim a reload was needed when nothing had
                    // actually changed.
                    const int cur_ctx = config_.context_len ? config_.context_len
                                      : (cfg_runner ? cfg_runner->max_context() : 0);
                    const int cur_slots = config_.slots ? config_.slots
                                        : (cfg_runner ? static_cast<int>(cfg_runner->expert_slots_per_layer()) : 0);
                    const int new_ctx   = static_cast<int>(req_json.int_or("context_len", cur_ctx));
                    const int new_slots = static_cast<int>(req_json.int_or("slots", cur_slots));

                    // Validate at accept time. These used to be stored verbatim and echoed
                    // back as SUCCESS, so `slots: 999` looked applied and only exploded on
                    // the next load -- which, now that a reload actually honours them, would
                    // turn a typo into a dead engine.
                    if (!reject.empty()) {
                        // A stop-field rejection above already has a message.
                    } else if (!sampling_is_valid(next, reject)) {
                        // Message filled by sampling_is_valid.
                    } else if (new_ctx != 0 && (new_ctx < 1 || new_ctx > ForwardRunner::kAttentionMaxSpan)) {
                        reject = "context_len must be between 1 and " +
                                 std::to_string(ForwardRunner::kAttentionMaxSpan) +
                                 " (0 auto-sizes from RAM).";
                    } else if (new_slots != 0 && cfg_runner) {
                        const auto& arch = cfg_runner->manifest().arch;
                        if (new_slots <= arch.top_k_experts) {
                            reject = "slots must exceed top_k_experts (" +
                                     std::to_string(arch.top_k_experts) + "); use at least " +
                                     std::to_string(arch.top_k_experts + 1) +
                                     " (0 auto-sizes from RAM).";
                        } else if (new_slots > arch.num_experts) {
                            reject = "slots cannot exceed the " +
                                     std::to_string(arch.num_experts) +
                                     " experts per layer; a larger pool can never fill.";
                        }
                    } else if (new_slots != 0 && new_slots < 1) {
                        reject = "slots must be positive (0 auto-sizes from RAM).";
                    }

                    if (reject.empty()) {
                        requires_reload = (new_ctx != cur_ctx) || (new_slots != cur_slots);
                        next.context_len = new_ctx;
                        next.slots = new_slots;
                        config_ = next;          // the single commit point
                    }
                    applied = config_;
                }

                if (!reject.empty()) {
                    send_json_response(client,
                        "{\"status\":\"ERROR\", \"message\":" + JsonValue::quote(reject) + "}", 400);
                    closesocket(client);
                    return;
                }

                if (cfg_runner) {
                    cfg_runner->set_eviction_policy(applied.eviction_policy);
                }

                std::ostringstream json;
                json << "{\"status\":\"SUCCESS\","
                     << "\"requires_reload\":" << (requires_reload ? "true" : "false") << ","
                     << "\"config\":{"
                     << "\"temperature\":" << applied.temperature << ","
                     << "\"repetition_penalty\":" << applied.repetition_penalty << ","
                     << "\"seed\":"
                     << (applied.has_seed ? std::to_string(applied.seed) : "null") << ","
                     << "\"stop\":" << json_string_array(applied.stop_strings) << ","
                     << "\"top_p\":" << applied.top_p << ","
                     << "\"top_k\":" << applied.top_k << ","
                     << "\"max_tokens\":" << applied.max_tokens << ","
                     << "\"eviction_policy\":\""
                     << (applied.eviction_policy == EvictionPolicy::LRU ? "LRU" : "LFU") << "\","
                     << "\"context_len\":" << applied.context_len << ","
                     << "\"slots\":" << applied.slots
                     << "}}";
                send_json_response(client, json.str());
            }

        } else if (path == "/api/load_model") {
            std::string target_path = body_ok
                ? req_json.string_or("model_path", "gemma-4-26b-a4b.gturbo")
                : std::string("gemma-4-26b-a4b.gturbo");

            std::string error;
            if (swap_runner(target_path, /*load=*/true, error)) {
                std::ostringstream json;
                json << "{\"status\":\"SUCCESS\", \"message\":\"Successfully loaded model bundle: "
                     << escape_json_string(target_path) << "\"}";
                send_json_response(client, json.str());
            } else {
                // 409 when a generation is in flight, 500 when the load itself failed. The
                // GUI needs to tell "try again in a moment" from "this bundle is broken".
                const bool busy = error.rfind("BUSY:", 0) == 0;
                std::ostringstream json;
                json << "{\"status\":\"ERROR\", \"message\":\""
                     << escape_json_string(busy ? error.substr(5) : "Failed to load model: " + error)
                     << "\"}";
                send_json_response(client, json.str(), busy ? 409 : 500);
            }

        } else if (path == "/api/unload_model") {
            std::string error;
            if (swap_runner("", /*load=*/false, error)) {
                send_json_response(client,
                    "{\"status\":\"SUCCESS\", \"message\":\"Model successfully unloaded from UMA RAM.\"}");
            } else {
                std::ostringstream json;
                json << "{\"status\":\"ERROR\", \"message\":\""
                     << escape_json_string(error.substr(5)) << "\"}";
                send_json_response(client, json.str(), 409);
            }

        } else if (path == "/api/stop") {
            // Deliberately does NOT take generate_mutex_: it would block on the very
            // generation it is trying to cancel. Safe because runner_ is only republished
            // while generate_mutex_ is held, so an in-flight generation is always running
            // on the currently published runner.
            auto runner = current_runner();
            if (runner) {
                runner->stop_generation();
            }
            std::ostringstream json;
            json << "{\"status\":\"SUCCESS\", \"had_runner\":" << (runner ? "true" : "false")
                 << ", \"message\":\"Generation stop signal sent.\"}";
            send_json_response(client, json.str());

        } else if (path == "/api/download") {
            // Starts a bundle build: tools/convert_hf_to_gturbo.py streams the pinned
            // checkpoint and repacks it. See include/gturbo/model_fetch.hpp for why this
            // drives the existing script rather than reimplementing it here.
            if (!body_ok) {
                send_json_response(client,
                    "{\"status\":\"ERROR\",\"message\":\"Request body must be a JSON object.\"}",
                    400);
                closesocket(client);
                return;
            }

            // Refuse while the engine is generating. Decode is I/O-bound and a conversion
            // saturates the same NVMe -- CLAUDE.md records a contended run at 5.07 tok/s
            // against 6.03 idle, and it also evicts the expert pages the streamer relies on
            // the OS page cache to hold.
            {
                std::unique_lock<std::mutex> busy(generate_mutex_, std::try_to_lock);
                if (!busy.owns_lock()) {
                    send_json_response(client,
                        "{\"status\":\"ERROR\",\"message\":\"A generation is in progress. "
                        "Downloading now would contend for the same disk and slow both.\"}",
                        409);
                    closesocket(client);
                    return;
                }
            }

            const std::string output = req_json.string_or("output", "");
            const std::string token = req_json.string_or("token", "");
            const bool resume = req_json.bool_or("resume", false);

            std::string start_error;
            if (!fetcher_.start(output, token, resume, start_error)) {
                // 409 when something is already running, 400 when the request itself is bad.
                const int code = fetcher_.is_running() ? 409 : 400;
                send_json_response(client,
                    "{\"status\":\"ERROR\",\"message\":" + JsonValue::quote(start_error) + "}",
                    code);
            } else {
                send_json_response(client,
                    "{\"status\":\"SUCCESS\",\"message\":\"Download started.\",\"output\":" +
                    JsonValue::quote(output) + "}", 202);
            }

        } else if (path == "/api/download/cancel") {
            const bool was_running = fetcher_.is_running();
            fetcher_.cancel();
            send_json_response(client,
                std::string("{\"status\":\"SUCCESS\",\"was_running\":") +
                (was_running ? "true" : "false") +
                ",\"message\":\"Cancelled. The partial download was kept for Resume.\"}");

        } else if (path == "/api/clear_cache") {
            // Clearing mid-generation is benign today (pinned slots survive), but taking
            // the lock costs two lines and removes the need for anyone to re-derive that.
            std::unique_lock<std::mutex> gen(generate_mutex_, std::try_to_lock);
            if (!gen.owns_lock()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":\"A generation is in progress.\"}", 409);
                closesocket(client);
                return;
            }
            auto runner = current_runner();
            if (runner) {
                runner->clear_expert_cache();
            }
            send_json_response(client, "{\"status\":\"SUCCESS\", \"message\":\"Expert DRAM Cache pool flushed.\"}");

        } else {
            send_json_response(client,
                "{\"status\":\"ERROR\", \"message\":\"Unknown endpoint.\"}", 404);
        }
    } else {
        send_json_response(client,
            "{\"status\":\"ERROR\", \"message\":\"Unsupported method.\"}", 405);
    }

    closesocket(client);
}

// ---------------------------------------------------------------------------
// OpenAI-compatible endpoints

bool HTTPServer::handle_openai(uintptr_t client, const HttpRequest& req) {
    auto send_err = [&](const ApiError& e) {
        send_http_response(client, e.status, "application/json", render_error(e), false);
    };

    if (req.path == "/health") {
        if (req.method != "GET") {
            send_err({405, "Method not allowed.", "invalid_request_error", "", "method_not_allowed"});
            return true;
        }
        send_http_response(client, 200, "application/json", "{\"status\":\"ok\"}", false);
        return true;
    }

    if (req.path == "/v1/models") {
        if (req.method != "GET") {
            send_err({405, "Method not allowed.", "invalid_request_error", "", "method_not_allowed"});
            return true;
        }
        send_http_response(client, 200, "application/json",
                           render_models_list(openai_config().model_id), false);
        return true;
    }

    if (req.path == "/v1/chat/completions") {
        if (req.method != "POST") {
            send_err({405, "Method not allowed.", "invalid_request_error", "", "method_not_allowed"});
            return true;
        }
        handle_chat_completion(client, req);
        return true;
    }

    send_err({404, "Unknown endpoint: " + req.path, "invalid_request_error", "", "not_found"});
    return true;
}

void HTTPServer::handle_chat_completion(uintptr_t client, const HttpRequest& req) {
    auto send_err = [&](const ApiError& e) {
        send_http_response(client, e.status, "application/json", render_error(e), false);
    };

    JsonValue body;
    try {
        body = JsonValue::parse(req.body);
    } catch (const std::exception& ex) {
        send_err({400, std::string("Invalid JSON: ") + ex.what(),
                  "invalid_request_error", "", "invalid_json"});
        return;
    }

    ValidatedChatRequest vr;
    ApiError err;
    if (!validate_chat_request(body, openai_config(), vr, err)) {
        send_err(err);
        return;
    }
    if (!current_runner()) {
        send_err({503, "No model is loaded.", "server_error", "", "model_not_loaded"});
        return;
    }

    // Admission before tokenization: an over-capacity request should not pay for a prompt it
    // will never run. coordinator_ is constructed in start(), so it is non-null for the
    // server's whole lifetime -- it used to be built lazily right here, from a detached
    // thread, which raced two concurrent first requests.
    auto lease = coordinator_->acquire();
    if (!lease) {
        send_err({429, "Server is at capacity; retry shortly.",
                  "server_error", "", "queue_full"});
        return;
    }

    // The generation lock, which this path did NOT take before.
    //
    // RequestCoordinator serializes only among /v1 callers, so a /v1 request and an
    // /api/generate request could run produce_token concurrently on the same runner and the
    // same GPU scratch buffers -- exactly the corruption generate_mutex_ exists to prevent.
    //
    // Blocking, not try_lock: the lease already guarantees at most one /v1 generation, so
    // this only ever waits on a GUI generation, and the caller opted into queueing by
    // getting a lease at all. Holding it across the SSE stream is correct -- that stream is
    // a generation -- and means a reload will 409 for its duration.
    std::lock_guard<std::mutex> gen(generate_mutex_);

    // Snapshot after locking: a reload may have republished runner_ while this request was
    // queued for its lease. Taking it here rather than at admission means the request runs
    // against one consistent runner, though possibly a different model than was loaded when
    // it was admitted.
    auto runner = current_runner();
    if (!runner) {
        send_err({503, "No model is loaded.", "server_error", "", "model_not_loaded"});
        return;
    }

    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    const std::string id = "chatcmpl-" + std::to_string(created) + "-" +
                           std::to_string(reinterpret_cast<uintptr_t>(&vr) & 0xFFFFFF);

    if (!vr.stream) {
        try {
            auto result = runner->generate_chat(vr.messages, vr.options);
            send_http_response(client, 200, "application/json",
                               render_completion(id, vr.model, result, created), false);
        } catch (const std::exception& ex) {
            send_err({500, ex.what(), "server_error", "", "generation_failed"});
        }
        return;
    }

    // --- SSE ------------------------------------------------------------------------
    if (!send_http_headers(client, 200, "text/event-stream")) return;

    auto send_event = [&](const std::string& payload) {
        return send_raw(client, "data: " + payload + "\n\n");
    };

    if (!send_event(render_chunk(id, vr.model, created, "", /*role_only=*/true))) return;

    // A long prefill can produce no output for many seconds, and an idle proxy will close
    // the connection in the meantime. A comment line keeps it alive without being visible
    // to the client as content.
    auto last_ping = std::chrono::steady_clock::now();
    bool alive = true;

    GenerationResult result;
    try {
        result = runner->generate_chat(
            vr.messages, vr.options,
            [&](const StreamEvent& ev) {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_ping).count() >= 5) {
                    last_ping = now;
                    if (!send_raw(client, ": ping\n\n")) { alive = false; return false; }
                }
                if (ev.kind == StreamEvent::Kind::Prefill || ev.delta.empty()) return true;

                if (!send_event(render_chunk(id, vr.model, created, ev.delta, false))) {
                    // The client is gone. Returning false abandons the generation instead of
                    // keeping the GPU busy for a socket nobody is reading.
                    alive = false;
                    return false;
                }
                return true;
            });
    } catch (const std::exception& ex) {
        if (alive) {
            send_event(render_error({500, ex.what(), "server_error", "", "generation_failed"}));
            send_raw(client, "data: [DONE]\n\n");
        }
        return;
    }

    if (!alive) return;
    if (!send_event(render_final_chunk(id, vr.model, created, result))) return;
    if (vr.include_usage) {
        if (!send_event(render_usage_chunk(id, vr.model, created, result))) return;
    }
    send_raw(client, "data: [DONE]\n\n");
}

} // namespace gturbo
