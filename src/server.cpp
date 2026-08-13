#include <winsock2.h>
#include <ws2tcpip.h>
#include "gturbo/server.hpp"
#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/json.hpp"
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

void HTTPServer::start(uint16_t port, std::shared_ptr<ForwardRunner> runner, std::shared_ptr<D3D12Context> ctx) {
    runner_ = runner;
    ctx_ = ctx;
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
    address.sin_addr.s_addr = INADDR_ANY;
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

            for (const auto& entry : fs::directory_iterator(".")) {
                if (entry.is_directory()) {
                    std::string p = entry.path().filename().string();
                    if (p.size() > 7 && p.compare(p.size() - 7, 7, ".gturbo") == 0) {
                        model_paths.push_back(p);
                    }
                }
            }

            for (const auto& mpath : model_paths) {
                if (!first) json << ",";
                first = false;
                bool is_active = (runner_ && runner_->model_dir() == mpath);
                json << "{"
                     << "\"path\":" << JsonValue::quote(mpath) << ","
                     << "\"id\":" << JsonValue::quote(mpath) << ","
                     << "\"name\":" << JsonValue::quote(mpath) << ",";
                // Architecture is only known for a loaded bundle; reporting it for an
                // unopened directory would be a guess.
                if (is_active) {
                    const auto& arch = runner_->manifest().arch;
                    json << "\"layers\":" << arch.num_layers << ","
                         << "\"experts\":" << arch.num_experts << ","
                         << "\"top_k\":" << arch.top_k_experts << ",";
                } else {
                    json << "\"layers\":null,\"experts\":null,\"top_k\":null,";
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
            int live_ctx   = runner_ ? runner_->max_context() : cfg.context_len;
            int live_slots = runner_ ? static_cast<int>(runner_->expert_slots_per_layer())
                                     : cfg.slots;

            std::ostringstream json;
            json << "{\"status\":\"OK\",\"config\":{"
                 << "\"temperature\":" << cfg.temperature << ","
                 << "\"top_p\":" << cfg.top_p << ","
                 << "\"top_k\":" << cfg.top_k << ","
                 << "\"max_tokens\":" << cfg.max_tokens << ","
                 << "\"eviction_policy\":\""
                 << ((runner_ && runner_->eviction_policy() == EvictionPolicy::LRU) ? "LRU" : "LFU")
                 << "\","
                 << "\"context_len\":" << (cfg.context_len ? cfg.context_len : live_ctx) << ","
                 << "\"slots\":" << (cfg.slots ? cfg.slots : live_slots)
                 << "}}";
            send_json_response(client, json.str());

        } else if (path == "/api/telemetry") {
            D3D12Context::SystemMemoryInfo sys_mem{};
            if (ctx_) {
                sys_mem = ctx_->query_memory_info();
            }

            ModelMemoryUsage mod_mem{};
            PerformanceMetrics perf{};
            std::string model_dir = "None (No Model Loaded)";
            bool active = false;
            std::vector<int> active_experts;

            if (runner_) {
                mod_mem = runner_->get_memory_usage();
                perf = runner_->metrics();
                model_dir = runner_->model_dir();
                active_experts = runner_->last_active_experts();
                active = true;
            }

            std::ostringstream json;
            json << "{"
                 << "\"status\":\"OK\","
                 // Every string here goes through quote(). model_dir is an absolute Windows
                 // path, so interpolating it raw emitted "C:\Users\..." -- an invalid escape
                 // that made the whole telemetry document unparseable, silently breaking the
                 // GUI's 1.5 s poll.
                 << "\"gpu_name\":" << JsonValue::quote(ctx_ ? ctx_->adapter_name() : "No D3D12 device") << ","
                 << "\"model_active\":" << (active ? "true" : "false") << ","
                 << "\"model_dir\":" << JsonValue::quote(model_dir) << ","
                 << "\"load_error\":"
                 << (load_error_.empty() ? "null" : JsonValue::quote(load_error_)) << ","
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
                     << "\"eviction_policy\":\""
                     // Read the live policy off the runner rather than a server-side mirror
                     // that nothing ever wrote, which always reported LFU.
                     << ((runner_ && runner_->eviction_policy() == EvictionPolicy::LRU) ? "LRU" : "LFU")
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
                opts.repetition_penalty =
                    static_cast<float>(req_json.double_or("repetition_penalty", 1.0));
                if (req_json.has("seed")) {
                    opts.has_seed = true;
                    opts.seed = static_cast<uint64_t>(req_json.int_or("seed", 0));
                }
                // The GUI has always sent this; the server used to drop it on the floor.
                opts.system_prompt = req_json.string_or("system_prompt", "");
            }

            if (!message_error.empty()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":" + JsonValue::quote(message_error) + "}", 400);
            } else if (prompt.empty() && messages.empty()) {
                send_json_response(client,
                    "{\"status\":\"ERROR\", \"message\":\"Request needs either 'prompt' or 'messages'.\"}", 400);
            } else if (!runner_) {
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
                        auto result = runner_->generate_chat(messages, opts);

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
                {
                    std::lock_guard<std::mutex> cfg(config_mutex_);
                    config_.temperature = static_cast<float>(req_json.double_or("temperature", config_.temperature));
                    config_.top_p       = static_cast<float>(req_json.double_or("top_p", config_.top_p));
                    config_.top_k       = static_cast<int>(req_json.int_or("top_k", config_.top_k));
                    config_.max_tokens  = static_cast<int>(req_json.int_or("max_tokens", config_.max_tokens));

                    if (req_json.has("eviction_policy")) {
                        std::string p = req_json.string_or("eviction_policy", "lfu");
                        std::transform(p.begin(), p.end(), p.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                        config_.eviction_policy = (p == "lru") ? EvictionPolicy::LRU : EvictionPolicy::LFU;
                    }

                    // Slot count and context size are fixed at initialize(); accepting them
                    // silently would report success for a change that never happened.
                    //
                    // Compare against the *resolved* values, because a stored 0 means
                    // "auto-size from RAM" rather than a literal zero. The GUI seeds its
                    // controls from the resolved numbers, so comparing against the raw 0 made
                    // the very first slider touch claim a reload was needed when nothing had
                    // actually changed.
                    int cur_ctx = config_.context_len ? config_.context_len
                                : (runner_ ? runner_->max_context() : 0);
                    int cur_slots = config_.slots ? config_.slots
                                  : (runner_ ? static_cast<int>(runner_->expert_slots_per_layer()) : 0);
                    int new_ctx   = static_cast<int>(req_json.int_or("context_len", cur_ctx));
                    int new_slots = static_cast<int>(req_json.int_or("slots", cur_slots));
                    requires_reload = (new_ctx != cur_ctx) || (new_slots != cur_slots);
                    config_.context_len = new_ctx;
                    config_.slots = new_slots;
                    applied = config_;
                }

                if (runner_) {
                    runner_->set_eviction_policy(applied.eviction_policy);
                }

                std::ostringstream json;
                json << "{\"status\":\"SUCCESS\","
                     << "\"requires_reload\":" << (requires_reload ? "true" : "false") << ","
                     << "\"config\":{"
                     << "\"temperature\":" << applied.temperature << ","
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

            try {
                if (!ctx_) {
                    ctx_ = std::make_shared<D3D12Context>();
                    ctx_->initialize(false);
                }

                auto manifest = GTurboManifestV1::from_json_string(
                    read_text_file(target_path + "/manifest.json"));
                auto layout = PackedExpertsLayoutV1::from_json_string(
                    read_text_file(target_path + "/packed_experts/layout.json"));
                layout.cross_validate(manifest);

                // Only replace the live runner once the new one has loaded, so a failed
                // load leaves the previously working model in place.
                auto next = std::make_shared<ForwardRunner>(ctx_, manifest, layout, target_path);
                next->initialize();
                runner_ = std::move(next);
                load_error_.clear();

                std::ostringstream json;
                json << "{\"status\":\"SUCCESS\", \"message\":\"Successfully loaded model bundle: " << target_path << "\"}";
                send_json_response(client, json.str());
            } catch (const std::exception& ex) {
                load_error_ = ex.what();
                std::ostringstream json;
                json << "{\"status\":\"ERROR\", \"message\":\"Failed to load model: "
                     << escape_json_string(ex.what()) << "\"}";
                send_json_response(client, json.str(), 500);
            }

        } else if (path == "/api/unload_model") {
            runner_ = nullptr;
            send_json_response(client, "{\"status\":\"SUCCESS\", \"message\":\"Model successfully unloaded from UMA RAM.\"}");

        } else if (path == "/api/stop") {
            if (runner_) {
                runner_->stop_generation();
            }
            send_json_response(client, "{\"status\":\"SUCCESS\", \"message\":\"Generation stop signal sent.\"}");

        } else if (path == "/api/clear_cache") {
            if (runner_) {
                runner_->clear_expert_cache();
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
                           render_models_list(openai_cfg_.model_id), false);
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
    if (!validate_chat_request(body, openai_cfg_, vr, err)) {
        send_err(err);
        return;
    }
    if (!runner_) {
        send_err({503, "No model is loaded.", "server_error", "", "model_not_loaded"});
        return;
    }

    // Admission before tokenization: an over-capacity request should not pay for a prompt it
    // will never run.
    if (!coordinator_) {
        coordinator_ = std::make_unique<RequestCoordinator>(openai_cfg_.queue_limit);
    }
    auto lease = coordinator_->acquire();
    if (!lease) {
        send_err({429, "Server is at capacity; retry shortly.",
                  "server_error", "", "queue_full"});
        return;
    }

    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    const std::string id = "chatcmpl-" + std::to_string(created) + "-" +
                           std::to_string(reinterpret_cast<uintptr_t>(&vr) & 0xFFFFFF);

    if (!vr.stream) {
        try {
            auto result = runner_->generate_chat(vr.messages, vr.options);
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
        result = runner_->generate_chat(
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
    if (!send_event(render_final_chunk(id, vr.model, created, result.reason))) return;
    if (vr.include_usage) {
        if (!send_event(render_usage_chunk(id, vr.model, created, result))) return;
    }
    send_raw(client, "data: [DONE]\n\n");
}

} // namespace gturbo
