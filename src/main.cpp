#include "gturbo/format.hpp"
#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/d3d12_context.hpp"
#include "gturbo/runner.hpp"
#include "gturbo/cpu_reference.hpp"
#include "gturbo/server.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <string>
#include <memory>
#include <filesystem>
#include <windows.h>
#include <shellapi.h>

// Where to look for the default bundle, in order, when --model was not given.
//
// The executable lives in build\ but the bundle lives at the repo root, so launching by
// double-click -- which makes build\ the working directory -- resolved the relative default
// against the wrong place. Worse, a stale placeholder bundle left in build\ by the retired
// repacker made that failure look like a corrupt model rather than a wrong path.
//
// Each candidate is validated by actually parsing it, not by checking that the files exist:
// the stale placeholder has both manifest.json and layout.json and still cannot be loaded,
// so an existence check would keep choosing it.
static bool bundle_loads(const std::string& dir) {
    try {
        auto manifest = gturbo::GTurboManifestV1::from_json_string(
            gturbo::read_text_file(dir + "/manifest.json"));
        auto layout = gturbo::PackedExpertsLayoutV1::from_json_string(
            gturbo::read_text_file(dir + "/packed_experts/layout.json"));
        layout.cross_validate(manifest);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static std::string resolve_default_model_path(const std::string& name) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;

    std::error_code ec;
    candidates.push_back(fs::current_path(ec) / name);

    // Next to the executable, then one level up -- that is build\ and the repo root.
    wchar_t exe_buf[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, exe_buf, MAX_PATH) > 0) {
        const fs::path exe_dir = fs::path(exe_buf).parent_path();
        candidates.push_back(exe_dir / name);
        candidates.push_back(exe_dir.parent_path() / name);
    }

    for (const auto& c : candidates) {
        if (fs::is_directory(c, ec) && bundle_loads(c.string())) {
            return c.string();
        }
    }
    // Nothing usable. Return the first candidate so the error names the place the user most
    // likely meant, rather than a directory they never mentioned.
    return candidates.empty() ? name : candidates.front().string();
}

// Ctrl-C asks the runner to stop between tokens instead of killing the process, so the
// generation ends cleanly and its stop reason (and exit code) are meaningful.
static gturbo::ForwardRunner* g_active_runner = nullptr;

static BOOL WINAPI console_ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        if (g_active_runner) {
            g_active_runner->stop_generation();
            return TRUE;
        }
    }
    return FALSE;
}

// Two decimals, so a 0.3 ms change is visible instead of rounding away.
static std::string metrics_fmt(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v;
    return ss.str();
}

static void print_usage() {
    std::cout <<
        "Usage: turbo-winfare [options]\n"
        "\n"
        "  --model <dir>        .gturbo bundle directory (default gemma-4-26b-a4b.gturbo)\n"
        "  --prompt <text>      Run one CLI generation instead of starting the GUI\n"
        "  --gui                Start the GUI (default when --prompt is absent)\n"
        "  --cpu                Use the scalar FP32 reference path; requires --prompt\n"
        "  --max-tokens <n>     Completion token budget (default 32)  [alias: --max-new]\n"
        "  --temperature <f>    0 = greedy (default)\n"
        "  --top-p <f>          Nucleus mass in (0,1]; needs --top-k when < 1\n"
        "  --top-k <n>          1..256, or 0 to disable\n"
        "  --repetition-penalty <f>  > 0; 1 disables\n"
        "  --seed <n>           Makes sampled output reproducible\n"
        "  --stop <text>        Stop string; repeatable\n"
        "  --quiet              Suppress the performance footer\n"
        "  --context <n>        Max context in tokens; 0 auto-sizes from installed RAM\n"
        "                       [alias: --max-context]\n"
        "  --slots <n>          Expert cache slots per layer; 0 auto-sizes from RAM\n"
        "                       [alias: --expert-cache-slots]\n"
        "  --dump-tensors <dir> Write per-stage FP32 tensors for the first token (--cpu only)\n"
        "  --port <n>           HTTP port (default 8080)\n"
        "  --serve              Start the server without opening a browser\n"
        "  --model-id <name>    Model name the OpenAI endpoints expect\n"
        "  --queue-limit <n>    Requests allowed to wait for the engine (default 4)\n"
        "  --no-open            Do not launch a browser with the GUI\n"
        "  --help               Show this message\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================================\n";
    std::cout << " Turbo-WinFare: Native Windows AMD Engine & GUI App      \n";
    std::cout << " Optimized for Lenovo Legion Go S (Ryzen Z1 / Radeon 780M)\n";
    std::cout << " Gemma 4 26B-A4B MoE SSD Streaming Architecture (DirectX 12)\n";
    std::cout << "=========================================================\n\n";

    std::string model_path = "gemma-4-26b-a4b.gturbo";
    bool model_path_explicit = false;
    std::string prompt;
    std::string dump_dir;
    bool launch_gui = true;
    bool cpu_mode = false;
    int max_tokens = 32;
    float temperature = 0.0f;
    // top_p/top_k only take effect once temperature > 0; these match the reference's
    // defaults so `--temperature 0.7` alone behaves the way the reference would.
    float top_p = 0.95f;
    int top_k = 64;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    bool has_seed = false;
    std::vector<std::string> stop_strings;
    bool quiet = false;
    gturbo::OpenAIServerConfig openai_cfg{};
    size_t expert_slots = 0;   // 0 = auto-size from RAM
    int max_context = 0;       // 0 = auto-size from RAM
    uint16_t gui_port = 8080;
    bool auto_open_browser = true;

    // A flag that needs a value but is last on the line used to fall through silently, as did
    // any misspelling -- so `--tempature 0.8` ran at the default and looked like it worked.
    auto need_value = [&](const std::string& flag, int i) -> bool {
        if (i + 1 < argc) return true;
        std::cerr << "Error: " << flag << " requires a value\n";
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--model") {
            if (!need_value(arg, i)) return 2;
            model_path = argv[++i];
            model_path_explicit = true;
        } else if (arg == "--prompt") {
            if (!need_value(arg, i)) return 2;
            prompt = argv[++i];
            if (prompt.empty()) {
                std::cerr << "Error: --prompt requires a non-empty string\n";
                return 2;
            }
            launch_gui = false;
        } else if (arg == "--gui") {
            launch_gui = true;
        } else if (arg == "--cpu") {
            // Scalar FP32 reference path: no D3D12, no shaders. Slow but correct, and the
            // ground truth the GPU kernels are diffed against.
            cpu_mode = true;
            launch_gui = false;
        } else if (arg == "--dump-tensors") {
            if (!need_value(arg, i)) return 2;
            dump_dir = argv[++i];
        // The `--max-context` / `--expert-cache-slots` / `--max-new` spellings are the Swift
        // reference's names. They are accepted so a command line written for the reference
        // runs here unchanged; ours stay canonical in --help.
        } else if (arg == "--context" || arg == "--max-context") {
            if (!need_value(arg, i)) return 2;
            // Max context in tokens; 0 (default) auto-sizes from installed RAM.
            max_context = std::stoi(argv[++i]);
        } else if (arg == "--slots" || arg == "--expert-cache-slots") {
            if (!need_value(arg, i)) return 2;
            // Expert cache slots per layer; 0 (default) auto-sizes from installed RAM.
            expert_slots = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--max-tokens" || arg == "--max-new") {
            if (!need_value(arg, i)) return 2;
            max_tokens = std::stoi(argv[++i]);
            if (max_tokens <= 0) {
                std::cerr << "Error: --max-tokens must be greater than 0\n";
                return 2;
            }
        } else if (arg == "--temperature") {
            if (!need_value(arg, i)) return 2;
            temperature = std::stof(argv[++i]);
            if (temperature < 0.0f) {
                std::cerr << "Error: --temperature must be >= 0 (0 = greedy)\n";
                return 2;
            }
        } else if (arg == "--top-p") {
            if (!need_value(arg, i)) return 2;
            top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k") {
            if (!need_value(arg, i)) return 2;
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--repetition-penalty") {
            if (!need_value(arg, i)) return 2;
            repetition_penalty = std::stof(argv[++i]);
        } else if (arg == "--seed") {
            if (!need_value(arg, i)) return 2;
            seed = std::stoull(argv[++i]);
            has_seed = true;
        } else if (arg == "--stop") {
            if (!need_value(arg, i)) return 2;
            stop_strings.emplace_back(argv[++i]);
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--port") {
            if (!need_value(arg, i)) return 2;
            gui_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--serve") {
            // Headless: same server, no browser. This is how the OpenAI endpoints are meant
            // to be run.
            launch_gui = true;
            auto_open_browser = false;
        } else if (arg == "--model-id") {
            if (!need_value(arg, i)) return 2;
            openai_cfg.model_id = argv[++i];
        } else if (arg == "--queue-limit") {
            if (!need_value(arg, i)) return 2;
            openai_cfg.queue_limit = std::stoi(argv[++i]);
            if (openai_cfg.queue_limit < 0) {
                std::cerr << "Error: --queue-limit must be >= 0\n";
                return 2;
            }
        } else if (arg == "--no-open") {
            auto_open_browser = false;
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'\n\n";
            print_usage();
            return 2;
        }
    }

    // An explicit --model is taken literally -- a search that second-guessed it would load a
    // different model than the one asked for.
    if (!model_path_explicit) {
        model_path = resolve_default_model_path(model_path);
    }

    try {
        // The CPU reference path is entirely self-contained: it never creates a D3D12
        // device or compiles a shader, so it works even when the GPU path is broken.
        if (cpu_mode) {
            if (prompt.empty()) {
                std::cerr << "Error: --cpu requires --prompt\n";
                return 2;
            }
            std::cout << "[CPU reference] Loading " << model_path << "...\n";
            auto manifest = gturbo::GTurboManifestV1::from_json_string(
                gturbo::read_text_file(model_path + "/manifest.json"));
            auto layout = gturbo::PackedExpertsLayoutV1::from_json_string(
                gturbo::read_text_file(model_path + "/packed_experts/layout.json"));
            layout.cross_validate(manifest);

            gturbo::CpuReference ref(manifest, layout, model_path);
            ref.load();

            gturbo::CpuGenerationOptions opts{};
            opts.max_tokens = max_tokens;
            opts.temperature = temperature;
            opts.top_p = top_p;
            opts.top_k = top_k;
            opts.repetition_penalty = repetition_penalty;
            opts.seed = has_seed ? seed : 0;
            opts.verbose = true;
            opts.dump_dir = dump_dir;

            std::cout << "[CPU reference] Prompt: \"" << prompt << "\"\n\n";
            auto start = std::chrono::high_resolution_clock::now();
            std::string text = ref.generate_text(prompt, opts);
            auto elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start).count();

            std::cout << "\n================ Output ================\n";
            std::cout << text << "\n";
            std::cout << "=======================================\n";
            std::cout << "(" << elapsed << " s)\n";
            return 0;
        }

        std::cout << "[1/4] Initializing DirectX 12 Compute Context...\n";
        auto ctx = std::make_shared<gturbo::D3D12Context>();
        ctx->initialize(false);

        std::cout << "      GPU Adapter:    " << ctx->adapter_name() << "\n";
        std::cout << "      Dedicated VRAM: " << (ctx->dedicated_video_memory() / (1024 * 1024)) << " MB\n";
        std::cout << "      Shared RAM:     " << (ctx->shared_system_memory() / (1024 * 1024)) << " MB\n\n";

        std::cout << "[2/4] Loading GTURBO Format Specifications...\n";
        std::shared_ptr<gturbo::ForwardRunner> runner;
        std::string load_error;
        // Reading the manifest and layout is INSIDE this try, not before it. A bundle whose
        // metadata fails to parse -- a stale placeholder, a half-finished conversion, the
        // wrong directory -- used to throw out to main() and exit(1). Launched by
        // double-clicking, that closes the console instantly and the user sees a window flash
        // and nothing else, which is precisely the situation the load-error banner exists for.
        try {
            // Read the bundle's own metadata. These used to be from_json_string("") calls that
            // ignored their argument and returned hardcoded structs, so a bundle's real stride
            // and layer mask were never consulted.
            auto manifest = gturbo::GTurboManifestV1::from_json_string(
                gturbo::read_text_file(model_path + "/manifest.json"));
            auto layout = gturbo::PackedExpertsLayoutV1::from_json_string(
                gturbo::read_text_file(model_path + "/packed_experts/layout.json"));
            layout.cross_validate(manifest);

            int full_layers = 0;
            for (int m : manifest.arch.full_attention_layer_mask) full_layers += m;

            std::cout << "      Model ID: " << manifest.model_id << "\n";
            std::cout << "      Layers: " << manifest.arch.num_layers
                      << " (" << full_layers << " full-attention, "
                      << (manifest.arch.num_layers - full_layers) << " sliding-window)\n";
            std::cout << "      Experts/Layer: " << manifest.arch.num_experts
                      << ", top-K: " << manifest.arch.top_k_experts << "\n";
            std::cout << "      Hidden: " << manifest.arch.hidden_size
                      << ", Vocab: " << manifest.arch.vocab_size << "\n";
            std::cout << "      Expert Block Stride: " << manifest.expert_stride << " bytes\n\n";

            std::cout << "[3/4] Initializing Streamer & UMA Expert DRAM Cache...\n";
            runner = std::make_shared<gturbo::ForwardRunner>(ctx, manifest, layout, model_path);
            runner->set_expert_slots(expert_slots);
            runner->set_max_context(max_context);
            runner->initialize();
            // Expert streamers open lazily per layer, so nothing is allocated yet -- the old
            // message here claimed a 166 MB slot pool had already been reserved.
            std::cout << "      Expert streamers open lazily on first use (8 UMA slots/layer)\n\n";
        } catch (const std::exception& ex) {
            // In GUI mode a bad model must not prevent the server from starting -- otherwise
            // the user gets a closed browser tab and no explanation. Report it here and let
            // the GUI show NO MODEL; /api/generate will refuse with the same detail.
            // In CLI mode there is nothing useful to do without a model, so rethrow.
            runner = nullptr;
            load_error = ex.what();
            // Print the RESOLVED path. A relative default plus whatever directory the
            // executable happened to be launched from is the most common cause of this, and
            // "gemma-4-26b-a4b.gturbo" on its own does not reveal which copy was opened.
            std::string resolved = model_path;
            try {
                resolved = std::filesystem::absolute(model_path).string();
            } catch (const std::exception&) {
                // Keep the unresolved path; this is already the error path.
            }
            std::cout << "      FAILED to load model:\n"
                      << "        path:   " << resolved << "\n"
                      << "        reason: " << load_error << "\n\n";
            if (!launch_gui) throw;
        }

        if (launch_gui) {
            std::cout << "[4/4] Starting GUI Desktop Web Bridge at http://localhost:" << gui_port << "...\n";
            auto server = std::make_unique<gturbo::HTTPServer>();
            if (!load_error.empty()) {
                server->set_load_error(load_error);
            }
            openai_cfg.max_context = runner ? runner->max_context() : 4096;
            server->set_openai_config(openai_cfg);
            server->start(gui_port, runner, ctx);

            std::cout << "      OpenAI-compatible API at http://127.0.0.1:" << gui_port
                      << "/v1  (model id: " << openai_cfg.model_id << ")\n";

            if (auto_open_browser) {
                std::wstring url = L"http://localhost:" + std::to_wstring(gui_port);
                ShellExecuteW(NULL, L"open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
            }

            std::cout << "\n=========================================================\n";
            std::cout << " GUI Application Running! Opening browser window...\n";
            std::cout << " Press Ctrl+C in this console window to exit.\n";
            std::cout << "=========================================================\n";

            while (server->is_running()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        } else {
            std::cout << "[4/4] Executing CLI Forward Pass for Prompt: \"" << prompt << "\"\n\n";
            gturbo::GenerationOptions opts{};
            // --max-tokens and --temperature were parsed but only ever reached the CPU path;
            // the GPU path hardcoded a 32-token budget, so the flag appeared to do nothing.
            opts.max_tokens = max_tokens;
            opts.temperature = temperature;
            opts.top_p = top_p;
            opts.top_k = top_k;
            opts.repetition_penalty = repetition_penalty;
            opts.has_seed = has_seed;
            opts.seed = seed;
            opts.stop_strings = stop_strings;

            // Ctrl-C stops generation rather than killing the process, so the metrics below
            // still print and the exit code can say why it ended.
            g_active_runner = runner.get();
            SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

            std::cout << "================ Generation Results ================\n";
            // Tokens go to stdout as they are produced. This is the same callback the HTTP
            // layer uses for SSE -- one streaming mechanism, not two.
            auto result = runner->generate_chat(
                {{"user", prompt}}, opts,
                [](const gturbo::StreamEvent& ev) {
                    if (ev.kind != gturbo::StreamEvent::Kind::Prefill && !ev.delta.empty()) {
                        std::cout << ev.delta << std::flush;
                    }
                    return true;
                });
            std::cout << "\n\n";

            if (result.reason == gturbo::StopReason::Cancelled) {
                // Matches the reference's CLI: 130 is the conventional "terminated by SIGINT".
                std::cerr << "[cancelled]\n";
                return 130;
            }
            if (quiet) return 0;

            const auto& m = runner->metrics();
            const double tok = static_cast<double>(m.tokens_measured ? m.tokens_measured : 1);
            const double io_mb = static_cast<double>(m.total_io_bytes) / (1024.0 * 1024.0);

            std::cout << "================ Performance Metrics ===============\n";
            std::cout << "Prefill Speed:  " << metrics_fmt(m.prefill_tokens_per_sec) << " tok/s\n";
            std::cout << "Decode Speed:   " << metrics_fmt(m.decode_tokens_per_sec) << " tok/s\n";
            std::cout << "Total Time:     " << metrics_fmt(m.total_time_ms) << " ms over "
                      << m.tokens_measured << " forward passes\n";
            std::cout << "---------------- Per-token breakdown ---------------\n";
            // Mirrors the reference's decode decomposition (docs/BENCHMARKS.md:38-52) so
            // each optimization can be attributed to a phase instead of guessed at.
            std::cout << "  expert I/O:   " << metrics_fmt(m.expert_io_ms / tok) << " ms  ("
                      << metrics_fmt(100.0 * m.expert_io_ms / m.total_time_ms) << "%)\n";
            std::cout << "  GPU wait:     " << metrics_fmt(m.gpu_wait_ms / tok) << " ms  ("
                      << metrics_fmt(100.0 * m.gpu_wait_ms / m.total_time_ms) << "%)  in "
                      << (m.gpu_waits / static_cast<uint64_t>(tok)) << " fences/token\n";
            std::cout << "  LM head:      " << metrics_fmt(m.lm_head_ms / tok) << " ms  ("
                      << metrics_fmt(100.0 * m.lm_head_ms / m.total_time_ms) << "%)\n";
            std::cout << "  CPU other:    " << metrics_fmt(m.cpu_other_ms / tok) << " ms  ("
                      << metrics_fmt(100.0 * m.cpu_other_ms / m.total_time_ms) << "%)\n";
            std::cout << "---------------- Expert streaming ------------------\n";
            std::cout << "  read:         " << metrics_fmt(io_mb / tok) << " MB/token, "
                      << io_mb << " MB total in " << m.total_io_calls << " reads\n";
            std::cout << "  throughput:   "
                      << metrics_fmt(m.expert_io_ms > 0 ? io_mb / (m.expert_io_ms / 1000.0) : 0.0)
                      << " MB/s while blocked\n";
            std::cout << "  cache hit:    " << metrics_fmt(runner->expert_cache_hit_rate()) << " %\n";
            std::cout << "====================================================\n\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Engine Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
