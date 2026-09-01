#pragma once

// Model acquisition: drives tools/convert_hf_to_gturbo.py as a child process and reports its
// progress, so a user with no bundle can get one without leaving the GUI.
//
// Why a subprocess rather than a native downloader. Building a bundle is not a download: it
// streams ~14.6 GB of MLX-affine safetensors from a pinned HuggingFace revision and repacks
// it into the .gturbo layout, 16 KB-aligned, byte-for-byte, with a manifest of sha256s. That
// logic already exists, is tested (tools/test_convert_streaming.py), and is the only
// implementation that has ever produced a bundle this engine can load. Reimplementing it in
// C++ would mean a second copy of the packing plan that must stay bit-identical to the first
// one forever, plus a TLS stack the engine does not otherwise link. Reusing it costs a
// dependency on Python, which this project already requires for its toolchain bootstrap.
//
// The child speaks --progress-json: one JSON object per line on stdout.

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gturbo {

// What the host environment can support, for GET /api/server_info. The GUI disables the
// download button with the specific unmet reason rather than letting it fail later.
struct HostEnvironment {
    bool python_available{false};
    std::string python_command;   // e.g. "py -3" or "python"
    std::string python_version;   // e.g. "3.12.4"
    bool converter_present{false};
    std::string converter_path;
    double free_disk_gb{0.0};
};

// Probes for Python and the converter script. Runs `python --version`, so it is not free;
// the server caches the result rather than probing per request.
HostEnvironment probe_host_environment();

// Approximate free space on the volume holding `path`, in GB. 0 when it cannot be determined.
double free_disk_gb_for(const std::string& path);

struct FetchProgress {
    enum class State { Idle, Running, Done, Failed, Cancelled };

    State state{State::Idle};
    std::string output;        // target bundle directory
    std::string stage;         // human-readable stage message
    int step{0};
    int steps{0};
    std::string label;         // which region is streaming ("resident", "experts", ...)
    uint64_t bytes_done{0};
    uint64_t bytes_total{0};
    double pct{0.0};
    double rate_mbs{0.0};
    int eta_s{0};
    std::string message;       // failure reason, or the closing note
    int exit_code{0};
};

const char* fetch_state_name(FetchProgress::State state);

class ModelFetcher {
public:
    ModelFetcher() = default;
    ~ModelFetcher();

    ModelFetcher(const ModelFetcher&) = delete;
    ModelFetcher& operator=(const ModelFetcher&) = delete;

    // Validates `output` as a bare bundle name: letters, digits, dot, dash and underscore,
    // ending in ".gturbo". No separator, no "..", no drive letter -- this string names a
    // directory the server will create, and it arrives over HTTP.
    static bool valid_output_name(const std::string& output, std::string& why);

    // Starts a conversion. Returns false and fills `error` when one is already running, the
    // name is rejected, or Python or the converter cannot be found.
    bool start(const std::string& output, const std::string& hf_token, bool resume,
               std::string& error);

    // Asks the child to stop. The <output>.partial directory is left in place so a later run
    // with resume=true continues rather than restarting a 14.6 GB transfer.
    void cancel();

    bool is_running() const { return running_; }
    FetchProgress snapshot() const;

    // The last few non-JSON lines the child printed, for diagnosing a failure whose JSON
    // error event never arrived (a crash, or a Python that could not start).
    std::vector<std::string> recent_output() const;

private:
    void reader_loop(void* read_handle);
    void handle_line(const std::string& line);

    mutable std::mutex mutex_;
    FetchProgress progress_;
    std::vector<std::string> recent_;

    std::atomic<bool> running_{false};
    std::atomic<bool> cancel_requested_{false};
    std::thread reader_;
    void* process_handle_{nullptr};   // HANDLE
};

} // namespace gturbo
