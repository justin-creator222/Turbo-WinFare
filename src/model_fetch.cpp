#include "gturbo/model_fetch.hpp"
#include "gturbo/json.hpp"
#include "gturbo/manifest.hpp"

#include <windows.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace gturbo {

const char* fetch_state_name(FetchProgress::State state) {
    switch (state) {
        case FetchProgress::State::Idle:      return "idle";
        case FetchProgress::State::Running:   return "running";
        case FetchProgress::State::Done:      return "done";
        case FetchProgress::State::Failed:    return "failed";
        case FetchProgress::State::Cancelled: return "cancelled";
    }
    return "idle";
}

// ---------------------------------------------------------------------------
// Host probing
// ---------------------------------------------------------------------------

namespace {

// Runs a command and captures its stdout. Used only for `--version` probes, so the output is
// tiny and read in one go.
bool capture_command(const std::string& command_line, std::string& out) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &sa, 0)) return false;
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_end;
    // Python prints its version to stdout on 3.4+, but capture stderr too so an older or
    // broken interpreter still produces something to report.
    si.hStdError = write_end;
    si.hStdInput = nullptr;

    PROCESS_INFORMATION pi{};
    std::string mutable_cmd = command_line;
    const BOOL ok = CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(write_end);
    if (!ok) {
        CloseHandle(read_end);
        return false;
    }

    std::string text;
    std::array<char, 512> buf{};
    DWORD read = 0;
    while (ReadFile(read_end, buf.data(), static_cast<DWORD>(buf.size()), &read, nullptr) &&
           read > 0) {
        text.append(buf.data(), read);
    }
    CloseHandle(read_end);

    WaitForSingleObject(pi.hProcess, 10000);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    out = text;
    return code == 0;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// The converter lives beside the repository root, which is not necessarily the working
// directory: launching by double-click makes build\ the cwd. Search the same roots the
// bundle loader searches, for the same reason.
std::string find_converter() {
    for (const auto& root : bundle_search_roots()) {
        const fs::path candidate = fs::path(root) / "tools" / "convert_hf_to_gturbo.py";
        std::error_code ec;
        if (fs::exists(candidate, ec)) return candidate.string();
    }
    return "";
}

} // namespace

double free_disk_gb_for(const std::string& path) {
    std::error_code ec;
    fs::path target = path.empty() ? fs::current_path(ec) : fs::path(path);
    if (ec) return 0.0;
    // space() needs an existing directory; walk up until one exists.
    while (!target.empty() && !fs::exists(target, ec)) {
        const fs::path parent = target.parent_path();
        if (parent == target) break;
        target = parent;
    }
    const fs::space_info info = fs::space(target, ec);
    if (ec) return 0.0;
    return static_cast<double>(info.available) / (1024.0 * 1024.0 * 1024.0);
}

HostEnvironment probe_host_environment() {
    HostEnvironment env;

    // `py -3` is the Windows launcher and is the more reliable of the two: a bare `python`
    // on a stock Windows resolves to the Store stub, which exits without running anything.
    for (const char* cmd : {"py -3", "python", "python3"}) {
        std::string out;
        if (capture_command(std::string(cmd) + " --version", out)) {
            const std::string text = trim(out);
            if (text.rfind("Python ", 0) == 0) {
                env.python_available = true;
                env.python_command = cmd;
                env.python_version = trim(text.substr(7));
                break;
            }
        }
    }

    env.converter_path = find_converter();
    env.converter_present = !env.converter_path.empty();
    env.free_disk_gb = free_disk_gb_for("");
    return env;
}

// ---------------------------------------------------------------------------
// ModelFetcher
// ---------------------------------------------------------------------------

ModelFetcher::~ModelFetcher() {
    cancel();
    if (reader_.joinable()) reader_.join();
    if (process_handle_) {
        CloseHandle(static_cast<HANDLE>(process_handle_));
        process_handle_ = nullptr;
    }
}

bool ModelFetcher::valid_output_name(const std::string& output, std::string& why) {
    if (output.empty()) {
        why = "A bundle name is required.";
        return false;
    }
    if (output.size() > 128) {
        why = "The bundle name is too long.";
        return false;
    }
    // A bare name only. This string becomes a directory the server creates, and it arrives
    // over HTTP -- a separator, a drive letter or a ".." would place it anywhere on disk.
    for (const char c : output) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
        if (!ok) {
            why = "The bundle name may contain only letters, digits, '.', '-' and '_'.";
            return false;
        }
    }
    if (output.find("..") != std::string::npos) {
        why = "The bundle name may not contain '..'.";
        return false;
    }
    if (output.size() <= 7 || output.compare(output.size() - 7, 7, ".gturbo") != 0) {
        why = "The bundle name must end in '.gturbo'.";
        return false;
    }
    return true;
}

bool ModelFetcher::start(const std::string& output, const std::string& hf_token, bool resume,
                         std::string& error) {
    if (running_) {
        error = "A model download is already running.";
        return false;
    }
    if (!valid_output_name(output, error)) return false;

    const HostEnvironment env = probe_host_environment();
    if (!env.python_available) {
        error = "Python 3 was not found on PATH. Install Python 3.10 or newer, then retry.";
        return false;
    }
    if (!env.converter_present) {
        error = "tools/convert_hf_to_gturbo.py was not found next to the executable.";
        return false;
    }

    // Reap a previous finished run before starting another.
    if (reader_.joinable()) reader_.join();
    if (process_handle_) {
        CloseHandle(static_cast<HANDLE>(process_handle_));
        process_handle_ = nullptr;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &sa, 0)) {
        error = "Could not create a pipe for the converter's output.";
        return false;
    }
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);

    std::ostringstream cmd;
    cmd << env.python_command << " \"" << env.converter_path << "\""
        << " --output \"" << output << "\""
        << " --progress-json";
    if (resume) cmd << " --resume";

    // The token goes in the child's environment, never on the command line: a command line
    // is readable by every process on the machine (Task Manager shows it), and it would also
    // land in any crash dump or process log.
    std::string env_block;
    {
        const LPCH parent = GetEnvironmentStrings();
        if (parent) {
            for (LPCH p = parent; *p; ) {
                const size_t len = std::strlen(p);
                const std::string entry(p, len);
                // Drop any inherited HF_TOKEN so a stale one cannot win.
                if (entry.rfind("HF_TOKEN=", 0) != 0) {
                    env_block.append(entry);
                    env_block.push_back('\0');
                }
                p += len + 1;
            }
            FreeEnvironmentStrings(parent);
        }
        if (!hf_token.empty()) {
            env_block.append("HF_TOKEN=" + hf_token);
            env_block.push_back('\0');
        }
        // Unbuffered stdout, so progress lines arrive as they are written rather than in
        // 8 KB blocks -- a pipe is not a terminal, so Python block-buffers it by default and
        // the GUI would sit at 0% for minutes.
        env_block.append("PYTHONUNBUFFERED=1");
        env_block.push_back('\0');
        env_block.push_back('\0');
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_end;
    si.hStdError = write_end;
    si.hStdInput = nullptr;

    PROCESS_INFORMATION pi{};
    std::string mutable_cmd = cmd.str();
    const BOOL ok = CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, env_block.data(), nullptr, &si, &pi);
    CloseHandle(write_end);
    if (!ok) {
        CloseHandle(read_end);
        error = "Could not start the converter process.";
        return false;
    }
    CloseHandle(pi.hThread);
    process_handle_ = pi.hProcess;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_ = FetchProgress{};
        progress_.state = FetchProgress::State::Running;
        progress_.output = output;
        progress_.stage = "Starting the converter...";
        recent_.clear();
    }
    cancel_requested_ = false;
    running_ = true;
    reader_ = std::thread(&ModelFetcher::reader_loop, this, read_end);
    return true;
}

void ModelFetcher::cancel() {
    if (!running_) return;
    cancel_requested_ = true;
    if (process_handle_) {
        // The child holds an <output>.partial directory that a later --resume run reuses, so
        // terminating it costs the transfer in flight, not the whole 14.6 GB.
        TerminateProcess(static_cast<HANDLE>(process_handle_), 1);
    }
}

void ModelFetcher::reader_loop(void* read_handle) {
    HANDLE read_end = static_cast<HANDLE>(read_handle);
    std::string buffer;
    std::array<char, 4096> chunk{};
    DWORD read = 0;

    while (ReadFile(read_end, chunk.data(), static_cast<DWORD>(chunk.size()), &read, nullptr) &&
           read > 0) {
        buffer.append(chunk.data(), read);
        size_t nl;
        while ((nl = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, nl);
            buffer.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            handle_line(line);
        }
    }
    if (!buffer.empty()) handle_line(buffer);
    CloseHandle(read_end);

    DWORD code = 1;
    if (process_handle_) {
        WaitForSingleObject(static_cast<HANDLE>(process_handle_), INFINITE);
        GetExitCodeProcess(static_cast<HANDLE>(process_handle_), &code);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_.exit_code = static_cast<int>(code);
        if (cancel_requested_) {
            progress_.state = FetchProgress::State::Cancelled;
            progress_.message = "Cancelled. The partial download was kept -- start again "
                                "with Resume to continue from where it stopped.";
        } else if (code == 0) {
            progress_.state = FetchProgress::State::Done;
            progress_.pct = 100.0;
            if (progress_.message.empty()) progress_.message = "Bundle ready.";
        } else if (progress_.state != FetchProgress::State::Failed) {
            // The child died without emitting an error event.
            progress_.state = FetchProgress::State::Failed;
            if (progress_.message.empty()) {
                progress_.message = "The converter exited with code " + std::to_string(code) +
                                    ".";
            }
        }
    }
    running_ = false;
}

void ModelFetcher::handle_line(const std::string& line) {
    const std::string trimmed = trim(line);
    if (trimmed.empty()) return;

    // Anything that is not a JSON object is the converter's human-readable output. Keep a
    // short tail of it: when the child dies without emitting an error event, this is the
    // only evidence of why.
    if (trimmed.front() != '{') {
        std::lock_guard<std::mutex> lock(mutex_);
        recent_.push_back(trimmed);
        if (recent_.size() > 20) recent_.erase(recent_.begin());
        return;
    }

    JsonValue obj;
    try {
        obj = JsonValue::parse(trimmed);
    } catch (const std::exception&) {
        // Not our protocol after all -- keep it as diagnostic text.
    }
    if (!obj.is_object()) {
        std::lock_guard<std::mutex> lock(mutex_);
        recent_.push_back(trimmed);
        if (recent_.size() > 20) recent_.erase(recent_.begin());
        return;
    }

    const std::string event = obj.string_or("event", "");
    std::lock_guard<std::mutex> lock(mutex_);

    if (event == "stage") {
        progress_.stage = obj.string_or("message", "");
        progress_.step = static_cast<int>(obj.int_or("step", 0));
        progress_.steps = static_cast<int>(obj.int_or("steps", 0));
        // A new stage restarts the byte counter; leaving the old one showing made a finished
        // stage look like it was still running.
        progress_.bytes_done = 0;
        progress_.bytes_total = 0;
        progress_.pct = 0.0;
    } else if (event == "progress") {
        progress_.label = obj.string_or("label", "");
        progress_.bytes_done = static_cast<uint64_t>(obj.int_or("done", 0));
        progress_.bytes_total = static_cast<uint64_t>(obj.int_or("total", 0));
        progress_.pct = obj.double_or("pct", 0.0);
        progress_.rate_mbs = obj.double_or("rate_mbs", 0.0);
        progress_.eta_s = static_cast<int>(obj.int_or("eta_s", 0));
    } else if (event == "error") {
        progress_.state = FetchProgress::State::Failed;
        progress_.message = obj.string_or("message", "The converter reported an error.");
    } else if (event == "cancelled") {
        progress_.state = FetchProgress::State::Cancelled;
        progress_.message = obj.string_or("message", "Cancelled.");
    } else if (event == "done") {
        progress_.state = FetchProgress::State::Done;
        progress_.pct = 100.0;
        progress_.message = "Bundle ready.";
    }
}

FetchProgress ModelFetcher::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

std::vector<std::string> ModelFetcher::recent_output() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recent_;
}

} // namespace gturbo
