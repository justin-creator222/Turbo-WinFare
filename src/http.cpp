#include <winsock2.h>
#include <ws2tcpip.h>

#include "gturbo/http.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace gturbo {

namespace {

std::string lower(std::string_view s) {
    std::string o(s);
    std::transform(o.begin(), o.end(), o.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return o;
}

std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    return std::string(s.substr(a, b - a));
}

// Reads more bytes onto `buf`. Returns false on close or error.
bool recv_more(SOCKET s, std::string& buf) {
    char chunk[8192];
    const int n = recv(s, chunk, sizeof(chunk), 0);
    if (n <= 0) return false;
    buf.append(chunk, static_cast<size_t>(n));
    return true;
}

} // namespace

std::string HttpRequest::header(std::string_view name) const {
    const std::string want = lower(name);
    for (const auto& [k, v] : headers) {
        if (lower(k) == want) return v;
    }
    return {};
}

const char* http_reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default:  return "Unknown";
    }
}

HttpReadResult read_http_request(uintptr_t socket_handle, HttpRequest& out,
                                 size_t max_body_bytes, int timeout_ms) {
    SOCKET s = static_cast<SOCKET>(socket_handle);
    out = HttpRequest{};

    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));

    // --- Headers: read until the blank line that terminates them. --------------------
    std::string buf;
    size_t header_end = std::string::npos;
    while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > 64 * 1024) return HttpReadResult::TooLarge;   // absurd header block
        if (!recv_more(s, buf)) {
            if (buf.empty()) return HttpReadResult::Closed;
            const int err = WSAGetLastError();
            return (err == WSAETIMEDOUT) ? HttpReadResult::Timeout : HttpReadResult::Malformed;
        }
    }

    std::istringstream head(buf.substr(0, header_end));
    std::string line;
    if (!std::getline(head, line)) return HttpReadResult::Malformed;

    {
        std::istringstream rl(line);
        std::string version;
        if (!(rl >> out.method >> out.path >> version)) return HttpReadResult::Malformed;
        // HTTP/1.1 keeps connections alive unless told otherwise; 1.0 is the reverse.
        out.keep_alive = (version.rfind("HTTP/1.1", 0) == 0);
    }

    if (const size_t q = out.path.find('?'); q != std::string::npos) {
        out.query = out.path.substr(q + 1);
        out.path.resize(q);
    }

    while (std::getline(head, line)) {
        if (line == "\r" || line.empty()) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        out.headers.emplace_back(trim(line.substr(0, colon)), trim(line.substr(colon + 1)));
    }

    if (const std::string conn = lower(out.header("connection")); !conn.empty()) {
        if (conn.find("close") != std::string::npos) out.keep_alive = false;
        else if (conn.find("keep-alive") != std::string::npos) out.keep_alive = true;
    }

    std::string rest = buf.substr(header_end + 4);

    // --- Body ------------------------------------------------------------------------
    const std::string encoding = lower(out.header("transfer-encoding"));
    if (encoding.find("chunked") != std::string::npos) {
        // Chunked: <hex size>\r\n<data>\r\n ... terminated by a zero-length chunk.
        std::string decoded;
        size_t pos = 0;
        for (;;) {
            size_t eol;
            while ((eol = rest.find("\r\n", pos)) == std::string::npos) {
                if (!recv_more(s, rest)) return HttpReadResult::Malformed;
            }
            const std::string size_line = rest.substr(pos, eol - pos);
            char* endp = nullptr;
            const unsigned long chunk_size = std::strtoul(size_line.c_str(), &endp, 16);
            if (endp == size_line.c_str()) return HttpReadResult::Malformed;
            pos = eol + 2;

            if (chunk_size == 0) break;
            if (decoded.size() + chunk_size > max_body_bytes) return HttpReadResult::TooLarge;

            while (rest.size() < pos + chunk_size + 2) {
                if (!recv_more(s, rest)) return HttpReadResult::Malformed;
            }
            decoded.append(rest, pos, chunk_size);
            pos += chunk_size + 2;      // skip the trailing CRLF
        }
        out.body = std::move(decoded);
    } else {
        const std::string len_header = out.header("content-length");
        if (!len_header.empty()) {
            char* endp = nullptr;
            const unsigned long long want = std::strtoull(len_header.c_str(), &endp, 10);
            if (endp == len_header.c_str()) return HttpReadResult::Malformed;
            if (want > max_body_bytes) return HttpReadResult::TooLarge;

            while (rest.size() < want) {
                if (!recv_more(s, rest)) {
                    const int err = WSAGetLastError();
                    return (err == WSAETIMEDOUT) ? HttpReadResult::Timeout
                                                 : HttpReadResult::Malformed;
                }
            }
            // A Content-Length shorter than what arrived means the surplus belongs to the
            // next pipelined request; this reader handles one request per call, so it is
            // simply not consumed here.
            out.body = rest.substr(0, static_cast<size_t>(want));
        }
        // No Content-Length and no chunking: no body. Not an error -- that is every GET.
    }

    // Only enforced when there is a body to interpret, so GET and OPTIONS are unaffected.
    if (!out.body.empty() && (out.method == "POST" || out.method == "PUT")) {
        const std::string ctype = lower(out.header("content-type"));
        if (!ctype.empty() && ctype.find("application/json") == std::string::npos) {
            return HttpReadResult::UnsupportedMediaType;
        }
    }

    return HttpReadResult::Ok;
}

bool send_raw(uintptr_t socket_handle, std::string_view data) {
    SOCKET s = static_cast<SOCKET>(socket_handle);
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = send(s, data.data() + sent,
                           static_cast<int>(data.size() - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool send_http_headers(uintptr_t socket, int status, std::string_view content_type,
                       const std::vector<std::pair<std::string, std::string>>& extra) {
    std::ostringstream h;
    h << "HTTP/1.1 " << status << " " << http_reason_phrase(status) << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Cache-Control: no-cache\r\n"
      << "Connection: keep-alive\r\n";
    for (const auto& [k, v] : extra) h << k << ": " << v << "\r\n";
    h << "\r\n";
    return send_raw(socket, h.str());
}

bool send_http_response(uintptr_t socket, int status, std::string_view content_type,
                        std::string_view body, bool keep_alive,
                        const std::vector<std::pair<std::string, std::string>>& extra) {
    std::ostringstream h;
    h << "HTTP/1.1 " << status << " " << http_reason_phrase(status) << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n";
    for (const auto& [k, v] : extra) h << k << ": " << v << "\r\n";
    h << "\r\n";

    if (!send_raw(socket, h.str())) return false;
    return body.empty() || send_raw(socket, body);
}

} // namespace gturbo
