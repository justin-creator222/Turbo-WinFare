#pragma once

// Minimal but correct HTTP/1.1 request reading and response writing.
//
// The server previously did a single 8192-byte recv() and treated whatever arrived as the
// entire request. That works for small hand-made requests and fails for everything else: a
// body split across TCP segments parsed as truncated JSON, a body over 8 KB was silently cut,
// Content-Length was never consulted, chunked bodies were not understood, and keep-alive
// connections were impossible. An OpenAI-shaped {"messages":[...]} request with any real
// conversation in it exceeds that buffer routinely.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gturbo {

// True when a request target is safe to act on.
//
// The static file handler builds a filesystem path directly from the request target, and
// nothing used to validate it: `GET /../../../../../Windows/win.ini` returned that file with
// a 200, on a server that also answered from every interface with no authentication. This
// runs before dispatch, so it guards the /api and /v1 routes as well.
//
// Percent-decoded first: `%2e%2e%2f` reaches the same place without a literal "..".
bool request_path_is_safe(const std::string& raw);

// Percent-decodes a request target. Malformed escapes are left as literal characters -- this
// feeds a validator, so anything undecodable must not be waved through.
std::string percent_decode(const std::string& in);


struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
    bool keep_alive{false};

    // Case-insensitive lookup; HTTP header names are not case-sensitive.
    std::string header(std::string_view name) const;
};

enum class HttpReadResult {
    Ok,
    Closed,                 // peer hung up cleanly, nothing pending
    Malformed,              // 400
    TooLarge,               // 413
    UnsupportedMediaType,   // 415
    Timeout,
};

// Reads exactly one request. Loops until the headers are complete, then reads either
// Content-Length bytes or a chunked body. `max_body_bytes` guards against a client claiming
// an enormous Content-Length.
HttpReadResult read_http_request(uintptr_t socket, HttpRequest& out,
                                 size_t max_body_bytes, int timeout_ms);

// Sends a complete response. Returns false if the socket died mid-write.
bool send_http_response(uintptr_t socket, int status, std::string_view content_type,
                        std::string_view body, bool keep_alive,
                        const std::vector<std::pair<std::string, std::string>>& extra = {});

// Writes only the status line and headers, leaving the body to be streamed. Used for SSE.
bool send_http_headers(uintptr_t socket, int status, std::string_view content_type,
                       const std::vector<std::pair<std::string, std::string>>& extra = {});

// Raw write for streamed bodies. Returns false once the peer has gone away, which is the
// signal to abandon the generation rather than keep the GPU busy for nobody.
bool send_raw(uintptr_t socket, std::string_view data);

const char* http_reason_phrase(int status);

} // namespace gturbo
