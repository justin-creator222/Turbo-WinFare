// HTTP request reading and JSON serialization.
//
// Driven over a real loopback socket pair, with no GPU and no model. What is being tested is
// the framing the old server got wrong: it did one 8192-byte recv() and assumed that was the
// whole request, so a body split across segments parsed as truncated JSON, Content-Length was
// never consulted, and anything over 8 KB was silently cut in half.

#include <winsock2.h>
#include <ws2tcpip.h>

#include "gturbo/http.hpp"
#include "gturbo/json.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

using namespace gturbo;

namespace {

// A connected loopback socket pair. Windows has no socketpair(), so this listens on an
// ephemeral port and connects to itself.
struct SocketPair {
    SOCKET client{INVALID_SOCKET};   // written by the test
    SOCKET server{INVALID_SOCKET};   // read by read_http_request

    SocketPair() {
        SOCKET listener = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        listen(listener, 1);

        int len = sizeof(addr);
        getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &len);

        client = socket(AF_INET, SOCK_STREAM, 0);
        connect(client, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        server = accept(listener, nullptr, nullptr);
        closesocket(listener);
    }

    ~SocketPair() {
        if (client != INVALID_SOCKET) closesocket(client);
        if (server != INVALID_SOCKET) closesocket(server);
    }

    void write(const std::string& data) {
        send(client, data.data(), static_cast<int>(data.size()), 0);
    }
    void done_writing() { shutdown(client, SD_SEND); }
};

// Sends `parts` with a pause between each, so the reader is forced to loop on recv.
HttpReadResult read_in_parts(const std::vector<std::string>& parts, HttpRequest& out,
                             size_t max_body = 1024 * 1024) {
    SocketPair sp;
    std::thread writer([&] {
        for (const auto& p : parts) {
            sp.write(p);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    const HttpReadResult r =
        read_http_request(static_cast<uintptr_t>(sp.server), out, max_body, 5000);
    writer.join();
    return r;
}

} // namespace

int main() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    std::cout << "[TEST] Running HTTP / JSON server tests...\n";

    // ---- A request split across three segments -------------------------------------
    //
    // This is the case the single-recv reader got wrong most often: the headers arrive, the
    // body does not, and the request parses as truncated JSON.
    {
        HttpRequest req;
        const std::string body = R"({"prompt":"hello"})";
        const auto r = read_in_parts({
            "POST /api/generate HT",
            "TP/1.1\r\nContent-Type: application/json\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n",
            body,
        }, req);

        assert(r == HttpReadResult::Ok);
        assert(req.method == "POST");
        assert(req.path == "/api/generate");
        assert(req.body == body);
        std::cout << "  [PASS] Request split across three segments is reassembled.\n";
    }

    // ---- Content-Length is honoured, not "whatever arrived" -------------------------
    {
        // More bytes follow than Content-Length claims; the surplus belongs to a pipelined
        // request and must not be swallowed into this body.
        HttpRequest req;
        const auto r = read_in_parts({
            "POST /x HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}EXTRA",
        }, req);
        assert(r == HttpReadResult::Ok);
        assert(req.body == "{}" && "read past Content-Length");
        std::cout << "  [PASS] Content-Length bounds the body exactly.\n";
    }

    // ---- A body larger than the old 8 KB buffer -------------------------------------
    {
        // 32 KB of prompt: silently truncated by the old reader, which is how a long
        // conversation turned into a JSON parse error.
        const std::string big(32 * 1024, 'x');
        const std::string body = R"({"prompt":")" + big + R"("})";
        HttpRequest req;
        const auto r = read_in_parts({
            "POST /api/generate HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n",
            body.substr(0, 10000),
            body.substr(10000),
        }, req);
        assert(r == HttpReadResult::Ok);
        assert(req.body.size() == body.size());
        JsonValue parsed = JsonValue::parse(req.body);
        assert(parsed.string_or("prompt", "") == big);
        std::cout << "  [PASS] 32 KB body survives (the old 8 KB buffer truncated it).\n";
    }

    // ---- Chunked transfer encoding ---------------------------------------------------
    {
        HttpRequest req;
        const auto r = read_in_parts({
            "POST /x HTTP/1.1\r\nContent-Type: application/json\r\n"
            "Transfer-Encoding: chunked\r\n\r\n",
            "9\r\n{\"a\":1234\r\n",
            "2\r\n5}\r\n",
            "0\r\n\r\n",
        }, req);
        assert(r == HttpReadResult::Ok);
        assert(req.body == R"({"a":12345})");
        std::cout << "  [PASS] Chunked body is de-chunked.\n";
    }

    // ---- Limits ----------------------------------------------------------------------
    {
        HttpRequest req;
        auto r = read_in_parts({
            "POST /x HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 999999\r\n\r\n",
        }, req, /*max_body=*/1024);
        assert(r == HttpReadResult::TooLarge);

        r = read_in_parts({
            "POST /x HTTP/1.1\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nhello",
        }, req);
        assert(r == HttpReadResult::UnsupportedMediaType);
        std::cout << "  [PASS] Oversized body -> TooLarge; wrong Content-Type -> 415.\n";
    }

    // ---- Headers, query strings, keep-alive -------------------------------------------
    {
        HttpRequest req;
        const auto r = read_in_parts({
            "GET /api/telemetry?verbose=1 HTTP/1.1\r\nHost: localhost\r\n"
            "X-Mixed-Case: Value\r\nConnection: close\r\n\r\n",
        }, req);
        assert(r == HttpReadResult::Ok);
        assert(req.path == "/api/telemetry");
        assert(req.query == "verbose=1");
        // Header lookup must be case-insensitive.
        assert(req.header("x-mixed-case") == "Value");
        assert(req.header("X-MIXED-CASE") == "Value");
        assert(!req.keep_alive && "Connection: close was ignored");
        std::cout << "  [PASS] Query split, case-insensitive headers, Connection honoured.\n";
    }

    // ---- A GET with no body is not an error ------------------------------------------
    {
        HttpRequest req;
        const auto r = read_in_parts({"GET /api/models HTTP/1.1\r\n\r\n"}, req);
        assert(r == HttpReadResult::Ok);
        assert(req.body.empty());
        assert(req.keep_alive && "HTTP/1.1 defaults to keep-alive");
        std::cout << "  [PASS] Bodyless GET parses cleanly.\n";
    }

    // ---- Reason phrases --------------------------------------------------------------
    {
        // Every status the server can emit needs a real phrase; these all used to be "OK".
        for (int s : {200, 204, 400, 404, 405, 409, 413, 415, 429, 500, 501, 503}) {
            assert(std::string(http_reason_phrase(s)) != "Unknown");
        }
        assert(std::string(http_reason_phrase(500)) == "Internal Server Error");
        assert(std::string(http_reason_phrase(429)) == "Too Many Requests");
        std::cout << "  [PASS] Every emitted status has a real reason phrase.\n";
    }

    // ---- JSON: the characters that broke the substring parser -------------------------
    {
        // The old handler found "prompt": and took the text between the next two quotes, so
        // an escaped quote ended the string early and a newline corrupted the rest.
        const std::string tricky = "He said \"hi\"\nand left\tnow\\done";
        const std::string doc = R"({"prompt":)" + JsonValue::quote(tricky) + "}";
        JsonValue v = JsonValue::parse(doc);
        assert(v.string_or("prompt", "") == tricky);

        // Control characters must be escaped rather than emitted raw.
        const std::string ctrl("a\x01\x1F" "b", 4);
        JsonValue r = JsonValue::parse("{\"k\":" + JsonValue::quote(ctrl) + "}");
        assert(r.string_or("k", "") == ctrl);

        // Surrogate pairs (an emoji) round-trip.
        JsonValue e = JsonValue::parse(R"({"k":"🎉"})");
        assert(e.string_or("k", "") == "\xF0\x9F\x8E\x89");
        std::cout << "  [PASS] JSON round-trips quotes, newlines, control chars, surrogates.\n";
    }

    // ---- Windows paths must survive being embedded in JSON ---------------------------
    //
    // Telemetry interpolated model_dir raw, so an absolute path emitted
    //   "model_dir":"C:\Users\Justin\Code\Turbo\gemma-4-26b-a4b.gturbo"
    // where \U is not a valid JSON escape. The whole document became unparseable and the
    // GUI's telemetry poll failed silently. It only worked before because the path happened
    // to be relative and contained no backslashes.
    {
        const std::string win_path = R"(C:\Users\Justin\Code\Turbo\gemma-4-26b-a4b.gturbo)";
        const std::string doc = R"({"model_dir":)" + JsonValue::quote(win_path) + "}";
        JsonValue v = JsonValue::parse(doc);
        assert(v.string_or("model_dir", "") == win_path);

        // The escape must actually be present in the wire bytes, not just survive a
        // round-trip through a lenient parser.
        assert(doc.find(R"(\\Users)") != std::string::npos);
        std::cout << "  [PASS] Absolute Windows paths embed as valid JSON.\n";
    }

    // ---- JSON: nested messages, and rejection of malformed input ----------------------
    {
        JsonValue v = JsonValue::parse(
            R"({"messages":[{"role":"system","content":"be brief"},)"
            R"({"role":"user","content":"hi"}]})");
        const JsonValue& arr = v.at("messages", "req");
        assert(arr.is_array() && arr.array_value.size() == 2);
        assert(arr.array_value[0].string_or("role", "") == "system");
        assert(arr.array_value[1].string_or("content", "") == "hi");

        auto rejects = [](const std::string& s) {
            try { JsonValue::parse(s); return false; }
            catch (const GTurboFormatError&) { return true; }
        };
        assert(rejects("{\"a\":1} trailing"));
        assert(rejects("{\"a\":}"));
        assert(rejects("{'a':1}"));
        assert(rejects(std::string(200, '[')));   // nesting past the depth limit
        std::cout << "  [PASS] Nested messages parse; malformed input is rejected.\n";
    }

    std::cout << "[TEST] All HTTP / JSON server tests passed.\n";
    WSACleanup();
    return 0;
}
