// Incremental detokenization and streaming stop matching.
//
// The invariant that matters most is at the bottom: streaming a sequence token by token must
// produce exactly the same bytes as decoding it in one batch. If it does not, streamed output
// differs from non-streamed output for the same generation -- which is the kind of bug that
// only shows up on non-ASCII text, and then only sometimes.

#include "gturbo/detokenizer.hpp"
#include "gturbo/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace gturbo;

namespace {

// A miniature tokenizer.json: a handful of ordinary pieces plus the byte-fallback range the
// incremental decoder exists to handle.
std::string write_mini_vocab() {
    std::string json = R"({"added_tokens":[{"id":0,"content":"<pad>","special":true},)"
                       R"({"id":1,"content":"<eos>","special":true}],)"
                       R"("model":{"vocab":{"<pad>":0,"<eos>":1,)"
                       R"("Hi":2,"▁there":3,"!":4)";
    // Byte fallback <0x00>..<0xFF> at ids 100..355.
    for (int b = 0; b < 256; ++b) {
        char piece[16];
        std::snprintf(piece, sizeof(piece), "<0x%02X>", b);
        json += ",\"";
        json += piece;
        json += "\":" + std::to_string(100 + b);
    }
    json += R"(},"merges":[]}})";

    fs::path p = fs::temp_directory_path() / "gturbo_test_detok_vocab.json";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << json;
    out.close();
    return p.string();
}

// Token ids for the raw bytes of a UTF-8 string, in byte-fallback form.
std::vector<uint32_t> bytes_of(const std::string& s) {
    std::vector<uint32_t> ids;
    for (unsigned char c : s) ids.push_back(100u + c);
    return ids;
}

std::string stream_all(const Tokenizer& tok, const std::vector<uint32_t>& ids) {
    IncrementalDetokenizer d(tok);
    std::string out;
    for (uint32_t id : ids) out += d.push(id);
    out += d.finish();
    return out;
}

} // namespace

int main() {
    std::cout << "[TEST] Running incremental detokenizer / stop matcher tests...\n";

    const std::string vocab_path = write_mini_vocab();
    Tokenizer tok;
    if (!tok.load_vocabulary(vocab_path)) {
        std::cout << "  [FAIL] could not load the synthetic vocabulary\n";
        return 1;
    }

    // ---- A multi-byte character split across byte-fallback tokens -------------------
    {
        // U+20AC EURO SIGN is E2 82 AC: three tokens that mean nothing individually.
        IncrementalDetokenizer d(tok);
        assert(d.push(100u + 0xE2).empty() && "emitted a partial UTF-8 sequence");
        assert(d.push(100u + 0x82).empty() && "emitted a partial UTF-8 sequence");
        const std::string third = d.push(100u + 0xAC);
        assert(third == "\xE2\x82\xAC");
        assert(d.finish().empty());
        std::cout << "  [PASS] Multi-byte character emitted only once complete.\n";
    }

    // ---- An incomplete trailing sequence becomes U+FFFD, and does not throw ----------
    {
        IncrementalDetokenizer d(tok);
        assert(d.push(100u + 0xE2).empty());
        const std::string tail = d.finish();
        assert(tail == "\xEF\xBF\xBD" && "dangling sequence should become U+FFFD");
        std::cout << "  [PASS] Dangling byte run flushes as U+FFFD.\n";
    }

    // ---- Byte run followed by an ordinary piece -------------------------------------
    {
        IncrementalDetokenizer d(tok);
        std::string out;
        out += d.push(100u + 0xE2);
        out += d.push(100u + 0x82);
        out += d.push(100u + 0xAC);
        out += d.push(2);            // "Hi"
        out += d.finish();
        assert(out == "\xE2\x82\xAC" "Hi");
        std::cout << "  [PASS] Byte run then ordinary piece concatenates correctly.\n";
    }

    // ---- Special tokens are skipped --------------------------------------------------
    {
        IncrementalDetokenizer d(tok);
        std::string out;
        out += d.push(2);            // "Hi"
        out += d.push(1);            // <eos>, special
        out += d.push(4);            // "!"
        out += d.finish();
        assert(out == "Hi!");
        std::cout << "  [PASS] Special tokens skipped while streaming.\n";
    }

    // ---- THE invariant: streaming == batch decode ------------------------------------
    {
        const std::vector<std::string> samples{
            "Hello, world!",
            "caf\xC3\xA9",                                  // e-acute
            "\xE2\x82\xAC 100",                             // euro sign
            "\xF0\x9F\x8E\x89 party",                       // 4-byte emoji
            "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",         // CJK
            "mixed \xC3\xA9 and \xF0\x9F\x8E\x89 together",
        };

        std::mt19937 rng(20260813);
        int checked = 0;
        for (const auto& s : samples) {
            auto ids = bytes_of(s);
            assert(stream_all(tok, ids) == tok.decode(ids, true));
            ++checked;
        }
        // Randomized sequences too: interleave byte-fallback runs with ordinary pieces so
        // the flush boundaries land in awkward places.
        std::uniform_int_distribution<int> pick(0, 3);
        for (int trial = 0; trial < 50; ++trial) {
            std::vector<uint32_t> ids;
            for (int n = 0; n < 40; ++n) {
                switch (pick(rng)) {
                    case 0: ids.push_back(2); break;
                    case 1: ids.push_back(4); break;
                    case 2: for (uint32_t b : bytes_of("\xE2\x82\xAC")) ids.push_back(b); break;
                    default: for (uint32_t b : bytes_of("\xC3\xA9")) ids.push_back(b); break;
                }
            }
            assert(stream_all(tok, ids) == tok.decode(ids, true) &&
                   "streamed text differs from batch decode");
            ++checked;
        }
        std::cout << "  [PASS] Streamed output equals batch decode over " << checked
                  << " sequences.\n";
    }

    // ---- Stop matcher: withhold a partial suffix, then release it -------------------
    {
        StreamingStopMatcher m({"</s>"});
        // "a</" ends with a prefix of the stop string, so "</" must be withheld.
        assert(m.push("a</") == "a");
        assert(!m.stopped());
        // It turns out not to be the stop string, so it is released.
        assert(m.push("x") == "</x");
        assert(!m.stopped());
        std::cout << "  [PASS] Partial suffix withheld, then released when it does not match.\n";
    }

    // ---- Stop matcher: a stop string spanning two pushes -----------------------------
    {
        StreamingStopMatcher m({"</s>"});
        assert(m.push("hello</") == "hello");
        assert(m.push("s>world") == "");
        assert(m.stopped());
        assert(m.matched() == "</s>");
        assert(m.push("more").empty() && "emitted text after stopping");
        assert(m.finish().empty());
        std::cout << "  [PASS] Stop string spanning a token boundary is caught, not leaked.\n";
    }

    // ---- Stop matcher: earliest match wins -------------------------------------------
    {
        StreamingStopMatcher m({"END", "X"});
        // "X" appears before "END", so it must win even though "END" is listed first.
        assert(m.push("abXcENDd") == "ab");
        assert(m.stopped());
        assert(m.matched() == "X");
        std::cout << "  [PASS] Earliest match wins regardless of list order.\n";
    }

    // ---- Stop matcher: no stops is pass-through, and finish() releases ---------------
    {
        StreamingStopMatcher none;
        assert(none.push("anything at all") == "anything at all");
        assert(!none.stopped());

        StreamingStopMatcher m({"</s>"});
        assert(m.push("tail</") == "tail");
        assert(m.finish() == "</" && "withheld text must be released when generation ends");
        std::cout << "  [PASS] Empty stop list passes through; finish() releases the hold.\n";
    }

    std::error_code ec;
    fs::remove(vocab_path, ec);

    std::cout << "[TEST] All detokenizer / stop matcher tests passed.\n";
    return 0;
}
