#pragma once

// Token-by-token detokenization and streaming stop-string matching.
//
// Both exist because streaming text is not just "call decode on each token". Two things go
// wrong if you do that:
//
//   1. A multi-byte UTF-8 character can be split across several <0xNN> byte-fallback tokens.
//      Decoded individually each yields the literal text "<0xE2>"; only fused do they form
//      the character. Tokenizer::decode handles this by fusing runs over the whole sequence,
//      which a stream does not have.
//
//   2. A stop string can straddle a token boundary. Emitting each token as it arrives leaks
//      the first half of the stop string to the client before the match is recognized.

#include "gturbo/tokenizer.hpp"

#include <string>
#include <vector>

namespace gturbo {

class IncrementalDetokenizer {
public:
    explicit IncrementalDetokenizer(const Tokenizer& tok, bool skip_special = true)
        : tok_(tok), skip_special_(skip_special) {}

    // Returns the text that is now definitely complete. May be empty when the token only
    // extended a pending byte run or a partial UTF-8 sequence.
    std::string push(uint32_t token);

    // Flushes anything still buffered. An incomplete trailing UTF-8 sequence becomes U+FFFD
    // rather than being dropped or emitted as invalid bytes.
    std::string finish();

    void reset() { pending_.clear(); }

private:
    // Emits whole UTF-8 sequences from pending_, keeping a 1-3 byte partial tail.
    std::string drain(bool flush_all);

    const Tokenizer& tok_;
    bool skip_special_;
    std::string pending_;
};

// Withholds any suffix that could still grow into a stop string, so a stop string spanning
// two tokens is caught and never partially emitted. Port of the reference's
// Runtime/Generation/StreamingStopMatcher.swift.
class StreamingStopMatcher {
public:
    StreamingStopMatcher() = default;
    explicit StreamingStopMatcher(std::vector<std::string> stops);

    // Returns text safe to emit now. Once a stop matches, everything from the match onward
    // is discarded and stopped() becomes true.
    std::string push(std::string_view text);

    // Releases whatever was being withheld, when generation ended for another reason.
    std::string finish();

    bool stopped() const { return stopped_; }
    const std::string& matched() const { return matched_; }
    bool empty() const { return stops_.empty(); }

private:
    std::vector<std::string> stops_;
    size_t max_stop_len_{0};
    std::string buffer_;
    std::string matched_;
    bool stopped_{false};
};

} // namespace gturbo
