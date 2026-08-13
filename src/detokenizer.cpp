#include "gturbo/detokenizer.hpp"

#include <algorithm>

namespace gturbo {

namespace {

// Length of the UTF-8 sequence starting with `lead`, or 0 if it is not a valid lead byte.
size_t utf8_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0;   // continuation byte or invalid lead
}

constexpr const char* kReplacement = "\xEF\xBF\xBD";   // U+FFFD

} // namespace

std::string IncrementalDetokenizer::drain(bool flush_all) {
    std::string out;
    size_t i = 0;

    while (i < pending_.size()) {
        const unsigned char lead = static_cast<unsigned char>(pending_[i]);
        const size_t need = utf8_len(lead);

        if (need == 0) {
            // Not a valid lead byte. Emit a replacement and resynchronize by one byte, so a
            // single corrupt byte cannot poison the rest of the stream.
            out += kReplacement;
            ++i;
            continue;
        }
        if (i + need > pending_.size()) {
            // Incomplete tail. Hold it unless this is the final flush, where there will be
            // no more bytes to complete it.
            if (!flush_all) break;
            out += kReplacement;
            i = pending_.size();
            break;
        }
        out.append(pending_, i, need);
        i += need;
    }

    pending_.erase(0, i);
    return out;
}

std::string IncrementalDetokenizer::push(uint32_t token) {
    uint8_t byte = 0;
    if (tok_.is_byte_fallback(token, byte)) {
        // Byte-fallback pieces contribute raw bytes; drain() decides when enough have
        // arrived to form a character.
        pending_ += static_cast<char>(byte);
        return drain(false);
    }

    if (skip_special_ && tok_.is_special_token(token)) {
        // A special token cannot be part of a byte run, so anything pending is terminal.
        return drain(true);
    }

    // Any ordinary piece ends whatever byte run was in progress. Flush it first so an
    // incomplete sequence does not absorb the leading bytes of the piece's own text.
    std::string out = drain(true);
    pending_ += tok_.decode_single(token, skip_special_);
    out += drain(false);
    return out;
}

std::string IncrementalDetokenizer::finish() {
    return drain(true);
}

// ---------------------------------------------------------------------------

StreamingStopMatcher::StreamingStopMatcher(std::vector<std::string> stops) {
    for (auto& s : stops) {
        if (s.empty()) continue;               // an empty stop would match instantly
        max_stop_len_ = std::max(max_stop_len_, s.size());
        stops_.push_back(std::move(s));
    }
}

std::string StreamingStopMatcher::push(std::string_view text) {
    if (stopped_) return {};
    if (stops_.empty()) return std::string(text);

    buffer_.append(text);

    // Earliest match wins, so a short stop string later in the list cannot be shadowed by a
    // longer one that happens to appear further along.
    size_t best = std::string::npos;
    const std::string* which = nullptr;
    for (const auto& s : stops_) {
        const size_t at = buffer_.find(s);
        if (at != std::string::npos && at < best) { best = at; which = &s; }
    }
    if (which) {
        stopped_ = true;
        matched_ = *which;
        std::string out = buffer_.substr(0, best);
        buffer_.clear();
        return out;
    }

    // No match yet. Withhold any suffix that is still a prefix of some stop string -- it may
    // become one once the next token arrives. Only the last (max_stop_len - 1) bytes can be.
    const size_t max_hold = (max_stop_len_ > 0) ? max_stop_len_ - 1 : 0;
    size_t hold = 0;
    const size_t limit = std::min(max_hold, buffer_.size());
    for (size_t n = limit; n >= 1; --n) {
        std::string_view tail(buffer_.data() + buffer_.size() - n, n);
        bool is_prefix = false;
        for (const auto& s : stops_) {
            if (s.size() > n && s.compare(0, n, tail) == 0) { is_prefix = true; break; }
        }
        if (is_prefix) { hold = n; break; }
    }

    std::string out = buffer_.substr(0, buffer_.size() - hold);
    buffer_.erase(0, buffer_.size() - hold);
    return out;
}

std::string StreamingStopMatcher::finish() {
    if (stopped_) return {};
    std::string out = std::move(buffer_);
    buffer_.clear();
    return out;
}

} // namespace gturbo
