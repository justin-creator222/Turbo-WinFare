#include "gturbo/tokenizer.hpp"
#include "gturbo/format.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gturbo {

// U+2581 LOWER ONE EIGHTH BLOCK -- the tokenizer's normalizer replaces every space with it,
// and the decoder replaces it back. See tokenizer.json: normalizer {Replace " " -> "▁"},
// decoder {Replace "▁" -> " ", ByteFallback, Fuse}.
static const char* SPM_SPACE = "\xe2\x96\x81";
static const size_t SPM_SPACE_LEN = 3;

// ---------------------------------------------------------------------------
// A focused scanner for tokenizer.json.
//
// The file is ~32 MB with ~262k vocab entries and ~250k merges. Building a generic JSON
// tree over that costs hundreds of megabytes, so this walks the text and pulls out only
// the three sections that matter: added_tokens, model.vocab, and model.merges.
// ---------------------------------------------------------------------------
namespace {

class JsonScanner {
public:
    explicit JsonScanner(const std::string& text) : s_(text) {}

    size_t pos{0};

    bool eof() const { return pos >= s_.size(); }
    char peek() const { return pos < s_.size() ? s_[pos] : '\0'; }

    void skip_ws() {
        while (pos < s_.size()) {
            char c = s_[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
            else break;
        }
    }

    void expect(char c, const char* what) {
        skip_ws();
        if (pos >= s_.size() || s_[pos] != c) {
            throw GTurboFormatError(std::string("tokenizer.json: expected '") + c +
                                    "' in " + what + " at byte " + std::to_string(pos));
        }
        ++pos;
    }

    // Parses a JSON string literal, resolving escapes (including surrogate pairs) to UTF-8.
    std::string parse_string() {
        skip_ws();
        if (pos >= s_.size() || s_[pos] != '"') {
            throw GTurboFormatError("tokenizer.json: expected a string at byte " +
                                    std::to_string(pos));
        }
        ++pos;
        std::string out;
        while (pos < s_.size()) {
            char c = s_[pos++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (pos >= s_.size()) break;
            char e = s_[pos++];
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos + 1 < s_.size() &&
                        s_[pos] == '\\' && s_[pos + 1] == 'u') {
                        size_t save = pos;
                        pos += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            pos = save;
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    throw GTurboFormatError("tokenizer.json: bad escape at byte " +
                                            std::to_string(pos));
            }
        }
        throw GTurboFormatError("tokenizer.json: unterminated string");
    }

    int64_t parse_int() {
        skip_ws();
        size_t start = pos;
        if (pos < s_.size() && (s_[pos] == '-' || s_[pos] == '+')) ++pos;
        while (pos < s_.size() && s_[pos] >= '0' && s_[pos] <= '9') ++pos;
        if (start == pos) {
            throw GTurboFormatError("tokenizer.json: expected an integer at byte " +
                                    std::to_string(pos));
        }
        return std::stoll(s_.substr(start, pos - start));
    }

    // Skips one complete JSON value, whatever its type.
    void skip_value() {
        skip_ws();
        char c = peek();
        if (c == '"') { parse_string(); return; }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0;
            while (pos < s_.size()) {
                char d = s_[pos];
                if (d == '"') { parse_string(); continue; }
                if (d == open) ++depth;
                else if (d == close && --depth == 0) { ++pos; return; }
                ++pos;
            }
            throw GTurboFormatError("tokenizer.json: unbalanced brackets");
        }
        while (pos < s_.size() && s_[pos] != ',' && s_[pos] != '}' && s_[pos] != ']') ++pos;
    }

    // Advances to the value of `key` at the current object level. Returns false if the
    // object ends first.
    bool seek_key(const std::string& key) {
        skip_ws();
        if (peek() == '{') ++pos;
        for (;;) {
            skip_ws();
            char c = peek();
            if (c == ',') { ++pos; continue; }
            if (c == '}' || c == '\0') return false;
            std::string k = parse_string();
            expect(':', "object");
            if (k == key) return true;
            skip_value();
        }
    }

private:
    const std::string& s_;

    uint32_t hex4() {
        if (pos + 4 > s_.size()) {
            throw GTurboFormatError("tokenizer.json: truncated \\u escape");
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos++];
            v <<= 4;
            if (c >= '0' && c <= '9')      v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else throw GTurboFormatError("tokenizer.json: bad hex digit");
        }
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
};

// Length in bytes of the UTF-8 character starting at s[i].
size_t utf8_char_len(const std::string& s, size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((c & 0xF8) == 0xF0)      len = 4;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xE0) == 0xC0) len = 2;
    return std::min(len, s.size() - i);
}

} // namespace

// ---------------------------------------------------------------------------

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

bool Tokenizer::parse_tokenizer_json(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw GTurboFormatError("Failed to open tokenizer: " + filepath);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str();
    if (text.empty()) {
        throw GTurboFormatError("Empty tokenizer file: " + filepath);
    }

    // --- added_tokens -------------------------------------------------------
    {
        JsonScanner sc(text);
        if (sc.seek_key("added_tokens")) {
            sc.expect('[', "added_tokens");
            for (;;) {
                sc.skip_ws();
                if (sc.peek() == ']') { ++sc.pos; break; }
                if (sc.peek() == ',') { ++sc.pos; continue; }

                // Each entry: {"id": N, "content": "...", ..., "special": bool}
                size_t obj_start = sc.pos;
                uint32_t id = 0;
                std::string content;
                bool special = false;

                sc.expect('{', "added_tokens entry");
                for (;;) {
                    sc.skip_ws();
                    if (sc.peek() == '}') { ++sc.pos; break; }
                    if (sc.peek() == ',') { ++sc.pos; continue; }
                    std::string k = sc.parse_string();
                    sc.expect(':', "added_tokens entry");
                    if (k == "id") {
                        id = static_cast<uint32_t>(sc.parse_int());
                    } else if (k == "content") {
                        content = sc.parse_string();
                    } else if (k == "special") {
                        sc.skip_ws();
                        special = (sc.peek() == 't');
                        sc.skip_value();
                    } else {
                        sc.skip_value();
                    }
                }
                (void)obj_start;
                if (!content.empty()) {
                    added_tokens_.emplace_back(content, id);
                    if (id >= is_special_.size()) is_special_.resize(id + 1, false);
                    is_special_[id] = special;
                }
            }
        }
    }

    // --- model.vocab and model.merges --------------------------------------
    JsonScanner sc(text);
    if (!sc.seek_key("model")) {
        throw GTurboFormatError("tokenizer.json: no \"model\" section");
    }
    size_t model_start = sc.pos;

    {
        JsonScanner vs(text);
        vs.pos = model_start;
        if (!vs.seek_key("vocab")) {
            throw GTurboFormatError("tokenizer.json: no model.vocab");
        }
        vs.expect('{', "model.vocab");
        for (;;) {
            vs.skip_ws();
            if (vs.peek() == '}') { ++vs.pos; break; }
            if (vs.peek() == ',') { ++vs.pos; continue; }
            std::string piece = vs.parse_string();
            vs.expect(':', "model.vocab");
            uint32_t id = static_cast<uint32_t>(vs.parse_int());

            if (id >= pieces_.size()) pieces_.resize(id + 1);
            pieces_[id] = piece;
            piece_map_[piece] = id;

            // Byte-fallback tokens look like "<0x41>".
            if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
                int hi = piece[3], lo = piece[4];
                auto val = [](int c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                int h = val(hi), l = val(lo);
                if (h >= 0 && l >= 0) {
                    byte_to_id_[static_cast<uint8_t>(h * 16 + l)] = id;
                }
            }
        }
    }

    {
        JsonScanner ms(text);
        ms.pos = model_start;
        if (ms.seek_key("merges")) {
            ms.expect('[', "model.merges");
            uint32_t rank = 0;
            for (;;) {
                ms.skip_ws();
                if (ms.peek() == ']') { ++ms.pos; break; }
                if (ms.peek() == ',') { ++ms.pos; continue; }

                std::string left, right;
                if (ms.peek() == '[') {
                    // Newer format: ["left", "right"]
                    ++ms.pos;
                    left = ms.parse_string();
                    ms.expect(',', "model.merges pair");
                    right = ms.parse_string();
                    ms.expect(']', "model.merges pair");
                } else {
                    // Classic format: "left right"
                    std::string joined = ms.parse_string();
                    size_t sp = joined.find(' ');
                    if (sp == std::string::npos) continue;
                    left = joined.substr(0, sp);
                    right = joined.substr(sp + 1);
                }
                merge_ranks_.emplace(left + '\0' + right, rank++);
            }
        }
    }

    if (pieces_.empty() || piece_map_.empty()) {
        throw GTurboFormatError("tokenizer.json: vocabulary is empty");
    }

    vocab_size_ = static_cast<uint32_t>(pieces_.size());
    is_special_.resize(pieces_.size(), false);

    // Resolve the special IDs from the vocabulary rather than trusting the defaults.
    auto lookup = [&](const char* name, uint32_t& slot) {
        auto it = piece_map_.find(name);
        if (it != piece_map_.end()) slot = it->second;
    };
    lookup("<bos>", bos_id_);
    lookup("<eos>", eos_id_);
    lookup("<pad>", pad_id_);
    lookup("<unk>", unk_id_);
    lookup("<|turn>", turn_start_id_);
    lookup("<turn|>", end_of_turn_id_);
    lookup("<|tool_response>", tool_response_id_);
    stop_token_ids_ = {eos_id_, end_of_turn_id_, tool_response_id_};

    // Longest content first, so "<|tool_response>" is matched before any shorter token
    // that happens to be a prefix of it.
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const auto& a, const auto& b) {
                  if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
                  return a.first < b.first;
              });

    loaded_ = true;
    return true;
}

bool Tokenizer::load_vocabulary(const std::string& vocab_file) {
    // There is no synthesized-vocabulary fallback. A tokenizer that silently substitutes a
    // few hundred ASCII pieces for a 262,144-entry vocabulary turns every decode into
    // plausible-looking garbage instead of a visible failure.
    if (vocab_file.empty()) {
        throw GTurboFormatError("Tokenizer::load_vocabulary called with no vocabulary path");
    }
    return parse_tokenizer_json(vocab_file);
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

void Tokenizer::emit_byte_fallback(const std::string& piece,
                                   std::vector<uint32_t>& out) const {
    for (unsigned char b : piece) {
        auto it = byte_to_id_.find(b);
        out.push_back(it != byte_to_id_.end() ? it->second : unk_id_);
    }
}

void Tokenizer::bpe_segment(const std::string& text, std::vector<uint32_t>& out) const {
    if (text.empty()) return;

    // Start from single UTF-8 characters, then repeatedly merge the adjacent pair with the
    // lowest merge rank until no adjacent pair appears in the merge table.
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        size_t len = utf8_char_len(text, i);
        symbols.emplace_back(text, i, len);
        i += len;
    }

    for (;;) {
        uint32_t best_rank = UINT32_MAX;
        size_t best_i = SIZE_MAX;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_ranks_.find(symbols[i] + '\0' + symbols[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i == SIZE_MAX) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_i) + 1);
    }

    for (const auto& sym : symbols) {
        auto it = piece_map_.find(sym);
        if (it != piece_map_.end()) {
            out.push_back(it->second);
        } else {
            // byte_fallback is true for this model: unknown pieces become <0xNN> per byte.
            emit_byte_fallback(sym, out);
        }
    }
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    if (!loaded_) {
        throw GTurboFormatError("Tokenizer::encode called before a vocabulary was loaded");
    }

    std::vector<uint32_t> tokens;
    if (add_bos) tokens.push_back(bos_id_);
    if (text.empty()) return tokens;

    // Walk the input, peeling off added/special tokens verbatim and BPE-encoding whatever
    // lies between them. Added tokens are matched even when add_bos is false -- that flag
    // controls only the leading <bos>, not special-token recognition.
    std::string pending;
    auto flush_pending = [&]() {
        if (pending.empty()) return;
        // Normalizer: every space becomes U+2581 before BPE sees it.
        std::string normalized;
        normalized.reserve(pending.size());
        for (char c : pending) {
            if (c == ' ') normalized += SPM_SPACE;
            else normalized += c;
        }
        bpe_segment(normalized, tokens);
        pending.clear();
    };

    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<' || text[i] == '[') {
            for (const auto& [content, id] : added_tokens_) {
                if (content.size() <= text.size() - i &&
                    std::memcmp(text.data() + i, content.data(), content.size()) == 0) {
                    flush_pending();
                    tokens.push_back(id);
                    i += content.size();
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            pending += text[i];
            ++i;
        }
    }
    flush_pending();
    return tokens;
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

std::string Tokenizer::decode_single(uint32_t token, bool skip_special) const {
    if (token >= pieces_.size()) return "";
    if (skip_special && token < is_special_.size() && is_special_[token]) return "";

    const std::string& piece = pieces_[token];
    std::string out;
    out.reserve(piece.size());
    for (size_t i = 0; i < piece.size();) {
        if (piece.compare(i, SPM_SPACE_LEN, SPM_SPACE) == 0) {
            out += ' ';
            i += SPM_SPACE_LEN;
        } else {
            out += piece[i];
            ++i;
        }
    }
    return out;
}

// Factored out of decode() so IncrementalDetokenizer does not have to re-parse piece text.
bool Tokenizer::is_byte_fallback(uint32_t token, uint8_t& value) const {
    if (token >= pieces_.size()) return false;
    const std::string& piece = pieces_[token];
    if (piece.size() != 6 || piece.compare(0, 3, "<0x") != 0 || piece[5] != '>') return false;

    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    const int h = val(piece[3]), l = val(piece[4]);
    if (h < 0 || l < 0) return false;
    value = static_cast<uint8_t>(h * 16 + l);
    return true;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens, bool skip_special) const {
    if (!loaded_) {
        throw GTurboFormatError("Tokenizer::decode called before a vocabulary was loaded");
    }

    std::string out;
    // Byte-fallback tokens must be fused before being interpreted, so that a multi-byte
    // UTF-8 character split across several <0xNN> tokens reassembles correctly.
    std::string byte_run;
    auto flush_bytes = [&]() {
        out += byte_run;
        byte_run.clear();
    };

    for (uint32_t t : tokens) {
        if (t >= pieces_.size()) continue;

        uint8_t byte = 0;
        if (is_byte_fallback(t, byte)) {
            byte_run += static_cast<char>(byte);
            continue;
        }
        flush_bytes();
        out += decode_single(t, skip_special);
    }
    flush_bytes();
    return out;
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages) const {
    static const std::string TURN_OPEN = "<|turn>";
    static const std::string TURN_CLOSE = "<turn|>";

    std::string out = "<bos>";
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        if (m.role == "system" && i != 0) {
            throw GTurboFormatError("chat template: a system message must come first");
        }
        // Trim surrounding whitespace, matching the reference template.
        size_t b = m.content.find_first_not_of(" \t\r\n");
        size_t e = m.content.find_last_not_of(" \t\r\n");
        std::string content = (b == std::string::npos) ? "" : m.content.substr(b, e - b + 1);

        const std::string role = (m.role == "assistant") ? "model" : m.role;
        out += TURN_OPEN + role + "\n" + content + TURN_CLOSE + "\n";
    }
    // Generation prompt, with thinking disabled.
    out += TURN_OPEN + "model\n<|channel>thought\n<channel|>";
    return out;
}

} // namespace gturbo
