#pragma once

// A small, strict, dependency-free JSON reader.
//
// The manifest and packed-experts layout were previously "parsed" by functions that ignored
// their argument entirely and returned hardcoded structs, so a bundle's real metadata was
// never read. This is the minimum needed to fix that: full JSON grammar, no allocation
// tricks, and errors that name the offending path instead of failing silently.
//
// Not a general-purpose library -- it parses into a variant tree eagerly, which is fine for
// files measured in kilobytes. Do not use it for multi-gigabyte input.

#include "gturbo/format.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace gturbo {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type{Type::Null};
    bool bool_value{false};
    double number_value{0.0};
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    bool is_null()   const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array()  const { return type == Type::Array; }

    bool has(const std::string& key) const {
        return type == Type::Object && object_value.count(key) > 0;
    }

    // Accessors. `context` appears in the error message so a bad field is identifiable.
    const JsonValue& at(const std::string& key, const std::string& context) const {
        if (type != Type::Object) {
            throw GTurboFormatError(context + "." + key + ": parent is not an object");
        }
        auto it = object_value.find(key);
        if (it == object_value.end()) {
            throw GTurboFormatError(context + ": missing required field '" + key + "'");
        }
        return it->second;
    }

    double as_number(const std::string& context) const {
        if (type != Type::Number) {
            throw GTurboFormatError(context + ": expected a number");
        }
        return number_value;
    }

    int64_t as_int(const std::string& context) const {
        double d = as_number(context);
        if (d < -9.2e18 || d > 9.2e18) {
            throw GTurboFormatError(context + ": integer out of range");
        }
        return static_cast<int64_t>(d);
    }

    const std::string& as_string(const std::string& context) const {
        if (type != Type::String) {
            throw GTurboFormatError(context + ": expected a string");
        }
        return string_value;
    }

    bool as_bool(const std::string& context) const {
        if (type != Type::Bool) {
            throw GTurboFormatError(context + ": expected a boolean");
        }
        return bool_value;
    }

    // Convenience readers with defaults, for genuinely optional fields.
    int64_t int_or(const std::string& key, int64_t fallback) const {
        if (!has(key) || object_value.at(key).type != Type::Number) return fallback;
        return static_cast<int64_t>(object_value.at(key).number_value);
    }
    double double_or(const std::string& key, double fallback) const {
        if (!has(key) || object_value.at(key).type != Type::Number) return fallback;
        return object_value.at(key).number_value;
    }
    bool bool_or(const std::string& key, bool fallback) const {
        if (!has(key) || object_value.at(key).type != Type::Bool) return fallback;
        return object_value.at(key).bool_value;
    }
    std::string string_or(const std::string& key, const std::string& fallback) const {
        if (!has(key) || object_value.at(key).type != Type::String) return fallback;
        return object_value.at(key).string_value;
    }

    // Array/object element access with the same error-naming discipline as at().
    const JsonValue& at(size_t i, const std::string& context) const {
        if (type != Type::Array) {
            throw GTurboFormatError(context + ": expected an array");
        }
        if (i >= array_value.size()) {
            throw GTurboFormatError(context + ": index out of range");
        }
        return array_value[i];
    }

    static JsonValue parse(const std::string& text);

    // Serializes a string as a JSON string literal, quotes included. Control characters go
    // out as \uXXXX so the result is valid regardless of what the model emitted.
    static std::string quote(std::string_view s);
};

namespace detail {

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text) {}

    JsonValue parse_document() {
        skip_ws();
        JsonValue v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) {
            fail("trailing content after the top-level value");
        }
        return v;
    }

private:
    static constexpr int MAX_DEPTH = 64;

    const std::string& s_;
    size_t pos_{0};

    [[noreturn]] void fail(const std::string& why) const {
        throw GTurboFormatError("json: " + why + " at byte " + std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    char peek() const {
        if (pos_ >= s_.size()) throw GTurboFormatError("json: unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (pos_ >= s_.size() || s_[pos_] != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    bool literal(const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (s_.compare(pos_, n, word) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    JsonValue parse_value(int depth) {
        if (depth > MAX_DEPTH) fail("nesting too deep");
        skip_ws();
        char c = peek();
        switch (c) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': {
                JsonValue v;
                v.type = JsonValue::Type::String;
                v.string_value = parse_string();
                return v;
            }
            case 't': {
                if (!literal("true")) fail("invalid literal");
                JsonValue v; v.type = JsonValue::Type::Bool; v.bool_value = true; return v;
            }
            case 'f': {
                if (!literal("false")) fail("invalid literal");
                JsonValue v; v.type = JsonValue::Type::Bool; v.bool_value = false; return v;
            }
            case 'n': {
                if (!literal("null")) fail("invalid literal");
                return JsonValue{};
            }
            default: return parse_number();
        }
    }

    JsonValue parse_object(int depth) {
        expect('{');
        JsonValue v;
        v.type = JsonValue::Type::Object;
        skip_ws();
        if (peek() == '}') { ++pos_; return v; }
        for (;;) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.object_value[key] = parse_value(depth + 1);
            skip_ws();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return v; }
            fail("expected ',' or '}' in object");
        }
    }

    JsonValue parse_array(int depth) {
        expect('[');
        JsonValue v;
        v.type = JsonValue::Type::Array;
        skip_ws();
        if (peek() == ']') { ++pos_; return v; }
        for (;;) {
            v.array_value.push_back(parse_value(depth + 1));
            skip_ws();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return v; }
            fail("expected ',' or ']' in array");
        }
    }

    // Appends `cp` to `out` as UTF-8.
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

    uint32_t parse_hex4() {
        if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9')      v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid hex digit in \\u escape");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("unterminated escape");
            char e = s_[pos_++];
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
                    uint32_t cp = parse_hex4();
                    // Combine surrogate pairs so astral characters survive round-tripping.
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 1 < s_.size() &&
                        s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                        size_t save = pos_;
                        pos_ += 2;
                        uint32_t lo = parse_hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            pos_ = save;  // Unpaired high surrogate; emit it as-is.
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid escape character");
            }
        }
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        bool any = false;
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') { ++pos_; any = true; }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') { ++pos_; any = true; }
        }
        if (any && pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        }
        if (!any) fail("invalid number");

        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.number_value = std::stod(s_.substr(start, pos_ - start));
        return v;
    }
};

} // namespace detail

inline JsonValue JsonValue::parse(const std::string& text) {
    return detail::JsonParser(text).parse_document();
}

inline std::string JsonValue::quote(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 2);
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    o += "\\u00";
                    o += kHex[(c >> 4) & 0xF];
                    o += kHex[c & 0xF];
                } else {
                    // UTF-8 continuation bytes pass through unchanged; the input is assumed
                    // to be valid UTF-8, which the incremental detokenizer guarantees.
                    o += static_cast<char>(c);
                }
        }
    }
    o += '"';
    return o;
}

} // namespace gturbo
