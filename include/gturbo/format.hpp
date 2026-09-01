#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>
#include <set>
#include <vector>

// Set by CMake (add_compile_definitions in CMakeLists.txt). Defined here as well so the
// header stays usable in a translation unit built outside this project's build system.
#ifndef GTURBO_VERSION_STRING
#define GTURBO_VERSION_STRING "0.0.0-dev"
#endif

namespace gturbo {

enum class DType : uint8_t {
    U32  = 0,
    BF16 = 1,
    FP16 = 2,
    FP32 = 3
};

struct GTurboFormatV1 {
    static constexpr const char* MAGIC = "GTURBO";
    static constexpr uint32_t VERSION_MAJOR = 1;
    static constexpr uint32_t VERSION_MINOR = 0;
    static constexpr uint64_t ALIGNMENT_BYTES = 16384; // 16 KB sector/DMA alignment
    static constexpr size_t RESIDENT_HEADER_BYTES = 24;
    static constexpr size_t RESIDENT_ENTRY_BYTES = 72;
    static constexpr uint64_t RESIDENT_INDEX_MAX_BYTES = 16 * 1024 * 1024; // 16 MB

    static inline bool is_known_flag(std::string_view flag) {
        return flag == "streamingPresent" || flag == "turboQuantKV" || flag == "aneSharedExpert";
    }
};

class GTurboFormatError : public std::runtime_error {
public:
    explicit GTurboFormatError(const std::string& message)
        : std::runtime_error(message) {}
};

inline uint64_t checked_add(uint64_t a, uint64_t b, std::string_view field) {
    uint64_t res = a + b;
    if (res < a) {
        throw GTurboFormatError(std::string(field) + ": arithmetic overflow");
    }
    return res;
}

inline uint64_t checked_multiply(uint64_t a, uint64_t b, std::string_view field) {
    if (a == 0 || b == 0) return 0;
    uint64_t res = a * b;
    if (res / a != b) {
        throw GTurboFormatError(std::string(field) + ": arithmetic overflow");
    }
    return res;
}

class PathValidator {
public:
    static bool validate_relative_path(std::string_view path, std::string_view field) {
        if (path.empty() || path.front() == '/' || path.front() == '\\') {
            throw GTurboFormatError(std::string(field) + ": unsafe relative path");
        }
        if (path.find('\0') != std::string_view::npos) {
            throw GTurboFormatError(std::string(field) + ": null character in path");
        }
        if (path.find("..") != std::string_view::npos) {
            throw GTurboFormatError(std::string(field) + ": non-canonical path containing '..'");
        }
        return true;
    }

    static bool validate_basename(std::string_view name, std::string_view field) {
        validate_relative_path(name, field);
        if (name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos) {
            throw GTurboFormatError(std::string(field) + ": expected basename without directory separators");
        }
        return true;
    }
};

} // namespace gturbo
