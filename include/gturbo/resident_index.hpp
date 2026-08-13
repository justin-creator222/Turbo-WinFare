#pragma once

#include "gturbo/format.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace gturbo {

struct ResidentIndexHeaderV1 {
    uint64_t index_size{0};
    uint64_t resident_size{0};
    uint64_t entry_count{0};
};

struct ResidentIndexEntryV1 {
    std::string name;
    uint8_t dtype{0};
    uint64_t file_offset{0};
    uint64_t size_bytes{0};
    std::vector<uint32_t> shape; // 4 dimensions
    uint64_t scale_offset{0};
    uint64_t scale_size{0};
    uint64_t bias_offset{0};
    uint64_t bias_size{0};
};

class ResidentIndexCodec {
public:
    static ResidentIndexHeaderV1 decode_header(const uint8_t* bytes, size_t count);
    static std::vector<ResidentIndexEntryV1> decode_region(const uint8_t* bytes, size_t count, const ResidentIndexHeaderV1& header);
    
    static void write_header(uint8_t* buffer, const ResidentIndexHeaderV1& header);
    static void write_entry(uint8_t* buffer, const ResidentIndexEntryV1& entry, uint32_t name_offset);
};

} // namespace gturbo
