#include "gturbo/resident_index.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace gturbo {

static inline uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

static inline uint64_t read_u64_le(const uint8_t* p) {
    uint64_t val = 0;
    for (int i = 7; i >= 0; --i) {
        val = (val << 8) | p[i];
    }
    return val;
}

static inline void write_u16_le(uint8_t* p, uint16_t val) {
    p[0] = static_cast<uint8_t>(val & 0xFF);
    p[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
}

static inline void write_u32_le(uint8_t* p, uint32_t val) {
    p[0] = static_cast<uint8_t>(val & 0xFF);
    p[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

static inline void write_u64_le(uint8_t* p, uint64_t val) {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>((val >> (i * 8)) & 0xFF);
    }
}

ResidentIndexHeaderV1 ResidentIndexCodec::decode_header(const uint8_t* bytes, size_t count) {
    if (count < GTurboFormatV1::RESIDENT_HEADER_BYTES) {
        throw GTurboFormatError("resident.header: truncated header");
    }
    ResidentIndexHeaderV1 header;
    header.index_size = read_u64_le(bytes + 0);
    header.resident_size = read_u64_le(bytes + 8);
    header.entry_count = read_u64_le(bytes + 16);
    return header;
}

std::vector<ResidentIndexEntryV1> ResidentIndexCodec::decode_region(const uint8_t* bytes, size_t count, const ResidentIndexHeaderV1& header) {
    if (header.index_size > GTurboFormatV1::RESIDENT_INDEX_MAX_BYTES) {
        throw GTurboFormatError("resident.indexSize: exceeds max cap");
    }
    if (header.index_size > count || header.index_size < GTurboFormatV1::RESIDENT_HEADER_BYTES ||
        header.index_size % GTurboFormatV1::ALIGNMENT_BYTES != 0) {
        throw GTurboFormatError("resident.index: truncated or unaligned index");
    }
    uint64_t table_bytes = checked_multiply(header.entry_count, GTurboFormatV1::RESIDENT_ENTRY_BYTES, "resident.entryTable");
    uint64_t table_end = checked_add(GTurboFormatV1::RESIDENT_HEADER_BYTES, table_bytes, "resident.entryTable");
    if (table_end > header.index_size) {
        throw GTurboFormatError("resident.entryTable: outside index range");
    }

    std::vector<ResidentIndexEntryV1> entries;
    entries.reserve(static_cast<size_t>(header.entry_count));

    for (uint64_t i = 0; i < header.entry_count; ++i) {
        size_t offset = GTurboFormatV1::RESIDENT_HEADER_BYTES + static_cast<size_t>(i) * GTurboFormatV1::RESIDENT_ENTRY_BYTES;
        const uint8_t* entry_ptr = bytes + offset;

        uint32_t name_offset = read_u32_le(entry_ptr + 0);
        uint16_t name_len = read_u16_le(entry_ptr + 4);
        uint8_t dtype = entry_ptr[6];

        if (entry_ptr[7] != 0) {
            throw GTurboFormatError("resident.entry: reserved byte must be zero");
        }
        uint64_t name_end = checked_add(name_offset, name_len, "resident.name");
        if (name_offset < table_end || name_end > header.index_size) {
            throw GTurboFormatError("resident.entry.name: range outside string table");
        }

        std::string name(reinterpret_cast<const char*>(bytes + name_offset), name_len);

        ResidentIndexEntryV1 entry;
        entry.name = name;
        entry.dtype = dtype;
        entry.file_offset = read_u64_le(entry_ptr + 8);
        entry.size_bytes = read_u64_le(entry_ptr + 16);
        entry.shape = {
            read_u32_le(entry_ptr + 24),
            read_u32_le(entry_ptr + 28),
            read_u32_le(entry_ptr + 32),
            read_u32_le(entry_ptr + 36)
        };
        entry.scale_offset = read_u64_le(entry_ptr + 40);
        entry.scale_size = read_u64_le(entry_ptr + 48);
        entry.bias_offset = read_u64_le(entry_ptr + 56);
        entry.bias_size = read_u64_le(entry_ptr + 64);

        entries.push_back(entry);
    }
    return entries;
}

void ResidentIndexCodec::write_header(uint8_t* buffer, const ResidentIndexHeaderV1& header) {
    write_u64_le(buffer + 0, header.index_size);
    write_u64_le(buffer + 8, header.resident_size);
    write_u64_le(buffer + 16, header.entry_count);
}

void ResidentIndexCodec::write_entry(uint8_t* buffer, const ResidentIndexEntryV1& entry, uint32_t name_offset) {
    write_u32_le(buffer + 0, name_offset);
    write_u16_le(buffer + 4, static_cast<uint16_t>(entry.name.length()));
    buffer[6] = entry.dtype;
    buffer[7] = 0; // Reserved
    write_u64_le(buffer + 8, entry.file_offset);
    write_u64_le(buffer + 16, entry.size_bytes);
    for (size_t i = 0; i < 4; ++i) {
        uint32_t dim = (i < entry.shape.size()) ? entry.shape[i] : 0;
        write_u32_le(buffer + 24 + i * 4, dim);
    }
    write_u64_le(buffer + 40, entry.scale_offset);
    write_u64_le(buffer + 48, entry.scale_size);
    write_u64_le(buffer + 56, entry.bias_offset);
    write_u64_le(buffer + 64, entry.bias_size);
}

} // namespace gturbo
