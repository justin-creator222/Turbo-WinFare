#pragma once

#include "gturbo/format.hpp"
#include "gturbo/manifest.hpp"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace gturbo {

// One sub-tensor within an expert block, at a fixed offset from the block start.
struct SubTensorV1 {
    uint64_t offset{0};
    uint64_t size{0};
    uint8_t dtype{0};   // DType enum in format.hpp
    std::vector<uint32_t> shape;
};

struct LayerV1 {
    int layer{0};
    std::string file; // "layer_00.bin"
};

// Layout of packed_experts/.
//
// Every expert block is byte-identical in structure, so the nine sub-tensor offsets are
// described once in `expert_block` rather than enumerated per expert. Expert e in layer L
// lives at `e * expert_stride` within `layers[L].file`, and its sub-tensors sit at
// `e * expert_stride + expert_block[name].offset`.
//
// The previous schema repeated the same nine entries for all 3,840 (layer, expert) pairs,
// which is what made layout.json 6.9 MB.
struct PackedExpertsLayoutV1 {
    uint64_t expert_stride{3358720};
    int num_layers{30};
    int experts_per_layer{128};
    std::map<std::string, SubTensorV1> expert_block; // keyed "gate_proj.weight" etc.
    std::vector<LayerV1> layers;

    // Byte offset of expert `expert_id`'s block within its layer file.
    uint64_t expert_offset(int expert_id) const {
        return static_cast<uint64_t>(expert_id) * expert_stride;
    }

    // Byte offset of one sub-tensor of one expert, within its layer file.
    uint64_t sub_tensor_offset(int expert_id, const std::string& name) const {
        auto it = expert_block.find(name);
        if (it == expert_block.end()) {
            throw GTurboFormatError("layout.expertBlock: no sub-tensor named '" + name + "'");
        }
        return expert_offset(expert_id) + it->second.offset;
    }

    void validate() const;
    void cross_validate(const GTurboManifestV1& manifest) const;
    std::string to_json_string() const;
    static PackedExpertsLayoutV1 from_json_string(const std::string& json_str);
};

} // namespace gturbo
