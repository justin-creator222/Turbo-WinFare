#include "gturbo/packed_experts.hpp"
#include "gturbo/json.hpp"
#include <sstream>
#include <set>

namespace gturbo {

// The nine sub-tensors every expert block must contain.
static const char* const REQUIRED_SUB_TENSORS[] = {
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
};

void PackedExpertsLayoutV1::validate() const {
    if (num_layers <= 0 || experts_per_layer <= 0 || expert_stride == 0 ||
        expert_stride % GTurboFormatV1::ALIGNMENT_BYTES != 0) {
        throw GTurboFormatError(
            "layout: invalid dimensions, or stride not a multiple of " +
            std::to_string(GTurboFormatV1::ALIGNMENT_BYTES));
    }
    if (static_cast<int>(layers.size()) != num_layers) {
        throw GTurboFormatError("layout.layers: expected " + std::to_string(num_layers) +
                                " entries, got " + std::to_string(layers.size()));
    }

    std::set<int> layer_ids;
    for (const auto& layer : layers) {
        if (layer.layer < 0 || layer.layer >= num_layers ||
            !layer_ids.insert(layer.layer).second) {
            throw GTurboFormatError("layout.layers: duplicate or out-of-range layer ID " +
                                    std::to_string(layer.layer));
        }
        PathValidator::validate_basename(layer.file, "layout.layers.file");
    }

    for (const char* name : REQUIRED_SUB_TENSORS) {
        if (expert_block.find(name) == expert_block.end()) {
            throw GTurboFormatError(
                std::string("layout.expertBlock: missing required sub-tensor '") + name + "'");
        }
    }

    // Every sub-tensor must fit inside one stride, and none may overlap another.
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (const auto& [name, t] : expert_block) {
        uint64_t end = checked_add(t.offset, t.size, "layout.expertBlock." + name);
        if (end > expert_stride) {
            throw GTurboFormatError("layout.expertBlock." + name + ": extends past expert_stride");
        }
        spans.emplace_back(t.offset, end);
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) {
            throw GTurboFormatError("layout.expertBlock: sub-tensors overlap");
        }
    }
}

void PackedExpertsLayoutV1::cross_validate(const GTurboManifestV1& manifest) const {
    validate();
    manifest.validate();
    if (manifest.num_layers != num_layers) {
        throw GTurboFormatError("manifest/layout: numLayers " +
                                std::to_string(manifest.num_layers) + " vs " +
                                std::to_string(num_layers));
    }
    if (manifest.experts_per_layer != experts_per_layer) {
        throw GTurboFormatError("manifest/layout: expertsPerLayer " +
                                std::to_string(manifest.experts_per_layer) + " vs " +
                                std::to_string(experts_per_layer));
    }
    if (manifest.expert_stride != expert_stride) {
        throw GTurboFormatError("manifest/layout: expertStride " +
                                std::to_string(manifest.expert_stride) + " vs " +
                                std::to_string(expert_stride));
    }
}

std::string PackedExpertsLayoutV1::to_json_string() const {
    validate();
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"expertStride\": " << expert_stride << ",\n";
    ss << "  \"numLayers\": " << num_layers << ",\n";
    ss << "  \"expertsPerLayer\": " << experts_per_layer << ",\n";

    ss << "  \"expertBlock\": {\n";
    size_t t_idx = 0;
    for (const auto& [name, t] : expert_block) {
        ss << "    \"" << name << "\": {\"offset\": " << t.offset
           << ", \"size\": " << t.size
           << ", \"dtype\": " << static_cast<int>(t.dtype)
           << ", \"shape\": [";
        for (size_t s = 0; s < t.shape.size(); ++s) {
            ss << t.shape[s] << (s + 1 < t.shape.size() ? ", " : "");
        }
        ss << "]}" << (++t_idx < expert_block.size() ? "," : "") << "\n";
    }
    ss << "  },\n";

    ss << "  \"layers\": [\n";
    for (size_t l = 0; l < layers.size(); ++l) {
        ss << "    {\"layer\": " << layers[l].layer
           << ", \"file\": \"" << layers[l].file << "\"}"
           << (l + 1 < layers.size() ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
}

PackedExpertsLayoutV1 PackedExpertsLayoutV1::from_json_string(const std::string& json_str) {
    // This used to ignore json_str and fabricate a layout from constants that did not match
    // the real model (expert_stride 692,224 instead of 3,358,720, and a group-32 "Q4_K_M"
    // sub-tensor table). Everything below now comes from the file.
    if (json_str.empty()) {
        throw GTurboFormatError("layout: empty JSON (did you forget to read layout.json?)");
    }

    JsonValue root = JsonValue::parse(json_str);
    if (!root.is_object()) {
        throw GTurboFormatError("layout: top level must be an object");
    }

    PackedExpertsLayoutV1 layout;
    layout.expert_stride     = static_cast<uint64_t>(root.at("expertStride", "layout").as_int("layout.expertStride"));
    layout.num_layers        = static_cast<int>(root.at("numLayers", "layout").as_int("layout.numLayers"));
    layout.experts_per_layer = static_cast<int>(root.at("expertsPerLayer", "layout").as_int("layout.expertsPerLayer"));

    const JsonValue& block = root.at("expertBlock", "layout");
    if (!block.is_object()) {
        throw GTurboFormatError("layout.expertBlock: expected an object");
    }
    for (const auto& [name, entry] : block.object_value) {
        const std::string ctx = "layout.expertBlock." + name;
        SubTensorV1 t;
        t.offset = static_cast<uint64_t>(entry.at("offset", ctx).as_int(ctx + ".offset"));
        t.size   = static_cast<uint64_t>(entry.at("size", ctx).as_int(ctx + ".size"));
        t.dtype  = static_cast<uint8_t>(entry.int_or("dtype", 0));
        if (entry.has("shape")) {
            const JsonValue& shape = entry.at("shape", ctx);
            if (!shape.is_array()) {
                throw GTurboFormatError(ctx + ".shape: expected an array");
            }
            for (const auto& dim : shape.array_value) {
                t.shape.push_back(static_cast<uint32_t>(dim.as_int(ctx + ".shape[]")));
            }
        }
        layout.expert_block[name] = t;
    }

    const JsonValue& layers = root.at("layers", "layout");
    if (!layers.is_array()) {
        throw GTurboFormatError("layout.layers: expected an array");
    }
    for (const auto& entry : layers.array_value) {
        LayerV1 l;
        l.layer = static_cast<int>(entry.at("layer", "layout.layers").as_int("layout.layers.layer"));
        l.file  = entry.at("file", "layout.layers").as_string("layout.layers.file");
        layout.layers.push_back(l);
    }

    layout.validate();
    return layout;
}

} // namespace gturbo
