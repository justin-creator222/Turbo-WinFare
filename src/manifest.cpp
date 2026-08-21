#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/json.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

namespace gturbo {

std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw GTurboFormatError("Cannot open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void GTurboManifestV1::validate() const {
    if (magic != GTurboFormatV1::MAGIC) {
        throw GTurboFormatError("manifest.magic: expected GTURBO");
    }
    if (version_major != GTurboFormatV1::VERSION_MAJOR || version_minor < 0) {
        throw GTurboFormatError("manifest.version: unsupported version");
    }
    for (const auto& [flag, _] : flags) {
        if (!GTurboFormatV1::is_known_flag(flag)) {
            throw GTurboFormatError("manifest.flags: unknown v1 flag: " + flag);
        }
    }
    if (num_layers <= 0 || experts_per_layer <= 0 || expert_stride == 0 ||
        expert_stride % 4096 != 0) {
        throw GTurboFormatError("manifest: invalid dimensions or stride");
    }
    if (arch.num_layers != num_layers || arch.num_experts != experts_per_layer) {
        throw GTurboFormatError("manifest.arch: dimensions disagree with streaming metadata");
    }
    if (arch.hidden_size <= 0 || arch.ffn_intermediate <= 0 || arch.moe_intermediate_size <= 0 ||
        arch.num_heads <= 0 || arch.num_kv_heads <= 0 || arch.num_full_kv_heads <= 0 ||
        arch.head_dim <= 0 || arch.full_head_dim <= 0 || arch.vocab_size <= 0 ||
        arch.sliding_window <= 0 || arch.top_k_experts <= 0 || arch.top_k_experts > arch.num_experts) {
        throw GTurboFormatError("manifest.arch: invalid architecture values");
    }
}

std::string GTurboManifestV1::to_json_string() const {
    validate();
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"magic\": \"" << magic << "\",\n";
    ss << "  \"versionMajor\": " << version_major << ",\n";
    ss << "  \"versionMinor\": " << version_minor << ",\n";
    ss << "  \"modelID\": \"" << model_id << "\",\n";
    ss << "  \"numLayers\": " << num_layers << ",\n";
    ss << "  \"expertsPerLayer\": " << experts_per_layer << ",\n";
    ss << "  \"expertStride\": " << expert_stride << ",\n";
    
    // Flags
    ss << "  \"flags\": {\n";
    size_t f_idx = 0;
    for (const auto& [k, v] : flags) {
        ss << "    \"" << k << "\": " << (v ? "true" : "false");
        if (++f_idx < flags.size()) ss << ",";
        ss << "\n";
    }
    ss << "  },\n";

    // Arch
    ss << "  \"arch\": {\n";
    ss << "    \"hiddenSize\": " << arch.hidden_size << ",\n";
    ss << "    \"ffnIntermediate\": " << arch.ffn_intermediate << ",\n";
    ss << "    \"moeIntermediateSize\": " << arch.moe_intermediate_size << ",\n";
    ss << "    \"numHeads\": " << arch.num_heads << ",\n";
    ss << "    \"numKVHeads\": " << arch.num_kv_heads << ",\n";
    ss << "    \"numFullKVHeads\": " << arch.num_full_kv_heads << ",\n";
    ss << "    \"headDim\": " << arch.head_dim << ",\n";
    ss << "    \"fullHeadDim\": " << arch.full_head_dim << ",\n";
    ss << "    \"vocabSize\": " << arch.vocab_size << ",\n";
    ss << "    \"slidingWindow\": " << arch.sliding_window << ",\n";
    ss << "    \"finalLogitSoftcap\": " << arch.final_logit_softcap << ",\n";
    ss << "    \"ropeTheta\": " << arch.rope_theta << ",\n";
    ss << "    \"fullRopeTheta\": " << arch.full_rope_theta << ",\n";
    ss << "    \"partialRotaryFactor\": " << arch.partial_rotary_factor << ",\n";
    ss << "    \"numLayers\": " << arch.num_layers << ",\n";
    ss << "    \"numExperts\": " << arch.num_experts << ",\n";
    ss << "    \"topKExperts\": " << arch.top_k_experts << ",\n";
    ss << "    \"tieWordEmbeddings\": " << (arch.tie_word_embeddings ? "true" : "false") << ",\n";
    ss << "    \"attentionKEqV\": " << (arch.attention_k_eq_v ? "true" : "false") << ",\n";
    ss << "    \"hiddenActivation\": \"" << arch.hidden_activation << "\",\n";
    ss << "    \"fullAttentionLayerMask\": [";
    for (size_t i = 0; i < arch.full_attention_layer_mask.size(); ++i) {
        ss << arch.full_attention_layer_mask[i];
        if (i + 1 < arch.full_attention_layer_mask.size()) ss << ", ";
    }
    ss << "]\n  },\n";

    // Quant
    if (quant) {
        ss << "  \"quant\": {\n";
        auto emit_slot = [&](const char* name, const ManifestQuantSlotV1& slot, bool last) {
            ss << "    \"" << name << "\": {\n";
            ss << "      \"weightBits\": " << slot.weight_bits << ",\n";
            ss << "      \"scheme\": \"" << slot.scheme << "\",\n";
            ss << "      \"scaleType\": \"" << slot.scale_type << "\",\n";
            ss << "      \"biasType\": \"" << slot.bias_type << "\",\n";
            ss << "      \"groupSize\": " << slot.group_size << "\n";
            ss << "    }" << (last ? "" : ",") << "\n";
        };
        emit_slot("embedding", quant->embedding, false);
        emit_slot("attention", quant->attention, false);
        emit_slot("router", quant->router, false);
        emit_slot("sharedExpert", quant->shared_expert, false);
        emit_slot("routedExpert", quant->routed_expert, true);
        ss << "  },\n";
    }

    // Files
    ss << "  \"files\": {\n";
    size_t file_idx = 0;
    for (const auto& [name, file] : files) {
        ss << "    \"" << name << "\": {\n";
        ss << "      \"size\": " << file.size << ",\n";
        ss << "      \"sha256\": \"" << file.sha256 << "\"\n";
        ss << "    }" << (++file_idx < files.size() ? "," : "") << "\n";
    }
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

static ManifestQuantSlotV1 parse_quant_slot(const JsonValue& v, const std::string& ctx) {
    ManifestQuantSlotV1 s;
    s.weight_bits = static_cast<int>(v.int_or("weightBits", s.weight_bits));
    s.scheme      = v.string_or("scheme", s.scheme);
    s.scale_type  = v.string_or("scaleType", s.scale_type);
    s.bias_type   = v.string_or("biasType", s.bias_type);
    s.group_size  = static_cast<int>(v.int_or("groupSize", s.group_size));
    if (s.weight_bits != 4 && s.weight_bits != 8) {
        throw GTurboFormatError(ctx + ".weightBits: expected 4 or 8, got " +
                                std::to_string(s.weight_bits));
    }
    if (s.group_size <= 0) {
        throw GTurboFormatError(ctx + ".groupSize: must be positive");
    }
    return s;
}

GTurboManifestV1 GTurboManifestV1::from_json_string(const std::string& json_str) {
    // This used to ignore json_str entirely and return a hardcoded struct, so a bundle's
    // real metadata -- including its expert stride and attention layer mask -- was never
    // read. Everything below now comes from the file.
    if (json_str.empty()) {
        throw GTurboFormatError("manifest: empty JSON (did you forget to read manifest.json?)");
    }

    JsonValue root = JsonValue::parse(json_str);
    if (!root.is_object()) {
        throw GTurboFormatError("manifest: top level must be an object");
    }

    GTurboManifestV1 m;
    m.magic         = root.at("magic", "manifest").as_string("manifest.magic");
    m.version_major = static_cast<int>(root.at("versionMajor", "manifest").as_int("manifest.versionMajor"));
    m.version_minor = static_cast<int>(root.at("versionMinor", "manifest").as_int("manifest.versionMinor"));
    m.model_id      = root.string_or("modelID", "");
    m.num_layers        = static_cast<int>(root.at("numLayers", "manifest").as_int("manifest.numLayers"));
    m.experts_per_layer = static_cast<int>(root.at("expertsPerLayer", "manifest").as_int("manifest.expertsPerLayer"));
    m.expert_stride     = static_cast<uint64_t>(root.at("expertStride", "manifest").as_int("manifest.expertStride"));

    if (root.has("sourceSnapshotHash")) {
        m.source_snapshot_hash = root.string_or("sourceSnapshotHash", "");
    }

    if (root.has("flags")) {
        const JsonValue& f = root.at("flags", "manifest");
        for (const auto& [key, value] : f.object_value) {
            m.flags[key] = value.as_bool("manifest.flags." + key);
        }
    }

    const JsonValue& a = root.at("arch", "manifest");
    ManifestArchV1& arch = m.arch;
    arch.hidden_size           = static_cast<int>(a.int_or("hiddenSize", arch.hidden_size));
    arch.ffn_intermediate      = static_cast<int>(a.int_or("ffnIntermediate", arch.ffn_intermediate));
    arch.moe_intermediate_size = static_cast<int>(a.int_or("moeIntermediateSize", arch.moe_intermediate_size));
    arch.num_heads             = static_cast<int>(a.int_or("numHeads", arch.num_heads));
    arch.num_kv_heads          = static_cast<int>(a.int_or("numKVHeads", arch.num_kv_heads));
    arch.num_full_kv_heads     = static_cast<int>(a.int_or("numFullKVHeads", arch.num_full_kv_heads));
    arch.head_dim              = static_cast<int>(a.int_or("headDim", arch.head_dim));
    arch.full_head_dim         = static_cast<int>(a.int_or("fullHeadDim", arch.full_head_dim));
    arch.vocab_size            = static_cast<int>(a.int_or("vocabSize", arch.vocab_size));
    arch.sliding_window        = static_cast<int>(a.int_or("slidingWindow", arch.sliding_window));
    arch.final_logit_softcap   = a.double_or("finalLogitSoftcap", arch.final_logit_softcap);
    arch.rope_theta            = a.double_or("ropeTheta", arch.rope_theta);
    arch.full_rope_theta       = a.double_or("fullRopeTheta", arch.full_rope_theta);
    arch.partial_rotary_factor = a.double_or("partialRotaryFactor", arch.partial_rotary_factor);
    arch.rms_norm_eps          = a.double_or("rmsNormEps", arch.rms_norm_eps);
    arch.num_layers            = static_cast<int>(a.int_or("numLayers", m.num_layers));
    arch.num_experts           = static_cast<int>(a.int_or("numExperts", m.experts_per_layer));
    arch.top_k_experts         = static_cast<int>(a.int_or("topKExperts", arch.top_k_experts));
    arch.tie_word_embeddings   = a.bool_or("tieWordEmbeddings", arch.tie_word_embeddings);
    arch.attention_k_eq_v      = a.bool_or("attentionKEqV", arch.attention_k_eq_v);
    arch.hidden_activation     = a.string_or("hiddenActivation", arch.hidden_activation);

    arch.full_attention_layer_mask.clear();
    if (a.has("fullAttentionLayerMask")) {
        const JsonValue& mask = a.at("fullAttentionLayerMask", "manifest.arch");
        if (!mask.is_array()) {
            throw GTurboFormatError("manifest.arch.fullAttentionLayerMask: expected an array");
        }
        for (const auto& entry : mask.array_value) {
            arch.full_attention_layer_mask.push_back(
                static_cast<int>(entry.as_int("manifest.arch.fullAttentionLayerMask[]")) ? 1 : 0);
        }
    }

    if (root.has("quant")) {
        const JsonValue& q = root.at("quant", "manifest");
        ManifestQuantV1 quant;
        if (q.has("embedding"))    quant.embedding     = parse_quant_slot(q.at("embedding", "manifest.quant"), "manifest.quant.embedding");
        if (q.has("attention"))    quant.attention     = parse_quant_slot(q.at("attention", "manifest.quant"), "manifest.quant.attention");
        if (q.has("router"))       quant.router        = parse_quant_slot(q.at("router", "manifest.quant"), "manifest.quant.router");
        if (q.has("sharedExpert")) quant.shared_expert = parse_quant_slot(q.at("sharedExpert", "manifest.quant"), "manifest.quant.sharedExpert");
        if (q.has("routedExpert")) quant.routed_expert = parse_quant_slot(q.at("routedExpert", "manifest.quant"), "manifest.quant.routedExpert");
        m.quant = quant;
    }

    if (root.has("files")) {
        const JsonValue& files = root.at("files", "manifest");
        for (const auto& [name, entry] : files.object_value) {
            ManifestFileV1 f;
            f.size   = static_cast<uint64_t>(entry.int_or("size", 0));
            f.sha256 = entry.string_or("sha256", "");
            m.files[name] = f;
        }
    }

    m.validate();
    return m;
}

bool bundle_loads(const std::string& dir) {
    try {
        auto manifest = GTurboManifestV1::from_json_string(read_text_file(dir + "/manifest.json"));
        auto layout = PackedExpertsLayoutV1::from_json_string(
            read_text_file(dir + "/packed_experts/layout.json"));
        layout.cross_validate(manifest);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> bundle_search_roots() {
    namespace fs = std::filesystem;
    std::vector<std::string> roots;
    std::error_code ec;

    const fs::path cwd = fs::current_path(ec);
    if (!ec) roots.push_back(cwd.string());

#ifdef _WIN32
    wchar_t exe_buf[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, exe_buf, MAX_PATH) > 0) {
        const fs::path exe_dir = fs::path(exe_buf).parent_path();
        roots.push_back(exe_dir.string());
        roots.push_back(exe_dir.parent_path().string());
    }
#endif

    // De-duplicate; running from build/ makes the first two identical.
    std::vector<std::string> unique;
    for (const auto& r : roots) {
        if (r.empty()) continue;
        bool seen = false;
        for (const auto& u : unique) seen = seen || (u == r);
        if (!seen) unique.push_back(r);
    }
    return unique;
}

std::string resolve_bundle_path(const std::string& name_or_path) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // An explicit path is taken literally, exactly as the CLI's --model is.
    const bool explicit_path =
        name_or_path.find('/') != std::string::npos ||
        name_or_path.find('\\') != std::string::npos ||
        fs::path(name_or_path).is_absolute();
    if (explicit_path) return name_or_path;

    for (const auto& root : bundle_search_roots()) {
        const fs::path candidate = fs::path(root) / name_or_path;
        if (fs::is_directory(candidate, ec) && bundle_loads(candidate.string())) {
            return candidate.string();
        }
    }
    return name_or_path;
}

} // namespace gturbo
