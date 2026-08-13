#include "gturbo/cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <windows.h>

namespace fs = std::filesystem;

namespace gturbo {

// ---------------------------------------------------------------------------
// Numeric primitives
// ---------------------------------------------------------------------------

static inline float bf16(uint16_t bits) {
    uint32_t w = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

// MLX affine dequantization: w = q * scale + bias, group_size 64 along the input dim.
// Weights are packed low-nibble-first, so element c lives in byte c/2, in the low nibble
// when c is even. Confirmed against moe.metal:224-241.
static constexpr uint32_t GROUP_SIZE = 64;

// gelu_pytorch_tanh, the activation named in config.json.
static inline float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCubic = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + kCubic * x * x * x)));
}

// y = x / sqrt(mean(x^2) + eps), optionally scaled by w.
// Note the weight is applied as plain `w`, NOT `1 + w` -- the offset is already baked into
// the checkpoint (rmsnorm.metal:88-92). Using 1+w here yields fluent but wrong output.
static void rms_norm(const float* x, const uint16_t* w, size_t n, float eps, float* out) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(acc / static_cast<double>(n)) + eps);
    if (w) {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * bf16(w[i]);
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv;
    }
}

// out[row] = sum_c (q[row][c] * scale + bias) * x[c]
static void matvec(const QuantTensor& t, const float* x, float* out) {
    const uint32_t groups = t.groups;
    const uint32_t in_dim = t.in_dim;

    if (t.bits == 4) {
        const size_t row_bytes = in_dim / 2;
        for (uint32_t r = 0; r < t.rows; ++r) {
            const uint8_t* wrow = t.data + static_cast<size_t>(r) * row_bytes;
            const uint16_t* srow = t.scales + static_cast<size_t>(r) * groups;
            const uint16_t* brow = t.biases + static_cast<size_t>(r) * groups;
            float acc = 0.0f;
            for (uint32_t g = 0; g < groups; ++g) {
                const float s = bf16(srow[g]);
                const float b = bf16(brow[g]);
                const uint32_t base = g * GROUP_SIZE;
                float dot = 0.0f, sum = 0.0f;
                for (uint32_t j = 0; j < GROUP_SIZE; j += 2) {
                    const uint8_t byte = wrow[(base + j) / 2];
                    const float x0 = x[base + j];
                    const float x1 = x[base + j + 1];
                    dot += static_cast<float>(byte & 0x0F) * x0;
                    dot += static_cast<float>(byte >> 4) * x1;
                    sum += x0 + x1;
                }
                acc += s * dot + b * sum;
            }
            out[r] = acc;
        }
        return;
    }

    // 8-bit: one byte per element (routers only).
    for (uint32_t r = 0; r < t.rows; ++r) {
        const uint8_t* wrow = t.data + static_cast<size_t>(r) * in_dim;
        const uint16_t* srow = t.scales + static_cast<size_t>(r) * groups;
        const uint16_t* brow = t.biases + static_cast<size_t>(r) * groups;
        float acc = 0.0f;
        for (uint32_t g = 0; g < groups; ++g) {
            const float s = bf16(srow[g]);
            const float b = bf16(brow[g]);
            const uint32_t base = g * GROUP_SIZE;
            float dot = 0.0f, sum = 0.0f;
            for (uint32_t j = 0; j < GROUP_SIZE; ++j) {
                const float xv = x[base + j];
                dot += static_cast<float>(wrow[base + j]) * xv;
                sum += xv;
            }
            acc += s * dot + b * sum;
        }
        out[r] = acc;
    }
}

// Dequantizes a single row (used for the embedding lookup).
static void dequant_row(const QuantTensor& t, uint32_t row, float* out) {
    const size_t row_bytes = t.in_dim / 2;
    const uint8_t* wrow = t.data + static_cast<size_t>(row) * row_bytes;
    const uint16_t* srow = t.scales + static_cast<size_t>(row) * t.groups;
    const uint16_t* brow = t.biases + static_cast<size_t>(row) * t.groups;
    for (uint32_t g = 0; g < t.groups; ++g) {
        const float s = bf16(srow[g]);
        const float b = bf16(brow[g]);
        const uint32_t base = g * GROUP_SIZE;
        for (uint32_t j = 0; j < GROUP_SIZE; ++j) {
            const uint8_t byte = wrow[(base + j) / 2];
            const uint32_t q = ((base + j) % 2 == 0) ? (byte & 0x0F) : (byte >> 4u);
            out[base + j] = static_cast<float>(q) * s + b;
        }
    }
}

// NeoX-style rotary: pairs are (i, i + head_dim/2), freq = theta^(-2i/head_dim).
// `rotated_pairs` is less than head_dim/2 on full-attention layers, which use
// partial_rotary_factor 0.25 -- 64 of 256 pairs (fused.metal:54-70).
static void rope_neox(float* vec, uint32_t head_dim, uint32_t rotated_pairs,
                      float position, float theta) {
    const uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < rotated_pairs; ++i) {
        const float freq = std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
        const float angle = position * freq;
        const float c = std::cos(angle), s = std::sin(angle);
        const float x0 = vec[i], x1 = vec[i + half];
        vec[i] = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

static void dump(const std::string& dir, const std::string& name,
                 const float* data, size_t n) {
    if (dir.empty()) return;
    std::error_code ec;
    fs::create_directories(dir, ec);
    std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

// ---------------------------------------------------------------------------

CpuReference::CpuReference(const GTurboManifestV1& manifest,
                           const PackedExpertsLayoutV1& layout,
                           const std::string& model_dir)
    : manifest_(manifest), layout_(layout), model_dir_(model_dir) {}

CpuReference::~CpuReference() {
    if (resident_) UnmapViewOfFile(resident_);
    if (map_handle_) CloseHandle(map_handle_);
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
    for (void* h : layer_files_) {
        if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
}

void CpuReference::load() {
    const std::string tok = model_dir_ + "/tokenizer/tokenizer.json";
    if (!fs::exists(tok)) {
        throw GTurboFormatError("No tokenizer at " + tok);
    }
    tokenizer_.load_vocabulary(tok);

    // Memory-map resident.bin rather than reading it, so the 1.3 GB stays file-backed.
    const std::string path = model_dir_ + "/resident.bin";
    std::wstring wpath(path.begin(), path.end());
    HANDLE fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw GTurboFormatError("Cannot open " + path);
    }
    file_handle_ = fh;
    LARGE_INTEGER sz{};
    GetFileSizeEx(fh, &sz);
    resident_size_ = static_cast<uint64_t>(sz.QuadPart);

    HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) throw GTurboFormatError("CreateFileMapping failed for " + path);
    map_handle_ = mh;
    resident_ = static_cast<const uint8_t*>(MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0));
    if (!resident_) throw GTurboFormatError("MapViewOfFile failed for " + path);

    auto header = ResidentIndexCodec::decode_header(resident_, resident_size_);
    auto entries = ResidentIndexCodec::decode_region(resident_, resident_size_, header);
    for (const auto& e : entries) entries_[e.name] = e;
    if (entries_.empty()) {
        throw GTurboFormatError("resident.bin declares no tensors");
    }

    layer_files_.assign(static_cast<size_t>(manifest_.num_layers), nullptr);
    for (int l = 0; l < manifest_.num_layers; ++l) {
        // Same fixed-buffer trap as the GPU path had: a char[64] holds only a short relative
        // model path, and an absolute one truncates into a file that does not exist.
        char suffix[32];
        snprintf(suffix, sizeof(suffix), "/packed_experts/layer_%02d.bin", l);
        const std::string lp = model_dir_ + suffix;
        std::wstring wlp(lp.begin(), lp.end());
        HANDLE h = CreateFileW(wlp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            throw GTurboFormatError("Cannot open " + lp);
        }
        layer_files_[static_cast<size_t>(l)] = h;
    }
    expert_block_.resize(static_cast<size_t>(layout_.expert_stride));

    kv_.assign(static_cast<size_t>(manifest_.num_layers), LayerKV{});
    position_ = 0;
}

QuantTensor CpuReference::tensor(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw GTurboFormatError("resident index has no tensor '" + name + "'");
    }
    const auto& e = it->second;

    QuantTensor t;
    t.data = resident_ + e.file_offset;
    t.rows = e.shape.empty() ? 1 : e.shape[0];
    t.quantized = (e.scale_size > 0);

    if (t.quantized) {
        t.scales = reinterpret_cast<const uint16_t*>(resident_ + e.scale_offset);
        t.biases = reinterpret_cast<const uint16_t*>(resident_ + e.bias_offset);
        // Packed width tells us the bit depth: the scales table has in_dim/64 entries per
        // row, so in_dim = (scale_bytes / 2 / rows) * 64.
        const uint32_t groups = static_cast<uint32_t>(e.scale_size / 2 / t.rows);
        t.groups = groups;
        t.in_dim = groups * GROUP_SIZE;
        const uint64_t packed_per_row = (e.shape.size() > 1) ? e.shape[1] : 0;
        // shape[1] counts u32 words: 8 values each at 4 bits, 4 at 8 bits.
        t.bits = (packed_per_row * 8 == t.in_dim) ? 4 : 8;
    } else {
        t.in_dim = t.rows;
        t.groups = 0;
        t.bits = 16;
    }
    return t;
}

QuantTensor CpuReference::layer_tensor(int layer, const std::string& suffix) const {
    return tensor("model.layers." + std::to_string(layer) + "." + suffix);
}

const uint16_t* CpuReference::bf16_tensor(const std::string& name, uint32_t expect_len) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw GTurboFormatError("resident index has no tensor '" + name + "'");
    }
    const auto& e = it->second;
    if (e.size_bytes < static_cast<uint64_t>(expect_len) * 2) {
        throw GTurboFormatError(name + ": expected at least " +
                                std::to_string(expect_len) + " BF16 values");
    }
    return reinterpret_cast<const uint16_t*>(resident_ + e.file_offset);
}

void CpuReference::load_expert(int layer, int expert) {
    HANDLE h = static_cast<HANDLE>(layer_files_[static_cast<size_t>(layer)]);
    LARGE_INTEGER off{};
    off.QuadPart = static_cast<LONGLONG>(layout_.expert_offset(expert));
    if (!SetFilePointerEx(h, off, nullptr, FILE_BEGIN)) {
        throw GTurboFormatError("Seek failed in layer " + std::to_string(layer));
    }
    DWORD got = 0;
    if (!ReadFile(h, expert_block_.data(), static_cast<DWORD>(expert_block_.size()), &got, nullptr) ||
        got != expert_block_.size()) {
        throw GTurboFormatError("Short read for expert " + std::to_string(expert) +
                                " in layer " + std::to_string(layer));
    }
}

// Builds a QuantTensor over the currently-loaded expert block.
QuantTensor CpuReference::expert_tensor(const std::string& role) const {
    auto find = [&](const std::string& key) -> const SubTensorV1& {
        auto it = layout_.expert_block.find(key);
        if (it == layout_.expert_block.end()) {
            throw GTurboFormatError("layout.expertBlock missing '" + key + "'");
        }
        return it->second;
    };
    const SubTensorV1& w = find(role + ".weight");
    const SubTensorV1& s = find(role + ".scales");
    const SubTensorV1& b = find(role + ".biases");

    QuantTensor t;
    t.data = expert_block_.data() + w.offset;
    t.scales = reinterpret_cast<const uint16_t*>(expert_block_.data() + s.offset);
    t.biases = reinterpret_cast<const uint16_t*>(expert_block_.data() + b.offset);
    t.rows = w.shape.empty() ? 0 : w.shape[0];
    t.groups = static_cast<uint32_t>(s.size / 2 / t.rows);
    t.in_dim = t.groups * GROUP_SIZE;
    t.bits = 4;
    t.quantized = true;
    return t;
}

// ---------------------------------------------------------------------------
// The forward pass
// ---------------------------------------------------------------------------

void CpuReference::forward(uint32_t token, int pos, std::vector<float>& logits,
                           const std::string& dump_dir) {
    const auto& a = manifest_.arch;
    const uint32_t D = static_cast<uint32_t>(a.hidden_size);
    const float eps = static_cast<float>(a.rms_norm_eps);

    std::vector<float> hidden(D), normed(D), attn(D), tmp(D);
    std::vector<float> dense_x(D), routed_x(D), router_x(D), h1(D), h2(D);

    // --- Embedding. Scaled by sqrt(hidden_size); the LM head reuses this same matrix. ---
    const QuantTensor embed = tensor("model.embed_tokens.weight");
    if (token >= embed.rows) {
        throw GTurboFormatError("token id " + std::to_string(token) + " out of vocabulary");
    }
    dequant_row(embed, token, hidden.data());
    const float embed_scale = std::sqrt(static_cast<float>(D));
    for (uint32_t i = 0; i < D; ++i) hidden[i] *= embed_scale;
    dump(dump_dir, "embed", hidden.data(), D);

    for (int L = 0; L < a.num_layers; ++L) {
        const bool is_full = (L < static_cast<int>(a.full_attention_layer_mask.size())) &&
                             a.full_attention_layer_mask[static_cast<size_t>(L)] != 0;
        const uint32_t head_dim = static_cast<uint32_t>(is_full ? a.full_head_dim : a.head_dim);
        const uint32_t kv_heads = static_cast<uint32_t>(is_full ? a.num_full_kv_heads : a.num_kv_heads);
        const uint32_t q_heads = static_cast<uint32_t>(a.num_heads);
        const float theta = static_cast<float>(is_full ? a.full_rope_theta : a.rope_theta);
        // Partial rotary applies to full-attention layers only.
        const uint32_t rotated_pairs = is_full
            ? static_cast<uint32_t>(head_dim * a.partial_rotary_factor / 2.0)
            : head_dim / 2;

        const std::string P = "model.layers." + std::to_string(L) + ".";

        // --- Attention -----------------------------------------------------
        rms_norm(hidden.data(), bf16_tensor(P + "input_layernorm.weight", D), D, eps, normed.data());

        std::vector<float> q(q_heads * head_dim), k(kv_heads * head_dim), v(kv_heads * head_dim);
        matvec(layer_tensor(L, "self_attn.q_proj.weight"), normed.data(), q.data());
        matvec(layer_tensor(L, "self_attn.k_proj.weight"), normed.data(), k.data());
        if (a.attention_k_eq_v && is_full) {
            // Full-attention layers have no v_proj at all: V reuses the raw k_proj output,
            // then takes a no-scale norm and skips RoPE. Confirmed absent from the
            // checkpoint (layer 5 has no self_attn.v_proj.*).
            v = k;
        } else {
            matvec(layer_tensor(L, "self_attn.v_proj.weight"), normed.data(), v.data());
        }

        const uint16_t* qn = bf16_tensor(P + "self_attn.q_norm.weight", head_dim);
        const uint16_t* kn = bf16_tensor(P + "self_attn.k_norm.weight", head_dim);
        std::vector<float> scratch(head_dim);
        for (uint32_t h = 0; h < q_heads; ++h) {
            rms_norm(q.data() + h * head_dim, qn, head_dim, eps, scratch.data());
            std::copy(scratch.begin(), scratch.end(), q.begin() + h * head_dim);
            rope_neox(q.data() + h * head_dim, head_dim, rotated_pairs,
                      static_cast<float>(pos), theta);
        }
        for (uint32_t h = 0; h < kv_heads; ++h) {
            rms_norm(k.data() + h * head_dim, kn, head_dim, eps, scratch.data());
            std::copy(scratch.begin(), scratch.end(), k.begin() + h * head_dim);
            rope_neox(k.data() + h * head_dim, head_dim, rotated_pairs,
                      static_cast<float>(pos), theta);
            // V gets a no-scale RMSNorm and no rotation.
            rms_norm(v.data() + h * head_dim, nullptr, head_dim, eps, scratch.data());
            std::copy(scratch.begin(), scratch.end(), v.begin() + h * head_dim);
        }

        LayerKV& cache = kv_[static_cast<size_t>(L)];
        cache.k.insert(cache.k.end(), k.begin(), k.end());
        cache.v.insert(cache.v.end(), v.begin(), v.end());

        const uint32_t gqa = q_heads / kv_heads;
        const int n_pos = pos + 1;
        // Sliding-window layers only attend to the last `sliding_window` positions.
        const int first = is_full ? 0 : std::max(0, n_pos - a.sliding_window);

        std::vector<float> ctx(q_heads * head_dim, 0.0f);
        std::vector<float> scores(static_cast<size_t>(n_pos));
        for (uint32_t h = 0; h < q_heads; ++h) {
            const uint32_t kvh = h / gqa;
            const float* qh = q.data() + h * head_dim;
            float best = -1e30f;
            for (int t = first; t < n_pos; ++t) {
                const float* kt = cache.k.data() +
                    static_cast<size_t>(t) * kv_heads * head_dim + kvh * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
                // Attention scale is 1.0, not 1/sqrt(head_dim): the query scaling is
                // already absorbed into q_norm (RealForwardRunner.swift:1404).
                scores[static_cast<size_t>(t)] = dot;
                best = std::max(best, dot);
            }
            float sum = 0.0f;
            for (int t = first; t < n_pos; ++t) {
                float e = std::exp(scores[static_cast<size_t>(t)] - best);
                scores[static_cast<size_t>(t)] = e;
                sum += e;
            }
            float* out = ctx.data() + h * head_dim;
            for (int t = first; t < n_pos; ++t) {
                const float wgt = scores[static_cast<size_t>(t)] / sum;
                const float* vt = cache.v.data() +
                    static_cast<size_t>(t) * kv_heads * head_dim + kvh * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) out[d] += wgt * vt[d];
            }
        }

        matvec(layer_tensor(L, "self_attn.o_proj.weight"), ctx.data(), attn.data());

        // --- Post-attention residual and the three pre-FFN views ------------
        rms_norm(attn.data(), bf16_tensor(P + "post_attention_layernorm.weight", D), D, eps, tmp.data());
        for (uint32_t i = 0; i < D; ++i) hidden[i] += tmp[i];
        rms_norm(hidden.data(), nullptr, D, eps, tmp.data());
        {
            const uint16_t* w_pre = bf16_tensor(P + "pre_feedforward_layernorm.weight", D);
            const uint16_t* w_pre2 = bf16_tensor(P + "pre_feedforward_layernorm_2.weight", D);
            for (uint32_t i = 0; i < D; ++i) {
                dense_x[i] = tmp[i] * bf16(w_pre[i]);   // shared expert
                routed_x[i] = tmp[i] * bf16(w_pre2[i]); // routed experts
                router_x[i] = tmp[i];                   // router (no scale)
            }
        }

        // --- Router ---------------------------------------------------------
        const QuantTensor router = layer_tensor(L, "router.proj.weight");
        std::vector<float> router_in(D);
        {
            const uint16_t* rs = bf16_tensor(P + "router.scale", D);
            const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
            for (uint32_t i = 0; i < D; ++i) router_in[i] = router_x[i] * bf16(rs[i]) * inv_sqrt_d;
        }
        std::vector<float> rlogits(static_cast<size_t>(a.num_experts));
        matvec(router, router_in.data(), rlogits.data());

        const int K = a.top_k_experts;
        std::vector<int> top(static_cast<size_t>(K));
        {
            std::vector<int> idx(static_cast<size_t>(a.num_experts));
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int x, int y) {
                                  if (rlogits[static_cast<size_t>(x)] !=
                                      rlogits[static_cast<size_t>(y)]) {
                                      return rlogits[static_cast<size_t>(x)] >
                                             rlogits[static_cast<size_t>(y)];
                                  }
                                  return x < y;  // ties go to the lower expert index
                              });
            std::copy(idx.begin(), idx.begin() + K, top.begin());
        }

        // Softmax over the top-K only, then scaled per expert.
        std::vector<float> weights(static_cast<size_t>(K));
        {
            const float best = rlogits[static_cast<size_t>(top[0])];
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                weights[static_cast<size_t>(i)] =
                    std::exp(rlogits[static_cast<size_t>(top[static_cast<size_t>(i)])] - best);
                sum += weights[static_cast<size_t>(i)];
            }
            const uint16_t* pes = bf16_tensor(P + "router.per_expert_scale",
                                              static_cast<uint32_t>(a.num_experts));
            for (int i = 0; i < K; ++i) {
                weights[static_cast<size_t>(i)] =
                    weights[static_cast<size_t>(i)] / sum * bf16(pes[top[static_cast<size_t>(i)]]);
            }
        }

        // --- Shared expert (added with no routing weight) --------------------
        {
            const QuantTensor wg = layer_tensor(L, "mlp.gate_proj.weight");
            const QuantTensor wu = layer_tensor(L, "mlp.up_proj.weight");
            const QuantTensor wd = layer_tensor(L, "mlp.down_proj.weight");
            std::vector<float> g(wg.rows), u(wu.rows), act(wg.rows), outv(D);
            matvec(wg, dense_x.data(), g.data());
            matvec(wu, dense_x.data(), u.data());
            for (uint32_t i = 0; i < wg.rows; ++i) act[i] = gelu_tanh(g[i]) * u[i];
            matvec(wd, act.data(), outv.data());
            rms_norm(outv.data(), bf16_tensor(P + "post_feedforward_layernorm_1.weight", D),
                     D, eps, h1.data());
        }

        // --- Routed experts --------------------------------------------------
        std::fill(h2.begin(), h2.end(), 0.0f);
        for (int i = 0; i < K; ++i) {
            load_expert(L, top[static_cast<size_t>(i)]);
            const QuantTensor eg = expert_tensor("gate_proj");
            const QuantTensor eu = expert_tensor("up_proj");
            const QuantTensor ed = expert_tensor("down_proj");
            std::vector<float> g(eg.rows), u(eu.rows), act(eg.rows), outv(D);
            matvec(eg, routed_x.data(), g.data());
            matvec(eu, routed_x.data(), u.data());
            for (uint32_t j = 0; j < eg.rows; ++j) act[j] = gelu_tanh(g[j]) * u[j];
            matvec(ed, act.data(), outv.data());
            const float wgt = weights[static_cast<size_t>(i)];
            for (uint32_t j = 0; j < D; ++j) h2[j] += wgt * outv[j];
        }

        // --- Layer tail (fused.metal:272-375) --------------------------------
        rms_norm(h2.data(), bf16_tensor(P + "post_feedforward_layernorm_2.weight", D),
                 D, eps, tmp.data());
        for (uint32_t i = 0; i < D; ++i) tmp[i] += h1[i];
        std::vector<float> t2(D);
        rms_norm(tmp.data(), bf16_tensor(P + "post_feedforward_layernorm.weight", D),
                 D, eps, t2.data());
        const float layer_scalar = bf16(*bf16_tensor(P + "layer_scalar", 1));
        for (uint32_t i = 0; i < D; ++i) hidden[i] = (hidden[i] + t2[i]) * layer_scalar;

        if (!dump_dir.empty()) {
            dump(dump_dir, "layer" + std::to_string(L) + "_hidden", hidden.data(), D);
        }
    }

    // --- Final norm and the tied LM head ------------------------------------
    rms_norm(hidden.data(), bf16_tensor("model.norm.weight", D), D, eps, normed.data());
    dump(dump_dir, "final_norm", normed.data(), D);

    logits.assign(static_cast<size_t>(a.vocab_size), 0.0f);
    matvec(embed, normed.data(), logits.data());  // tie_word_embeddings

    // Final logit softcapping: 30 * tanh(z / 30).
    const float cap = static_cast<float>(a.final_logit_softcap);
    if (cap > 0.0f) {
        for (auto& z : logits) z = cap * std::tanh(z / cap);
    }
    dump(dump_dir, "logits", logits.data(), logits.size());
}

// ---------------------------------------------------------------------------

std::vector<uint32_t> CpuReference::generate(const std::vector<uint32_t>& prompt_tokens,
                                             const CpuGenerationOptions& opts) {
    if (prompt_tokens.empty()) {
        throw GTurboFormatError("generate: empty prompt");
    }
    for (auto& c : kv_) { c.k.clear(); c.v.clear(); }

    std::vector<float> logits;
    const SamplingParams sampling{opts.temperature, opts.top_p, opts.top_k,
                                  opts.repetition_penalty, opts.seed != 0, opts.seed};
    validate_sampling(sampling);
    const auto& stops = tokenizer_.stop_token_ids();

    std::vector<uint32_t> history;
    if (opts.repetition_penalty != 1.0f) history = prompt_tokens;

    // Prefill: every prompt token is consumed to build KV state.
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        forward(prompt_tokens[i], static_cast<int>(i),
                logits, (i == 0) ? opts.dump_dir : std::string());
        if (opts.verbose) {
            std::cout << "\r  prefill " << (i + 1) << "/" << prompt_tokens.size()
                      << std::flush;
        }
    }
    if (opts.verbose) std::cout << "\n";

    std::vector<uint32_t> out;
    int pos = static_cast<int>(prompt_tokens.size());

    for (int step = 0; step < opts.max_tokens; ++step) {
        // Sampling lives in src/sampling.cpp so this path and the GPU path are the same code,
        // not two implementations that agree until one of them is edited. `logits` here are
        // already softcapped by forward(), which is what the penalty expects.
        apply_repetition_penalty(logits.data(), static_cast<uint32_t>(logits.size()), history,
                                 sampling.repetition_penalty, 30.0f);
        const uint32_t next = sample_token(logits.data(),
                                           static_cast<uint32_t>(logits.size()),
                                           sampling, seed_for(sampling, pos));

        if (std::find(stops.begin(), stops.end(), next) != stops.end()) break;
        out.push_back(next);
        if (opts.repetition_penalty != 1.0f) history.push_back(next);
        if (opts.verbose) {
            std::cout << tokenizer_.decode_single(next, true) << std::flush;
        }
        if (step + 1 >= opts.max_tokens) break;
        forward(next, pos++, logits, std::string());
    }
    if (opts.verbose) std::cout << "\n";
    return out;
}

std::string CpuReference::generate_text(const std::string& prompt,
                                        const CpuGenerationOptions& opts) {
    std::vector<Tokenizer::ChatMessage> msgs{{"user", prompt}};
    auto ids = tokenizer_.encode(tokenizer_.apply_chat_template(msgs), false);
    auto completion = generate(ids, opts);
    if (completion.empty()) {
        throw GTurboFormatError("CPU reference produced no tokens");
    }
    return tokenizer_.decode(completion, true);
}

} // namespace gturbo
