// GPU-vs-CPU kernel verification.
//
// Each DirectCompute kernel is checked against the same maths the CPU reference
// (src/cpu_reference.cpp) performs, on synthetic data with a known answer. This is the
// mechanism that makes the Stage 3 port tractable: when the logits are wrong, this says
// *which kernel* is wrong instead of leaving 30 layers to bisect by hand.
//
// Synthetic data is used rather than real weights so the test runs in milliseconds and
// needs no model bundle.

#include "gturbo/d3d12_context.hpp"
#include "gturbo/pipeline.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace gturbo;

namespace {

constexpr uint32_t GROUP_SIZE = 64;

float bf16_of(uint16_t bits) {
    uint32_t w = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

uint16_t bf16_bits(float f) {
    uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    // Round-to-nearest-even on the truncated mantissa.
    uint32_t rounding = 0x7FFF + ((w >> 16) & 1);
    return static_cast<uint16_t>((w + rounding) >> 16);
}

// Uploads bytes into a UMA buffer.
ComPtr<ID3D12Resource> upload(D3D12Context& ctx, const void* data, size_t bytes,
                              const char* name) {
    auto buf = ctx.create_uma_buffer(bytes, name);
    void* p = nullptr;
    D3D12_RANGE r{0, bytes};
    if (FAILED(buf->Map(0, &r, &p)) || !p) throw std::runtime_error("map failed");
    std::memcpy(p, data, bytes);
    buf->Unmap(0, nullptr);
    return buf;
}

std::vector<float> readback(ComPtr<ID3D12Resource>& buf, size_t count) {
    std::vector<float> out(count);
    void* p = nullptr;
    D3D12_RANGE r{0, count * sizeof(float)};
    if (FAILED(buf->Map(0, &r, &p)) || !p) throw std::runtime_error("map failed");
    std::memcpy(out.data(), p, count * sizeof(float));
    buf->Unmap(0, nullptr);
    return out;
}

// Agreement is judged with numpy's allclose criterion: |got - want| <= atol + rtol*|want|.
//
// A pure relative test is the wrong metric here. GPU transcendentals (tanh, sin/cos, pow)
// differ from libm in the last couple of ULPs, and near-zero results then show a large
// relative error for an absolute error of ~1e-7 -- Attention's worst case was 2.0173e-05
// against 2.0182e-05. Meanwhile a real logic bug is never marginal: the Wave64 reduction
// bug showed up as a 0.29 relative error across every element.
struct Compare {
    double max_abs{0.0};
    double max_rel{0.0};
    double worst_excess{0.0};   // how far past tolerance the worst element was
    size_t worst{0};
};

constexpr double ATOL = 1e-5;
constexpr double RTOL = 1e-4;

Compare compare(const std::vector<float>& got, const std::vector<float>& want) {
    Compare c;
    for (size_t i = 0; i < want.size(); ++i) {
        const double a = std::fabs(static_cast<double>(got[i]) - want[i]);
        const double denom = std::max(1e-12, std::fabs(static_cast<double>(want[i])));
        c.max_rel = std::max(c.max_rel, a / denom);
        c.max_abs = std::max(c.max_abs, a);
        const double excess = a - (ATOL + RTOL * std::fabs(static_cast<double>(want[i])));
        if (excess > c.worst_excess) { c.worst_excess = excess; c.worst = i; }
    }
    return c;
}

void report(const char* label, const Compare& c,
            const std::vector<float>& got, const std::vector<float>& want, bool& ok) {
    const bool pass = (c.worst_excess <= 0.0);
    std::cout << (pass ? "  [PASS] " : "  [FAIL] ") << label
              << "  max_abs=" << c.max_abs << "  max_rel=" << c.max_rel;
    if (!pass) {
        std::cout << "  worst[" << c.worst << "] got=" << got[c.worst]
                  << " want=" << want[c.worst] << " excess=" << c.worst_excess;
        ok = false;
    }
    std::cout << "\n";
}

// A synthetic affine-quantized matrix plus its exact CPU dequantized reference.
struct QuantMatrix {
    std::vector<uint8_t> packed;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> biases;
    uint32_t rows{0};
    uint32_t in_dim{0};
    int bits{4};

    void build(uint32_t r, uint32_t n, int b, std::mt19937& rng) {
        rows = r; in_dim = n; bits = b;
        const uint32_t groups = n / GROUP_SIZE;
        packed.assign(bits == 4 ? static_cast<size_t>(r) * n / 2
                                : static_cast<size_t>(r) * n, 0);
        scales.resize(static_cast<size_t>(r) * groups);
        biases.resize(static_cast<size_t>(r) * groups);

        std::uniform_int_distribution<int> q(0, bits == 4 ? 15 : 255);
        std::uniform_real_distribution<float> sf(0.001f, 0.05f);
        std::uniform_real_distribution<float> bf(-0.4f, 0.4f);

        for (size_t i = 0; i < scales.size(); ++i) {
            scales[i] = bf16_bits(sf(rng));
            biases[i] = bf16_bits(bf(rng));
        }
        for (uint32_t row = 0; row < r; ++row) {
            for (uint32_t c = 0; c < n; ++c) {
                const uint32_t v = static_cast<uint32_t>(q(rng));
                if (bits == 4) {
                    const size_t idx = static_cast<size_t>(row) * (n / 2) + c / 2;
                    // Low nibble first: element c is the low half when c is even.
                    if ((c & 1u) == 0u) packed[idx] = static_cast<uint8_t>((packed[idx] & 0xF0) | v);
                    else                packed[idx] = static_cast<uint8_t>((packed[idx] & 0x0F) | (v << 4));
                } else {
                    packed[static_cast<size_t>(row) * n + c] = static_cast<uint8_t>(v);
                }
            }
        }
    }

    uint32_t q_at(uint32_t row, uint32_t c) const {
        if (bits == 4) {
            const uint8_t byte = packed[static_cast<size_t>(row) * (in_dim / 2) + c / 2];
            return ((c & 1u) == 0u) ? (byte & 0x0Fu) : (byte >> 4);
        }
        return packed[static_cast<size_t>(row) * in_dim + c];
    }

    // The exact operation GemvInt4/GemvInt8 must reproduce.
    std::vector<float> matvec(const std::vector<float>& x) const {
        const uint32_t groups = in_dim / GROUP_SIZE;
        std::vector<float> out(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            float acc = 0.0f;
            for (uint32_t g = 0; g < groups; ++g) {
                const float s = bf16_of(scales[static_cast<size_t>(row) * groups + g]);
                const float b = bf16_of(biases[static_cast<size_t>(row) * groups + g]);
                for (uint32_t j = 0; j < GROUP_SIZE; ++j) {
                    const uint32_t c = g * GROUP_SIZE + j;
                    acc += (static_cast<float>(q_at(row, c)) * s + b) * x[c];
                }
            }
            out[row] = acc;
        }
        return out;
    }

    std::vector<float> dequant_row(uint32_t row) const {
        const uint32_t groups = in_dim / GROUP_SIZE;
        std::vector<float> out(in_dim);
        for (uint32_t g = 0; g < groups; ++g) {
            const float s = bf16_of(scales[static_cast<size_t>(row) * groups + g]);
            const float b = bf16_of(biases[static_cast<size_t>(row) * groups + g]);
            for (uint32_t j = 0; j < GROUP_SIZE; ++j) {
                const uint32_t c = g * GROUP_SIZE + j;
                out[c] = static_cast<float>(q_at(row, c)) * s + b;
            }
        }
        return out;
    }
};

} // namespace

int main(int argc, char** argv) {
    // Unbuffered: a GPU-side fault takes the process down hard, and buffered progress output
    // would be lost exactly when it is most needed to localize the failure.
    std::cout.setf(std::ios::unitbuf);
    std::cout << "[TEST] GPU kernels vs CPU reference maths\n";

    // `--dump <path>` writes the raw GEMV outputs so a refactor can be proven BIT-IDENTICAL
    // rather than merely within tolerance. Optimizations that only reorder memory access must
    // not change a single bit; floating-point addition is not associative, so "close enough"
    // would hide a reordered accumulation.
    std::string dump_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dump") dump_path = argv[i + 1];
    }
    std::vector<float> dump_blob;

    auto ctx = std::make_shared<D3D12Context>();
    try {
        ctx->initialize(false);
    } catch (const std::exception& ex) {
        std::cout << "  [SKIP] no D3D12 device: " << ex.what() << "\n";
        return 0;
    }
    std::cout << "  device: " << ctx->adapter_name() << "\n";

    ComputePipelineManager pm(ctx);
    try {
        pm.initialize_pipelines(gturbo::ComputePipelineManager::kLegacyDescriptorCapacity);
    } catch (const std::exception& ex) {
        std::cout << "  [FAIL] pipeline init: " << ex.what() << "\n";
        return 1;
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> xf(-1.5f, 1.5f);
    bool ok = true;

    auto run = [&](KernelType k, const std::vector<ID3D12Resource*>& srvs,
                   const std::vector<ID3D12Resource*>& uavs,
                   const KernelDispatchParams& p) {
        auto cl = ctx->reset_command_list();
        pm.dispatch(cl.Get(), k, srvs, uavs, p);
        cl->Close();
        ID3D12CommandList* lists[] = {cl.Get()};
        ctx->command_queue()->ExecuteCommandLists(1, lists);
        ctx->flush_gpu();
    };

    // --- GemvInt4: the workhorse (q/k/v/o, gate/up/down, experts, LM head) -------
    {
        const uint32_t rows = 96, in_dim = 2816;   // production hidden size
        QuantMatrix m;
        m.build(rows, in_dim, 4, rng);
        std::vector<float> x(in_dim);
        for (auto& v : x) v = xf(rng);

        auto bw = upload(*ctx, m.packed.data(), m.packed.size(), "w");
        auto bs = upload(*ctx, m.scales.data(), m.scales.size() * 2, "s");
        auto bb = upload(*ctx, m.biases.data(), m.biases.size() * 2, "b");
        auto bx = upload(*ctx, x.data(), x.size() * 4, "x");
        auto bo = ctx->create_uma_buffer(rows * 4, "out");

        KernelDispatchParams p{};
        p.grid_x = rows;
        p.set({rows, in_dim, 0, 0, /*gp1*/ 0, 0, 0, 0});
        run(KernelType::GemvInt4, {bw.Get(), bs.Get(), bb.Get(), bx.Get()}, {bo.Get()}, p);

        auto got = readback(bo, rows);
        auto want = m.matvec(x);
        // FP32 accumulation in a different order than the CPU loop, so exact equality is
        // not expected; 1e-4 relative is far tighter than any real divergence.
        report("GemvInt4  (96 rows x 2816, group 64)", compare(got, want), got, want, ok);
        dump_blob.insert(dump_blob.end(), got.begin(), got.end());
    }

    // --- EmbedLookup: dequantize one row, scaled by sqrt(D) ---------------------
    {
        const uint32_t rows = 64, in_dim = 2816;
        QuantMatrix m;
        m.build(rows, in_dim, 4, rng);
        const uint32_t token = 37;
        const float scale = std::sqrt(static_cast<float>(in_dim));

        auto bw = upload(*ctx, m.packed.data(), m.packed.size(), "w");
        auto bs = upload(*ctx, m.scales.data(), m.scales.size() * 2, "s");
        auto bb = upload(*ctx, m.biases.data(), m.biases.size() * 2, "b");
        auto bo = ctx->create_uma_buffer(in_dim * 4, "out");

        uint32_t sbits;
        std::memcpy(&sbits, &scale, 4);
        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({token, in_dim, 0, 0, /*gp1*/ 0, 0, sbits, 0});
        run(KernelType::EmbedLookup, {bw.Get(), bs.Get(), bb.Get()}, {bo.Get()}, p);

        auto got = readback(bo, in_dim);
        auto want = m.dequant_row(token);
        for (auto& v : want) v *= scale;
        report("EmbedLookup (row 37, x sqrt(2816))", compare(got, want), got, want, ok);
    }

    // --- RMSNormK, unweighted and weighted --------------------------------------
    {
        const uint32_t n = 2816;
        const float eps = 1e-6f;
        std::vector<float> x(n);
        for (auto& v : x) v = xf(rng);
        std::vector<uint16_t> w(n);
        for (auto& v : w) v = bf16_bits(xf(rng) * 0.3f);

        auto bx = upload(*ctx, x.data(), x.size() * 4, "x");
        auto bw = upload(*ctx, w.data(), w.size() * 2, "w");
        auto bo = ctx->create_uma_buffer(n * 4, "out");

        uint32_t ebits;
        std::memcpy(&ebits, &eps, 4);

        double acc = 0.0;
        for (float v : x) acc += static_cast<double>(v) * v;
        const float inv = 1.0f / std::sqrt(static_cast<float>(acc / n) + eps);

        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({n, 0, 0, 0, /*gp1*/ 0, ebits, 0, 0});
            run(KernelType::RMSNormK, {bx.Get(), bw.Get()}, {bo.Get()}, p);
            auto got = readback(bo, n);
            std::vector<float> want(n);
            for (uint32_t i = 0; i < n; ++i) want[i] = x[i] * inv;
            report("RMSNormK  (no weight)", compare(got, want), got, want, ok);
        }
        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({n, 0, 0, 0, /*gp1*/ 1, ebits, 0, 0});
            run(KernelType::RMSNormK, {bx.Get(), bw.Get()}, {bo.Get()}, p);
            auto got = readback(bo, n);
            std::vector<float> want(n);
            // Plain `w`, NOT `1 + w` -- the offset is baked into the checkpoint. This
            // assertion is the guard against that regression.
            for (uint32_t i = 0; i < n; ++i) want[i] = x[i] * inv * bf16_of(w[i]);
            report("RMSNormK  (BF16 weight, w not 1+w)", compare(got, want), got, want, ok);
        }
    }

    // --- GemvInt8: routers only -------------------------------------------------
    {
        const uint32_t rows = 128, in_dim = 2816;   // 128 experts
        QuantMatrix m;
        m.build(rows, in_dim, 8, rng);
        std::vector<float> x(in_dim);
        for (auto& v : x) v = xf(rng);

        auto bw = upload(*ctx, m.packed.data(), m.packed.size(), "w");
        auto bs = upload(*ctx, m.scales.data(), m.scales.size() * 2, "s");
        auto bb = upload(*ctx, m.biases.data(), m.biases.size() * 2, "b");
        auto bx = upload(*ctx, x.data(), x.size() * 4, "x");
        auto bo = ctx->create_uma_buffer(rows * 4, "out");

        KernelDispatchParams p{};
        p.grid_x = rows;
        p.set({rows, in_dim, 0, 0, 0, 0, 0, 0});
        run(KernelType::GemvInt8, {bw.Get(), bs.Get(), bb.Get(), bx.Get()}, {bo.Get()}, p);

        auto got = readback(bo, rows);
        auto want = m.matvec(x);
        report("GemvInt8  (128 router rows x 2816)", compare(got, want), got, want, ok);
        dump_blob.insert(dump_blob.end(), got.begin(), got.end());
    }

    // --- GeGLU -------------------------------------------------------------------
    {
        const uint32_t n = 2112;    // dense MLP width
        std::vector<float> g(n), u(n);
        for (auto& v : g) v = xf(rng);
        for (auto& v : u) v = xf(rng);

        auto bg = upload(*ctx, g.data(), n * 4, "g");
        auto bu = upload(*ctx, u.data(), n * 4, "u");
        auto bo = ctx->create_uma_buffer(n * 4, "out");

        KernelDispatchParams p{};
        p.grid_x = (n + 255) / 256;
        p.set({n, 0, 0, 0});
        run(KernelType::GeGLU, {bg.Get(), bu.Get()}, {bo.Get()}, p);

        auto got = readback(bo, n);
        std::vector<float> want(n);
        for (uint32_t i = 0; i < n; ++i) {
            const float x0 = g[i];
            const float t = 0.7978845608028654f * (x0 + 0.044715f * x0 * x0 * x0);
            want[i] = 0.5f * x0 * (1.0f + std::tanh(t)) * u[i];
        }
        report("GeGLU     (gelu_pytorch_tanh)", compare(got, want), got, want, ok);
    }

    // --- Softcap -----------------------------------------------------------------
    {
        const uint32_t n = 4096;
        const float cap = 30.0f;
        std::vector<float> z(n);
        std::uniform_real_distribution<float> big(-200.0f, 200.0f);
        for (auto& v : z) v = big(rng);

        auto bz = upload(*ctx, z.data(), n * 4, "z");
        auto bo = ctx->create_uma_buffer(n * 4, "out");
        uint32_t cbits;
        std::memcpy(&cbits, &cap, 4);

        KernelDispatchParams p{};
        p.grid_x = (n + 255) / 256;
        p.set({n, 0, 0, cbits});
        run(KernelType::Softcap, {bz.Get()}, {bo.Get()}, p);

        auto got = readback(bo, n);
        std::vector<float> want(n);
        for (uint32_t i = 0; i < n; ++i) want[i] = cap * std::tanh(z[i] / cap);
        report("Softcap   (30*tanh(z/30))", compare(got, want), got, want, ok);
    }

    // --- MulBF16: out = in * bf16(w) * scale ---------------------------------------
    //
    // Builds the router's input, rmsnorm_no_scale(hidden) * router.scale / sqrt(D). A wrong
    // scale here does not break anything visibly -- it just biases which experts get routed,
    // so the model keeps producing fluent text chosen by a slightly wrong router.
    {
        const uint32_t n = 2816;
        const float scale = 0.0188478f;                 // router.scale / sqrt(2816)
        std::vector<float> x(n);
        std::vector<uint16_t> w(n);
        for (auto& v : x) v = xf(rng) * 2.0f;
        for (auto& v : w) v = bf16_bits(xf(rng) * 0.3f);

        auto bx = upload(*ctx, x.data(), n * 4, "x");
        auto bw = upload(*ctx, w.data(), n * 2, "w");
        auto bo = ctx->create_uma_buffer(n * 4, "out");
        uint32_t sbits;
        std::memcpy(&sbits, &scale, 4);

        KernelDispatchParams p{};
        p.grid_x = (n + 255) / 256;
        p.set({n, 0, 0, 0, sbits});                     // gp0 = offsets, gp1.x = scale
        run(KernelType::MulBF16, {bx.Get(), bw.Get()}, {bo.Get()}, p);

        auto got = readback(bo, n);
        std::vector<float> want(n);
        for (uint32_t i = 0; i < n; ++i) want[i] = x[i] * bf16_of(w[i]) * scale;
        report("MulBF16   (in * bf16(w) * scale)", compare(got, want), got, want, ok);
    }

    // --- ScaleAccum: weighted reduction of the routed experts into h2 ---------------
    //
    // The first expert overwrites (which is also what zeroes the buffer) and the rest
    // accumulate. Getting the overwrite/accumulate flag wrong leaves the previous token's
    // h2 underneath, which again stays fluent and is simply wrong.
    {
        const uint32_t n = 2816, K = 8;
        std::vector<std::vector<float>> outs(K, std::vector<float>(n));
        for (auto& o : outs) for (auto& v : o) v = xf(rng);
        std::vector<float> weights(K);
        for (auto& v : weights) v = 0.05f + std::fabs(xf(rng)) * 0.2f;

        auto bw = upload(*ctx, weights.data(), K * 4, "rw");
        auto bo = ctx->create_uma_buffer(n * 4, "h2");
        // Pre-dirty the destination so a missing overwrite on the first write is visible.
        {
            std::vector<float> junk(n, 12345.0f);
            void* p = nullptr;
            D3D12_RANGE r{0, 0};
            bo->Map(0, &r, &p);
            std::memcpy(p, junk.data(), n * 4);
            bo->Unmap(0, nullptr);
        }

        std::vector<ComPtr<ID3D12Resource>> ins;
        for (uint32_t k = 0; k < K; ++k) {
            ins.push_back(upload(*ctx, outs[k].data(), n * 4, "e"));
            KernelDispatchParams p{};
            p.grid_x = (n + 255) / 256;
            p.set({n, 0, 0, 0, (k == 0) ? 0u : 1u, k});   // gp1 = (accumulate, weight_index)
            run(KernelType::ScaleAccum, {ins.back().Get(), bw.Get()}, {bo.Get()}, p);
        }

        auto got = readback(bo, n);
        std::vector<float> want(n, 0.0f);
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t i = 0; i < n; ++i) want[i] += weights[k] * outs[k][i];
        }
        report("ScaleAccum (8 experts, first overwrites)", compare(got, want), got, want, ok);
    }

    // --- RouterTopK: selection, top-K-only softmax, per-expert scale ---------------
    {
        const uint32_t NE = 128, K = 8;
        std::vector<float> logits(NE);
        for (auto& v : logits) v = xf(rng) * 4.0f;
        // Force a tie so the lower-index rule is actually exercised.
        logits[11] = logits[70] = 9.5f;
        std::vector<uint16_t> pes(NE);
        for (auto& v : pes) v = bf16_bits(0.5f + std::fabs(xf(rng)));

        auto bl = upload(*ctx, logits.data(), NE * 4, "logits");
        auto bp = upload(*ctx, pes.data(), NE * 2, "pes");
        auto bi = ctx->create_uma_buffer(K * 4, "idx");
        auto bw = ctx->create_uma_buffer(K * 4, "wgt");

        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({NE, K, 0, 0, 0, 0, 0, 0});
        run(KernelType::RouterTopK, {bl.Get(), bp.Get()}, {bi.Get(), bw.Get()}, p);

        std::vector<uint32_t> gidx(K);
        {
            void* q = nullptr;
            D3D12_RANGE r{0, K * 4};
            bi->Map(0, &r, &q);
            std::memcpy(gidx.data(), q, K * 4);
            bi->Unmap(0, nullptr);
        }
        auto gwgt = readback(bw, K);

        // CPU reference selection: descending score, ties to the lower index.
        std::vector<int> order(NE);
        for (uint32_t i = 0; i < NE; ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (logits[a] != logits[b]) return logits[a] > logits[b];
            return a < b;
        });
        bool idx_ok = true;
        for (uint32_t i = 0; i < K; ++i) {
            if (gidx[i] != static_cast<uint32_t>(order[i])) idx_ok = false;
        }
        std::cout << (idx_ok ? "  [PASS] " : "  [FAIL] ")
                  << "RouterTopK (indices, ties to lower expert)";
        if (!idx_ok) {
            std::cout << "\n           got ";
            for (auto v : gidx) std::cout << v << " ";
            std::cout << "\n           want ";
            for (uint32_t i = 0; i < K; ++i) std::cout << order[i] << " ";
            ok = false;
        }
        std::cout << "\n";

        // Softmax must be over the top-K ONLY, then scaled per expert.
        float best = logits[order[0]], sum = 0.0f;
        std::vector<float> want(K);
        for (uint32_t i = 0; i < K; ++i) { want[i] = std::exp(logits[order[i]] - best); sum += want[i]; }
        for (uint32_t i = 0; i < K; ++i) want[i] = want[i] / sum * bf16_of(pes[order[i]]);
        report("RouterTopK (top-K-only softmax x scale)", compare(gwgt, want), gwgt, want, ok);
    }

    // --- PostAttn: residual + the three distinct pre-FFN views --------------------
    {
        const uint32_t D = 2816;
        const float eps = 1e-6f;
        std::vector<float> attn(D), hidden(D);
        for (auto& v : attn) v = xf(rng);
        for (auto& v : hidden) v = xf(rng);
        std::vector<uint16_t> w_post(D), w_pre(D), w_pre2(D);
        for (auto& v : w_post) v = bf16_bits(xf(rng) * 0.3f);
        for (auto& v : w_pre)  v = bf16_bits(xf(rng) * 0.3f);
        for (auto& v : w_pre2) v = bf16_bits(xf(rng) * 0.3f);

        auto ba = upload(*ctx, attn.data(), D * 4, "attn");
        auto bh = upload(*ctx, hidden.data(), D * 4, "hidden");
        auto bwp = upload(*ctx, w_post.data(), D * 2, "wpost");
        auto bw1 = upload(*ctx, w_pre.data(), D * 2, "wpre");
        auto bw2 = upload(*ctx, w_pre2.data(), D * 2, "wpre2");
        auto o_hidden = ctx->create_uma_buffer(D * 4, "o_hidden");
        auto o_dense  = ctx->create_uma_buffer(D * 4, "o_dense");
        auto o_routed = ctx->create_uma_buffer(D * 4, "o_routed");
        auto o_router = ctx->create_uma_buffer(D * 4, "o_router");

        uint32_t ebits;
        std::memcpy(&ebits, &eps, 4);
        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({D, 0, 0, 0, ebits, 0, 0, 0, 0, 0, 0, 0});
        run(KernelType::PostAttn, {ba.Get(), bh.Get(), bwp.Get(), bw1.Get(), bw2.Get()},
            {o_hidden.Get(), o_dense.Get(), o_routed.Get(), o_router.Get()}, p);

        // CPU reference, identical to cpu_reference.cpp's post-attention block.
        auto rms_inv = [&](const std::vector<float>& v) {
            double a = 0.0;
            for (float x0 : v) a += static_cast<double>(x0) * x0;
            return 1.0f / std::sqrt(static_cast<float>(a / v.size()) + eps);
        };
        const float ai = rms_inv(attn);
        std::vector<float> want_hidden(D);
        for (uint32_t i = 0; i < D; ++i)
            want_hidden[i] = hidden[i] + attn[i] * ai * bf16_of(w_post[i]);
        const float hi = rms_inv(want_hidden);
        std::vector<float> want_dense(D), want_routed(D), want_router(D);
        for (uint32_t i = 0; i < D; ++i) {
            const float h = want_hidden[i] * hi;
            want_dense[i]  = h * bf16_of(w_pre[i]);
            want_routed[i] = h * bf16_of(w_pre2[i]);
            want_router[i] = h;
        }

        auto g_hidden = readback(o_hidden, D);
        auto g_dense  = readback(o_dense, D);
        auto g_routed = readback(o_routed, D);
        auto g_router = readback(o_router, D);
        report("PostAttn  (residual)",  compare(g_hidden, want_hidden), g_hidden, want_hidden, ok);
        report("PostAttn  (dense_x)",   compare(g_dense, want_dense),   g_dense, want_dense, ok);
        report("PostAttn  (routed_x)",  compare(g_routed, want_routed), g_routed, want_routed, ok);
        report("PostAttn  (router_x, unscaled)", compare(g_router, want_router), g_router, want_router, ok);
    }

    // --- LayerTail ----------------------------------------------------------------
    {
        const uint32_t D = 2816;
        const float eps = 1e-6f, layer_scalar = 1.37f;
        std::vector<float> h2(D), h1(D), hidden(D);
        for (auto& v : h2) v = xf(rng);
        for (auto& v : h1) v = xf(rng);
        for (auto& v : hidden) v = xf(rng);
        std::vector<uint16_t> w_pf2(D), w_pf(D);
        for (auto& v : w_pf2) v = bf16_bits(xf(rng) * 0.3f);
        for (auto& v : w_pf)  v = bf16_bits(xf(rng) * 0.3f);

        auto b2 = upload(*ctx, h2.data(), D * 4, "h2");
        auto b1 = upload(*ctx, h1.data(), D * 4, "h1");
        auto bh = upload(*ctx, hidden.data(), D * 4, "hidden");
        auto bw2 = upload(*ctx, w_pf2.data(), D * 2, "wpf2");
        auto bw1 = upload(*ctx, w_pf.data(), D * 2, "wpf");
        auto bo = ctx->create_uma_buffer(D * 4, "out");
        auto bscratch = ctx->create_uma_buffer(D * 4, "scratch");

        uint32_t ebits, sbits;
        std::memcpy(&ebits, &eps, 4);
        std::memcpy(&sbits, &layer_scalar, 4);
        KernelDispatchParams p{};
        p.grid_x = 1;
        p.set({D, 0, 0, 0, ebits, 0, 0, 0, sbits, 0, 0, 0});
        run(KernelType::LayerTail, {b2.Get(), b1.Get(), bh.Get(), bw2.Get(), bw1.Get()},
            {bo.Get(), bscratch.Get()}, p);

        auto rms_inv = [&](const std::vector<float>& v) {
            double a = 0.0;
            for (float x0 : v) a += static_cast<double>(x0) * x0;
            return 1.0f / std::sqrt(static_cast<float>(a / v.size()) + eps);
        };
        const float i2 = rms_inv(h2);
        std::vector<float> h12(D);
        for (uint32_t i = 0; i < D; ++i) h12[i] = h1[i] + h2[i] * i2 * bf16_of(w_pf2[i]);
        const float i12 = rms_inv(h12);
        std::vector<float> want(D);
        for (uint32_t i = 0; i < D; ++i)
            want[i] = (hidden[i] + h12[i] * i12 * bf16_of(w_pf[i])) * layer_scalar;

        auto got = readback(bo, D);
        report("LayerTail (sandwich norms x layer_scalar)", compare(got, want), got, want, ok);
    }

    // --- QKVEpilogue: per-head norm, then partial NeoX rotary ---------------------
    {
        // Full-attention geometry: head_dim 512, and only 64 of 256 pairs rotated.
        const uint32_t head_dim = 512, heads = 2, rotated = 64;
        const float eps = 1e-6f, theta = 1000000.0f;
        const uint32_t position = 7;
        std::vector<float> vec(heads * head_dim);
        for (auto& v : vec) v = xf(rng);
        std::vector<uint16_t> w(head_dim);
        for (auto& v : w) v = bf16_bits(0.5f + xf(rng) * 0.2f);

        auto bv = upload(*ctx, vec.data(), vec.size() * 4, "vec");
        auto bw = upload(*ctx, w.data(), w.size() * 2, "w");
        auto bo = ctx->create_uma_buffer(vec.size() * 4, "out");

        uint32_t ebits, tbits;
        std::memcpy(&ebits, &eps, 4);
        std::memcpy(&tbits, &theta, 4);
        KernelDispatchParams p{};
        p.grid_x = heads;
        p.set({head_dim, heads, 0, 0, ebits, 0, 1, 1, rotated, position, tbits, 0});
        run(KernelType::QKVEpilogue, {bv.Get(), bw.Get()}, {bo.Get()}, p);

        std::vector<float> want(vec.size());
        for (uint32_t h = 0; h < heads; ++h) {
            const float* src = vec.data() + h * head_dim;
            float* dst = want.data() + h * head_dim;
            double a = 0.0;
            for (uint32_t i = 0; i < head_dim; ++i) a += static_cast<double>(src[i]) * src[i];
            const float inv = 1.0f / std::sqrt(static_cast<float>(a / head_dim) + eps);
            for (uint32_t i = 0; i < head_dim; ++i) dst[i] = src[i] * inv * bf16_of(w[i]);
            const uint32_t half = head_dim / 2;
            for (uint32_t i = 0; i < rotated; ++i) {
                const float freq = std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
                const float ang = static_cast<float>(position) * freq;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = dst[i], x1 = dst[i + half];
                dst[i] = x0 * c - x1 * s;
                dst[i + half] = x0 * s + x1 * c;
            }
        }
        auto got = readback(bo, vec.size());
        report("QKVEpilogue (partial rotary, 64/256 pairs)", compare(got, want), got, want, ok);
    }

    // --- Attention: GQA with a sliding window -------------------------------------
    {
        const uint32_t q_heads = 16, kv_heads = 8, head_dim = 256;
        const uint32_t n_pos = 300, window = 128;
        const uint32_t first = n_pos - window;   // sliding-window layers mask the tail

        std::vector<float> q(q_heads * head_dim);
        std::vector<float> kc(static_cast<size_t>(n_pos) * kv_heads * head_dim);
        std::vector<float> vc(kc.size());
        for (auto& v : q) v = xf(rng) * 0.1f;
        for (auto& v : kc) v = xf(rng) * 0.1f;
        for (auto& v : vc) v = xf(rng);

        auto bq = upload(*ctx, q.data(), q.size() * 4, "q");
        auto bk = upload(*ctx, kc.data(), kc.size() * 4, "k");
        auto bv = upload(*ctx, vc.data(), vc.size() * 4, "v");
        auto bo = ctx->create_uma_buffer(q.size() * 4, "ctx");

        KernelDispatchParams p{};
        p.grid_x = q_heads;
        // capacity == n_pos makes the ring modulo an identity, like a full-attention layer.
        p.set({q_heads, kv_heads, head_dim, n_pos, first, 0, 0, 0, 0, n_pos, 0, 0});
        run(KernelType::Attention, {bq.Get(), bk.Get(), bv.Get()}, {bo.Get()}, p);

        std::vector<float> want(q.size(), 0.0f);
        const uint32_t gqa = q_heads / kv_heads;
        for (uint32_t h = 0; h < q_heads; ++h) {
            const uint32_t kvh = h / gqa;
            std::vector<float> sc(n_pos, 0.0f);
            float best = -1e30f;
            for (uint32_t t = first; t < n_pos; ++t) {
                const float* kt = kc.data() + (static_cast<size_t>(t) * kv_heads + kvh) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) dot += q[h * head_dim + d] * kt[d];
                sc[t] = dot;                 // scale 1.0, not 1/sqrt(head_dim)
                best = std::max(best, dot);
            }
            float sum = 0.0f;
            for (uint32_t t = first; t < n_pos; ++t) { sc[t] = std::exp(sc[t] - best); sum += sc[t]; }
            for (uint32_t t = first; t < n_pos; ++t) {
                const float wt = sc[t] / sum;
                const float* vt = vc.data() + (static_cast<size_t>(t) * kv_heads + kvh) * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) want[h * head_dim + d] += wt * vt[d];
            }
        }
        auto got = readback(bo, q.size());
        report("Attention (GQA 16/8, window 128)", compare(got, want), got, want, ok);
    }

    // --- LMHeadGreedy + ArgmaxReduce: must pick the same row a full GEMV+argmax would ------
    {
        // Small vocabulary, but the same shape of work: one wave per row, reduce to a token.
        const uint32_t rows = 4096, in_dim = 2816;
        QuantMatrix m;
        m.build(rows, in_dim, 4, rng);
        std::vector<float> x(in_dim);
        for (auto& v : x) v = xf(rng);

        auto bw = upload(*ctx, m.packed.data(), m.packed.size(), "w");
        auto bs = upload(*ctx, m.scales.data(), m.scales.size() * 2, "s");
        auto bb = upload(*ctx, m.biases.data(), m.biases.size() * 2, "b");
        auto bx = upload(*ctx, x.data(), x.size() * 4, "x");
        auto bsum = ctx->create_uma_buffer(static_cast<uint64_t>(rows) * 8, "summaries");
        auto btok = ctx->create_uma_buffer(4, "token");

        constexpr uint32_t LMH_THREADS = 512;
        constexpr uint32_t MIN_ROWS_PER_GROUP = LMH_THREADS / 64;
        const uint32_t groups = (rows + MIN_ROWS_PER_GROUP - 1) / MIN_ROWS_PER_GROUP;

        {
            KernelDispatchParams p{};
            p.grid_x = groups;
            p.set({rows, in_dim, 0, 0, 0, 0, 0, 0});
            run(KernelType::LMHeadGreedy, {bw.Get(), bs.Get(), bb.Get(), bx.Get()},
                {bsum.Get()}, p);
        }
        {
            KernelDispatchParams p{};
            p.grid_x = 1;
            p.set({groups, 0, 0, 0});
            run(KernelType::ArgmaxReduce, {bsum.Get()}, {btok.Get()}, p);
        }

        uint32_t got_tok = 0;
        {
            void* q = nullptr;
            D3D12_RANGE r{0, 4};
            btok->Map(0, &r, &q);
            std::memcpy(&got_tok, q, 4);
            btok->Unmap(0, nullptr);
        }

        // CPU reference: full GEMV, then argmax with ties to the lowest row.
        auto logits = m.matvec(x);
        uint32_t want_tok = 0;
        for (uint32_t i = 1; i < rows; ++i) {
            if (logits[i] > logits[want_tok]) want_tok = i;
        }

        const bool tok_ok = (got_tok == want_tok);
        std::cout << (tok_ok ? "  [PASS] " : "  [FAIL] ")
                  << "LMHeadGreedy + ArgmaxReduce (fused argmax, no logits)";
        if (!tok_ok) {
            std::cout << "  got=" << got_tok << " (" << logits[got_tok] << ")"
                      << " want=" << want_tok << " (" << logits[want_tok] << ")";
            ok = false;
        }
        std::cout << "\n";
    }

    // --- Attention past the ring wrap point --------------------------------------
    // A 24-token generation never wraps, so without this the ring indexing is untested.
    // Window 128 with n_pos 300 means positions 172..299 map to slots that have already been
    // reused twice; getting the modulo wrong reads a neighbouring position's K/V and still
    // produces plausible output.
    {
        const uint32_t q_heads = 16, kv_heads = 8, head_dim = 256;
        const uint32_t n_pos = 300, window = 128;
        const uint32_t first = n_pos - window;
        const uint32_t capacity = window;          // sliding-window layer

        std::vector<float> q(q_heads * head_dim);
        // The cache only holds `capacity` rows now, not n_pos.
        std::vector<float> kc(static_cast<size_t>(capacity) * kv_heads * head_dim);
        std::vector<float> vc(kc.size());
        for (auto& v : q) v = xf(rng) * 0.1f;
        for (auto& v : kc) v = xf(rng) * 0.1f;
        for (auto& v : vc) v = xf(rng);

        auto bq = upload(*ctx, q.data(), q.size() * 4, "q");
        auto bk = upload(*ctx, kc.data(), kc.size() * 4, "k");
        auto bv = upload(*ctx, vc.data(), vc.size() * 4, "v");
        auto bo = ctx->create_uma_buffer(q.size() * 4, "ctx");

        KernelDispatchParams p{};
        p.grid_x = q_heads;
        p.set({q_heads, kv_heads, head_dim, n_pos, first, 0, 0, 0, 0, capacity, 0, 0});
        run(KernelType::Attention, {bq.Get(), bk.Get(), bv.Get()}, {bo.Get()}, p);

        std::vector<float> want(q.size(), 0.0f);
        const uint32_t gqa = q_heads / kv_heads;
        for (uint32_t h = 0; h < q_heads; ++h) {
            const uint32_t kvh = h / gqa;
            std::vector<float> sc(n_pos, 0.0f);
            float best = -1e30f;
            for (uint32_t t = first; t < n_pos; ++t) {
                const size_t slot = t % capacity;    // the ring mapping under test
                const float* kt = kc.data() + (slot * kv_heads + kvh) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) dot += q[h * head_dim + d] * kt[d];
                sc[t] = dot;
                best = std::max(best, dot);
            }
            float sum = 0.0f;
            for (uint32_t t = first; t < n_pos; ++t) { sc[t] = std::exp(sc[t] - best); sum += sc[t]; }
            for (uint32_t t = first; t < n_pos; ++t) {
                const float wt = sc[t] / sum;
                const size_t slot = t % capacity;
                const float* vt = vc.data() + (slot * kv_heads + kvh) * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) want[h * head_dim + d] += wt * vt[d];
            }
        }
        auto got = readback(bo, q.size());
        report("Attention (ring wrap: window 128, n_pos 300)", compare(got, want), got, want, ok);
    }

    if (!dump_path.empty()) {
        std::ofstream f(dump_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(dump_blob.data()),
                static_cast<std::streamsize>(dump_blob.size() * sizeof(float)));
        std::cout << "  [DUMP] " << dump_blob.size() << " GEMV floats -> " << dump_path << "\n";
    }

    if (!ok) {
        std::cout << "[TEST FAILED] at least one kernel diverges from the CPU reference\n";
        return 1;
    }
    std::cout << "[TEST SUCCESS] all implemented kernels match the CPU reference\n";
    return 0;
}
