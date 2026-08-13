#include "gturbo/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace gturbo {

namespace {

// Upper bound on top_k, from the reference. It is also what lets sample_token avoid sorting
// the whole 262,144-entry vocabulary: whenever top_p < 1 the survivors are guaranteed to lie
// within the global top-256, so a partial selection over that many elements is exact.
constexpr int kMaxTopK = 256;

// xorshift64*, seeded per position. Only used to draw one uniform per token.
inline uint64_t next_random(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
}

inline float next_uniform(uint64_t& state) {
    // 24 bits of mantissa is ample for choosing among at most 256 candidates.
    return static_cast<float>((next_random(state) >> 40) & 0xFFFFFF) / 16777216.0f;
}

} // namespace

void validate_sampling(const SamplingParams& p) {
    if (!std::isfinite(p.temperature) || p.temperature < 0.0f) {
        throw GTurboFormatError("temperature must be finite and >= 0 (0 = greedy)");
    }
    if (!std::isfinite(p.top_p) || p.top_p <= 0.0f || p.top_p > 1.0f) {
        throw GTurboFormatError("top_p must be in (0, 1]");
    }
    if (p.top_k < 0 || p.top_k > kMaxTopK) {
        throw GTurboFormatError("top_k must be in 0.." + std::to_string(kMaxTopK) +
                                " (0 disables it)");
    }
    if (!std::isfinite(p.repetition_penalty) || p.repetition_penalty <= 0.0f) {
        throw GTurboFormatError("repetition_penalty must be finite and > 0 (1 = disabled)");
    }
    // Deliberate, not an oversight: the reference does not implement nucleus sampling over
    // the full vocabulary, so a nucleus cut without a top-k bound is rejected rather than
    // silently approximated.
    if (p.temperature > 0.0f && p.top_p < 1.0f && p.top_k == 0) {
        throw GTurboFormatError(
            "top_p < 1 with temperature > 0 requires top_k in 1.." + std::to_string(kMaxTopK) +
            "; full-vocabulary nucleus sampling is not implemented");
    }
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t seed_for(const SamplingParams& p, int position) {
    uint64_t base = p.has_seed
        ? p.seed + static_cast<uint64_t>(position)
        // No seed: derive one from the position and a fixed odd constant so successive
        // positions still decorrelate. Callers wanting reproducibility must set has_seed.
        : (0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(position) * 0xD1B54A32D192ED03ULL));
    uint64_t s = splitmix64(base);
    // A zero state would make xorshift64* emit zeros forever.
    return s ? s : 0x9E3779B97F4A7C15ULL;
}

void apply_repetition_penalty(float* logits, uint32_t vocab,
                              const std::vector<uint32_t>& history,
                              float penalty, float softcap) {
    if (penalty == 1.0f || history.empty()) return;

    const float limit = (softcap > 0.0f) ? softcap * 0.9999f : 0.0f;

    for (uint32_t id : history) {
        if (id >= vocab) continue;              // out-of-range ids are ignored, not fatal
        float v = logits[id];
        // HF convention: a positive logit is divided, a negative one multiplied, so both move
        // toward zero rather than a penalty flipping the sign of a negative logit.
        v = (v > 0.0f) ? (v / penalty) : (v * penalty);
        if (limit > 0.0f) v = std::clamp(v, -limit, limit);
        logits[id] = v;
    }
}

uint32_t sample_token(const float* logits, uint32_t vocab,
                      const SamplingParams& p, uint64_t seed) {
    if (vocab == 0) throw GTurboFormatError("sample_token: empty vocabulary");

    // Greedy. Strictly-greater keeps ties on the lower index, matching the reference.
    if (p.temperature <= 0.0f) {
        uint32_t best = 0;
        float best_v = logits[0];
        for (uint32_t i = 1; i < vocab; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        return best;
    }

    // One pass for the max, one for the full-distribution denominator. That denominator is
    // what makes this "top-p over the full distribution": the nucleus threshold is measured
    // against every token's probability, not just the shortlist's.
    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab; ++i) max_logit = std::max(max_logit, logits[i]);

    double total = 0.0;
    for (uint32_t i = 0; i < vocab; ++i) {
        total += std::exp(static_cast<double>(logits[i]) - max_logit);
    }
    if (!(total > 0.0)) {
        throw GTurboFormatError("sample_token: logit distribution is degenerate");
    }

    // Candidate shortlist. validate_sampling guarantees top_k <= 256 whenever top_p < 1, so
    // the survivors always lie inside the global top-256 and a partial selection is exact.
    const size_t want = (p.top_k > 0)
        ? std::min<size_t>(static_cast<size_t>(p.top_k), vocab)
        : std::min<size_t>(static_cast<size_t>(kMaxTopK), vocab);

    std::vector<uint32_t> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0u);
    auto by_logit_desc = [&](uint32_t a, uint32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;                              // stable, and ties resolve to lower index
    };
    if (want < idx.size()) {
        std::nth_element(idx.begin(), idx.begin() + static_cast<ptrdiff_t>(want), idx.end(),
                         by_logit_desc);
        idx.resize(want);
    }
    std::sort(idx.begin(), idx.end(), by_logit_desc);

    // Top-P over the full distribution: accumulate the shortlist's true probabilities until
    // the nucleus mass is reached.
    size_t cut = idx.size();
    if (p.top_p < 1.0f) {
        double run = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) {
            run += std::exp(static_cast<double>(logits[idx[i]]) - max_logit) / total;
            if (run >= static_cast<double>(p.top_p)) { cut = i + 1; break; }
        }
    }
    // Then Top-K over the survivors.
    if (p.top_k > 0) cut = std::min(cut, static_cast<size_t>(p.top_k));
    cut = std::max<size_t>(1, cut);

    // Temperature applies only to the final draw over the survivors.
    std::vector<double> q(cut);
    double denom = 0.0;
    for (size_t i = 0; i < cut; ++i) {
        q[i] = std::exp((static_cast<double>(logits[idx[i]]) - max_logit) / p.temperature);
        denom += q[i];
    }

    uint64_t state = seed;
    double r = static_cast<double>(next_uniform(state)) * denom;
    double acc = 0.0;
    for (size_t i = 0; i < cut; ++i) {
        acc += q[i];
        if (acc >= r) return idx[i];
    }
    return idx[cut - 1];                            // only reachable through rounding
}

} // namespace gturbo
