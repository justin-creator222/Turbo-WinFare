#pragma once

// Token sampling, shared by the GPU path (src/runner.cpp) and the CPU reference
// (src/cpu_reference.cpp).
//
// This lives in one place because the two paths are diffed against each other. When the
// truncation logic was inlined in the CPU reference only, the GPU path had no sampling at
// all -- `temperature > 0` took the slower materializing-logits route and then argmaxed
// anyway, so every "sampled" GPU run was silently greedy.

#include "gturbo/format.hpp"

#include <cstdint>
#include <vector>

namespace gturbo {

// Defaults match the reference (README.md:116-117 / Sampler.swift:9-35): temperature 0.2,
// top-K 64, top-P 0.95. Here temperature defaults to 0 (greedy) because the callers that
// construct this directly are tests and the CLI; GenerationOptions carries the 0.2 default.
struct SamplingParams {
    float temperature{0.0f};        // 0 = greedy
    float top_p{1.0f};              // 1 = no nucleus truncation
    int   top_k{0};                 // 0 = no top-k truncation
    float repetition_penalty{1.0f}; // 1 = disabled
    bool  has_seed{false};
    uint64_t seed{0};

    bool is_greedy() const { return temperature <= 0.0f && repetition_penalty == 1.0f; }
};

// Throws GTurboFormatError on anything out of range.
//
// Mirrors Sampler.swift:37-58, including the rule that is easy to mistake for an oversight:
// top_p < 1 with temperature > 0 REQUIRES a top_k in 1..256. Full-vocabulary nucleus
// sampling is deliberately unimplemented upstream, and sample_token's partial-selection
// optimization depends on that bound holding.
void validate_sampling(const SamplingParams& p);

// Per-position PRNG seeding, matching Sampler.swift:198-215. Using one generator seeded once
// makes a run reproducible only if every draw happens in the same order; seeding per position
// makes token N reproducible on its own, which is what the reference guarantees.
uint64_t splitmix64(uint64_t x);
uint64_t seed_for(const SamplingParams& p, int position);   // never returns 0

// Divides logits of already-seen tokens (HF convention), in place.
//
// `softcap` is the value the logits have ALREADY been capped at -- Gemma 4 uses 30.0. The
// penalty must be applied to the capped value; applied to raw logits it silently does almost
// nothing, because real Gemma 4 logits reach the hundreds and are deep in tanh saturation
// where dividing by 1.1 moves the capped output by a rounding error.
//
// The reference penalizes the pre-softcap buffer and inverts through atanh
// (Sampler.swift:166-191) only because its softcap runs afterwards in a kernel it will not
// duplicate on the host. Here the CPU sees the buffer after the Softcap dispatch, so the
// penalty applies directly and the transcendentals are unnecessary.
void apply_repetition_penalty(float* logits, uint32_t vocab,
                              const std::vector<uint32_t>& history,
                              float penalty, float softcap);

// Truncation order is Top-P over the FULL distribution, then Top-K over the survivors, then
// temperature on the final draw -- the mlx-lm order (Sampler.swift:80-85). The common
// HuggingFace order (Top-K first) selects a different candidate set and diverges from the
// reference's output.
//
// Greedy (temperature <= 0) returns argmax, ties going to the lower index.
uint32_t sample_token(const float* logits, uint32_t vocab,
                      const SamplingParams& p, uint64_t seed);

} // namespace gturbo
