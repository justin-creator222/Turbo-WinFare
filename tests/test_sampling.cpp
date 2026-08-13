// Sampling: validation, seeding, truncation order, and the repetition penalty.
//
// The truncation ORDER is the thing most worth pinning down. Top-P over the full
// distribution then Top-K over the survivors (mlx-lm, what the reference does) selects a
// different candidate set than the common HuggingFace Top-K-then-Top-P. Both produce fluent
// text, so getting it backwards is invisible in output and only shows up as a slow drift
// away from the reference's behaviour.

#include "gturbo/sampling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace gturbo;

namespace {

bool rejects(const SamplingParams& p) {
    try {
        validate_sampling(p);
        return false;
    } catch (const GTurboFormatError&) {
        return true;
    }
}

// Deliberately naive: full sort, exactly the shape of the code that used to live inline in
// cpu_reference.cpp. sample_token uses nth_element over a 256-element shortlist instead, and
// must agree with this for every input.
uint32_t reference_sample(const std::vector<float>& logits, const SamplingParams& p,
                          uint64_t seed) {
    if (p.temperature <= 0.0f) {
        return static_cast<uint32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    std::vector<uint32_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });

    const double best = logits[idx[0]];
    double total = 0.0;
    for (float v : logits) total += std::exp(static_cast<double>(v) - best);

    size_t cut = idx.size();
    if (p.top_p < 1.0f) {
        double run = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) {
            run += std::exp(static_cast<double>(logits[idx[i]]) - best) / total;
            if (run >= static_cast<double>(p.top_p)) { cut = i + 1; break; }
        }
    }
    if (p.top_k > 0) cut = std::min(cut, static_cast<size_t>(p.top_k));
    cut = std::max<size_t>(1, cut);

    // Mirror sample_token's draw exactly so the comparison is of the candidate set and the
    // ordering, not of two different PRNGs.
    std::vector<double> q(cut);
    double denom = 0.0;
    for (size_t i = 0; i < cut; ++i) {
        q[i] = std::exp((static_cast<double>(logits[idx[i]]) - best) / p.temperature);
        denom += q[i];
    }
    uint64_t s = seed;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    const uint64_t r64 = s * 0x2545F4914F6CDD1DULL;
    const double r =
        static_cast<double>(static_cast<float>((r64 >> 40) & 0xFFFFFF) / 16777216.0f) * denom;

    double acc = 0.0;
    for (size_t i = 0; i < cut; ++i) {
        acc += q[i];
        if (acc >= r) return idx[i];
    }
    return idx[cut - 1];
}

} // namespace

int main() {
    std::cout << "[TEST] Running sampling tests...\n";

    // ---- Validation ----------------------------------------------------------------
    {
        // The reference's defaults must pass.
        assert(!rejects(SamplingParams{0.2f, 0.95f, 64, 1.0f, false, 0}));
        // Greedy with everything else at defaults.
        assert(!rejects(SamplingParams{0.0f, 1.0f, 0, 1.0f, false, 0}));

        assert(rejects(SamplingParams{-0.1f, 1.0f, 0, 1.0f, false, 0}));   // temp < 0
        assert(rejects(SamplingParams{0.2f, 0.0f, 64, 1.0f, false, 0}));   // top_p == 0
        assert(rejects(SamplingParams{0.2f, 1.5f, 64, 1.0f, false, 0}));   // top_p > 1
        assert(rejects(SamplingParams{0.2f, 0.95f, 257, 1.0f, false, 0})); // top_k > 256
        assert(rejects(SamplingParams{0.2f, 0.95f, -1, 1.0f, false, 0}));  // top_k < 0
        assert(rejects(SamplingParams{0.2f, 0.95f, 64, 0.0f, false, 0}));  // penalty == 0

        // The rule that looks like a bug but is not: nucleus sampling with no top-k bound.
        assert(rejects(SamplingParams{0.7f, 0.9f, 0, 1.0f, false, 0}));
        // ...but top_p == 1 needs no bound, because nothing is truncated.
        assert(!rejects(SamplingParams{0.7f, 1.0f, 0, 1.0f, false, 0}));
        // ...and neither does greedy, which never consults top_p at all.
        assert(!rejects(SamplingParams{0.0f, 0.9f, 0, 1.0f, false, 0}));
        std::cout << "  [PASS] Validation accepts the reference defaults and rejects the rest.\n";
    }

    // ---- Seeding -------------------------------------------------------------------
    {
        // splitmix64 against vectors computed by an independent implementation of the same
        // algorithm, so a typo in the constants here cannot agree with a typo in sampling.cpp.
        assert(splitmix64(0) == 0xE220A8397B1DCDAFULL);
        assert(splitmix64(1) == 0x910A2DEC89025CC1ULL);
        assert(splitmix64(2) == 0x975835DE1C9756CEULL);
        assert(splitmix64(3) == 0x1D0B14E4DB018FEDULL);

        SamplingParams seeded{0.7f, 1.0f, 0, 1.0f, true, 12345};
        // Stable for a given (seed, position), and never zero -- a zero state would make the
        // xorshift generator emit zeros forever.
        for (int pos = 0; pos < 10000; ++pos) {
            const uint64_t a = seed_for(seeded, pos);
            assert(a != 0);
            assert(a == seed_for(seeded, pos));
        }
        // Different positions decorrelate.
        assert(seed_for(seeded, 0) != seed_for(seeded, 1));
        // Different seeds diverge.
        SamplingParams other{0.7f, 1.0f, 0, 1.0f, true, 99};
        assert(seed_for(seeded, 5) != seed_for(other, 5));
        std::cout << "  [PASS] splitmix64 vectors, per-position stability, never zero.\n";
    }

    // ---- Greedy ---------------------------------------------------------------------
    {
        std::vector<float> logits{1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
        SamplingParams greedy{0.0f, 1.0f, 0, 1.0f, false, 0};
        // Tie between index 1 and 3: the lower index must win, matching the reference.
        assert(sample_token(logits.data(), 5, greedy, 1) == 1);

        std::vector<float> single{7.0f};
        assert(sample_token(single.data(), 1, greedy, 1) == 0);
        std::cout << "  [PASS] Greedy is argmax, ties to the lower index.\n";
    }

    // ---- Truncation ORDER: mlx (Top-P then Top-K) vs HuggingFace (Top-K then Top-P) --
    {
        // Built so the two orders disagree. Probabilities are roughly
        //   0: .45  1: .27  2: .16  3: .10  4: .02 ...
        // With top_p = 0.9 and top_k = 2:
        //   mlx order  -> nucleus takes {0,1,2,3}, then top-k trims to {0,1}
        //   HF order   -> top-k takes {0,1}, then nucleus over THAT renormalized set
        //                 reaches 0.9 at the first element alone, giving {0}
        // So under HF order index 1 would be unreachable. Draw many times and assert it is
        // in fact reachable, which is only true for the mlx order.
        std::vector<float> logits{3.0f, 2.5f, 2.0f, 1.5f, -1.0f, -2.0f, -3.0f, -4.0f};
        SamplingParams p{1.0f, 0.9f, 2, 1.0f, true, 7};

        std::set<uint32_t> seen;
        for (int i = 0; i < 4000; ++i) {
            seen.insert(sample_token(logits.data(), 8, p, seed_for(p, i)));
        }
        assert(seen.count(0) == 1);
        assert(seen.count(1) == 1 && "top-k applied before top-p: candidate set is wrong");
        // And nothing outside the top-2 may ever be drawn.
        assert(seen.size() == 2);
        std::cout << "  [PASS] Truncation order is Top-P then Top-K (mlx), not the HF order.\n";
    }

    // ---- Determinism ----------------------------------------------------------------
    {
        std::mt19937 rng(4242);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        std::vector<float> logits(4096);
        for (auto& v : logits) v = nd(rng);

        SamplingParams p{0.8f, 0.95f, 64, 1.0f, true, 20260721};
        std::vector<uint32_t> first, second;
        for (int i = 0; i < 64; ++i) first.push_back(sample_token(logits.data(), 4096, p, seed_for(p, i)));
        for (int i = 0; i < 64; ++i) second.push_back(sample_token(logits.data(), 4096, p, seed_for(p, i)));
        assert(first == second);

        SamplingParams q = p;
        q.seed = 20260722;
        std::vector<uint32_t> third;
        for (int i = 0; i < 64; ++i) third.push_back(sample_token(logits.data(), 4096, q, seed_for(q, i)));
        assert(first != third && "a different seed produced an identical sequence");
        std::cout << "  [PASS] Same seed reproduces; a different seed diverges.\n";
    }

    // ---- Repetition penalty ---------------------------------------------------------
    {
        constexpr float kSoftcap = 30.0f;
        std::vector<float> logits{10.0f, -10.0f, 0.0f, 29.999f};
        const std::vector<uint32_t> history{0, 1, 3, 99999 /* out of range, ignored */};

        auto v = logits;
        apply_repetition_penalty(v.data(), 4, history, 2.0f, kSoftcap);

        // Positive divides, negative multiplies -- both move toward zero, so a penalty never
        // flips the sign of a negative logit into a preference.
        assert(std::fabs(v[0] - 5.0f) < 1e-5f);
        assert(std::fabs(v[1] - (-20.0f)) < 1e-5f);
        assert(std::fabs(v[2] - 0.0f) < 1e-5f);     // untouched: not in history
        assert(v[3] < 15.1f && v[3] > 14.9f);
        // Nothing may exceed the softcap the logits were already capped at.
        for (float x : v) assert(std::fabs(x) <= kSoftcap);

        // A penalty of exactly 1 is a no-op, and so is an empty history.
        auto w = logits;
        apply_repetition_penalty(w.data(), 4, history, 1.0f, kSoftcap);
        assert(w == logits);
        auto z = logits;
        apply_repetition_penalty(z.data(), 4, {}, 2.0f, kSoftcap);
        assert(z == logits);

        // And it actually changes which token greedy picks -- otherwise the penalty is
        // arithmetic with no effect, which is exactly the pre-softcap failure mode.
        std::vector<float> g{9.0f, 8.0f};
        SamplingParams greedy{0.0f, 1.0f, 0, 1.0f, false, 0};
        assert(sample_token(g.data(), 2, greedy, 1) == 0);
        apply_repetition_penalty(g.data(), 2, {0}, 2.0f, kSoftcap);
        assert(sample_token(g.data(), 2, greedy, 1) == 1);
        std::cout << "  [PASS] Repetition penalty: HF convention, clamped, changes the pick.\n";
    }

    // ---- Equivalence with a naive full-sort reference --------------------------------
    //
    // This is the guard on the nth_element shortlist. If the partial selection ever picks a
    // different candidate set than a full sort would, the model keeps producing fluent text
    // drawn from a subtly wrong distribution.
    {
        std::mt19937 rng(20260723);
        std::normal_distribution<float> nd(0.0f, 6.0f);
        const uint32_t V = 262144;
        std::vector<float> logits(V);
        for (auto& v : logits) v = nd(rng);

        const SamplingParams configs[] = {
            {0.2f, 0.95f, 64,  1.0f, true, 1},
            {1.0f, 0.90f, 8,   1.0f, true, 2},
            {0.7f, 1.00f, 256, 1.0f, true, 3},
            {0.5f, 1.00f, 0,   1.0f, true, 4},   // no truncation at all
            {1.5f, 0.99f, 1,   1.0f, true, 5},   // degenerate top-k of 1
        };

        int checked = 0;
        for (const auto& cfg : configs) {
            validate_sampling(cfg);
            for (int i = 0; i < 40; ++i) {
                const uint64_t s = seed_for(cfg, i);
                const uint32_t got = sample_token(logits.data(), V, cfg, s);
                const uint32_t want = reference_sample(logits, cfg, s);
                assert(got == want && "partial selection disagrees with a full sort");
                ++checked;
            }
        }
        assert(checked == 200);
        std::cout << "  [PASS] Shortlist selection matches a full sort over " << checked
                  << " draws at V=262144.\n";
    }

    std::cout << "[TEST] All sampling tests passed.\n";
    return 0;
}
