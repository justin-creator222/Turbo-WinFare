// Top-K expert selection, then softmax over the top-K ONLY, scaled per expert.
// Mirrors router_topk_select_k8 (moe.metal:135-185).
//
// Two details that matter:
//   * The softmax denominator is the sum over the K selected experts, not all 128. Using
//     the full set changes every routing weight.
//   * Ties go to the LOWER expert index, so routing is deterministic across runs and
//     matches the CPU reference exactly.
//
// Single-threaded: K is 8 and the expert count is 128, so a parallel selection would cost
// more in synchronization than it saves.
//
//   t0 logits (FP32, [num_experts])   t1 per_expert_scale (BF16, [num_experts])
//   u0 indices (uint, [K])            u1 weights (FP32, [K])
//
//   gp0 = (num_experts, K, logits_off, scale_off)
//   gp1 = (idx_off, wgt_off, 0, 0)

#include "Common.hlsli"

#define MAX_TOPK 8

[numthreads(1, 1, 1)]
void main() {
    const uint num_experts = gp0.x;
    const uint K           = min(gp0.y, MAX_TOPK);
    const uint logits_off  = gp0.z;
    const uint scale_off   = gp0.w;
    const uint idx_off     = gp1.x;
    const uint wgt_off     = gp1.y;

    uint  top_idx[MAX_TOPK];
    float top_score[MAX_TOPK];
    for (uint i = 0; i < MAX_TOPK; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -1e30f;
    }

    for (uint e = 0; e < num_experts; ++e) {
        const float s = f32_load(g_in0, logits_off + e * 4);
        if (s <= top_score[K - 1]) continue;
        uint pos = MAX_TOPK;
        for (uint i = 0; i < K; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) { pos = i; break; }
        }
        if (pos >= K) continue;
        for (uint j = K - 1; j > pos; --j) {
            top_idx[j] = top_idx[j - 1];
            top_score[j] = top_score[j - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float best = top_score[0];
    float sum_exp = 0.0f;
    float exps[MAX_TOPK];
    for (uint i = 0; i < K; ++i) {
        exps[i] = exp(top_score[i] - best);
        sum_exp += exps[i];
    }
    for (uint i = 0; i < K; ++i) {
        const uint e = top_idx[i];
        g_out0.Store(idx_off + i * 4, e);
        f32_store(g_out1, wgt_off + i * 4,
                  (exps[i] / sum_exp) * bf16_load(g_in1, scale_off + e * 2));
    }
}
