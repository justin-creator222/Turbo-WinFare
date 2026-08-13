// Per-head Q/K/V post-processing, then NeoX rotary on Q and K only.
// Mirrors fused_qkv_epilogue (fused.metal:85-173).
//
//   Q: rmsnorm(q_head) * q_norm  -> RoPE
//   K: rmsnorm(k_head) * k_norm  -> RoPE
//   V: rmsnorm(v_head), NO scale weight, NO RoPE
//
// One threadgroup per logical head. `is_kv` selects which of the three roles this dispatch
// is handling, since Q has more heads than K/V under GQA.
//
//   t0 vec (q, k or v)   t1 norm weight (BF16; ignored when has_weight = 0)
//   u0 out
//
//   gp0 = (head_dim, num_heads, in_off, out_off)
//   gp1 = (eps_bits, w_off, has_weight, do_rope)
//   gp2 = (rotated_pairs, position, theta_bits, 0)

#include "Common.hlsli"

#define QKV_THREADS 128

groupshared float s_partial[QKV_THREADS / 4];
groupshared float s_vec[512];   // max head_dim (full-attention layers use 512)

[numthreads(QKV_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint head_dim   = gp0.x;
    const uint num_heads  = gp0.y;
    const uint in_off     = gp0.z;
    const uint out_off    = gp0.w;
    const float eps       = asfloat(gp1.x);
    const uint w_off      = gp1.y;
    const bool has_weight = (gp1.z != 0u);
    const bool do_rope    = (gp1.w != 0u);
    const uint rotated    = gp2.x;
    const float position  = float(gp2.y);
    const float theta     = asfloat(gp2.z);

    const uint head = gid.x;
    if (head >= num_heads) return;
    const uint base_in  = in_off  + head * head_dim * 4;
    const uint base_out = out_off + head * head_dim * 4;

    // --- RMSNorm over this head's slice ---
    float acc = 0.0f;
    for (uint i = tid; i < head_dim; i += QKV_THREADS) {
        const float v = f32_load(g_in0, base_in + i * 4);
        acc += v * v;
    }
    const uint lane_count = WaveGetLaneCount();
    const uint num_waves = QKV_THREADS / lane_count;
    acc = WaveActiveSum(acc);
    if (WaveIsFirstLane()) s_partial[tid / lane_count] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float total = 0.0f;
        for (uint w = 0; w < num_waves; ++w) total += s_partial[w];
        s_partial[0] = rsqrt(total / float(head_dim) + eps);
    }
    GroupMemoryBarrierWithGroupSync();
    const float inv = s_partial[0];

    for (uint i = tid; i < head_dim; i += QKV_THREADS) {
        float v = f32_load(g_in0, base_in + i * 4) * inv;
        if (has_weight) v *= bf16_load(g_in1, w_off + i * 2);
        s_vec[i] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    // --- NeoX rotary: pairs are (i, i + head_dim/2) ---
    // `rotated` is head_dim/2 on sliding-window layers but only 64 of 256 pairs on
    // full-attention layers (partial_rotary_factor 0.25). Pairs beyond it pass through.
    if (do_rope) {
        const uint half_dim = head_dim / 2;
        for (uint p = tid; p < rotated; p += QKV_THREADS) {
            const float freq = pow(theta, -float(2u * p) / float(head_dim));
            const float angle = position * freq;
            const float c = cos(angle);
            const float s = sin(angle);
            const float x0 = s_vec[p];
            const float x1 = s_vec[p + half_dim];
            s_vec[p]            = x0 * c - x1 * s;
            s_vec[p + half_dim] = x0 * s + x1 * c;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = tid; i < head_dim; i += QKV_THREADS) {
        f32_store(g_out0, base_out + i * 4, s_vec[i]);
    }
}
