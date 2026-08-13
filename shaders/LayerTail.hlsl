// The Gemma 4 decoder-layer tail. Mirrors fused_layer_tail (fused.metal:272-375):
//
//   tmp    = rmsnorm(h2) * w_postffn2
//   h12    = h1 + tmp
//   tmp2   = rmsnorm(h12) * w_postffn
//   hidden = (hidden + tmp2) * layer_scalar
//
// Note the per-layer learned scalar multiplying the *whole* residual stream at the end --
// a BF16 [1] tensor that is easy to overlook and silently rescales every layer if dropped.
//
//   t0 h2   t1 h1   t2 hidden(in)   t3 w_postffn2   t4 w_postffn
//   u0 hidden(out)   u1 scratch (>= D floats)
//
//   gp0 = (D, h2_off, h1_off, hidden_off)
//   gp1 = (eps_bits, w_postffn2_off, w_postffn_off, out_off)
//   gp2 = (layer_scalar_bits, scratch_off, 0, 0)

#include "Common.hlsli"

#define LT_THREADS 256

groupshared float s_partial[LT_THREADS / 4];

float group_rms_inv(float acc, uint tid, uint n, float eps) {
    const uint lane_count = WaveGetLaneCount();
    const uint num_waves = LT_THREADS / lane_count;
    acc = WaveActiveSum(acc);
    if (WaveIsFirstLane()) s_partial[tid / lane_count] = acc;
    AllMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float total = 0.0f;
        for (uint w = 0; w < num_waves; ++w) total += s_partial[w];
        s_partial[0] = rsqrt(total / float(n) + eps);
    }
    AllMemoryBarrierWithGroupSync();
    return s_partial[0];
}

[numthreads(LT_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint D           = gp0.x;
    const uint h2_off      = gp0.y;
    const uint h1_off      = gp0.z;
    const uint hidden_off  = gp0.w;
    const float eps        = asfloat(gp1.x);
    const uint w_pf2_off   = gp1.y;
    const uint w_pf_off    = gp1.z;
    const uint out_off     = gp1.w;
    const float layer_scl  = asfloat(gp2.x);
    const uint scratch_off = gp2.y;

    // 1. rmsnorm(h2) * w_postffn2, then + h1 -> scratch (h12)
    float acc = 0.0f;
    for (uint i = tid; i < D; i += LT_THREADS) {
        const float v = f32_load(g_in0, h2_off + i * 4);
        acc += v * v;
    }
    const float inv_h2 = group_rms_inv(acc, tid, D, eps);

    acc = 0.0f;
    for (uint i = tid; i < D; i += LT_THREADS) {
        const float t = f32_load(g_in0, h2_off + i * 4) * inv_h2 *
                        bf16_load(g_in3, w_pf2_off + i * 2);
        const float h12 = f32_load(g_in1, h1_off + i * 4) + t;
        f32_store(g_out1, scratch_off + i * 4, h12);
        acc += h12 * h12;
    }
    AllMemoryBarrierWithGroupSync();
    const float inv_h12 = group_rms_inv(acc, tid, D, eps);

    // 2. hidden = (hidden + rmsnorm(h12) * w_postffn) * layer_scalar
    for (uint i = tid; i < D; i += LT_THREADS) {
        const float t = f32_load(g_out1, scratch_off + i * 4) * inv_h12 *
                        bf16_load(g_in4, w_pf_off + i * 2);
        const float h = f32_load(g_in2, hidden_off + i * 4) + t;
        f32_store(g_out0, out_off + i * 4, h * layer_scl);
    }
}
