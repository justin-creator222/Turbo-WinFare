// Post-attention residual, then the three distinct pre-FFN views.
// Mirrors fused_post_attn_setup (fused.metal:191-269) exactly:
//
//   attn_norm = rmsnorm(attn) * w_post_attn
//   hidden   += attn_norm
//   h         = rmsnorm_no_scale(hidden)
//   dense_x   = h * w_pre_ffn      -> shared expert
//   routed_x  = h * w_pre_ffn2     -> routed experts
//   router_x  = h                  -> router (UNSCALED)
//
// The three views are genuinely different tensors. Feeding the same one to all three
// consumers is a silent correctness bug that still produces fluent text.
//
//   t0 attn   t1 hidden(in)   t2 w_post_attn   t3 w_pre_ffn   t4 w_pre_ffn2
//   u0 hidden(out)   u1 dense_x   u2 routed_x   u3 router_x
//
//   gp0 = (D, attn_off, hidden_off, out_hidden_off)
//   gp1 = (eps_bits, w_post_off, w_pre_off, w_pre2_off)
//   gp2 = (dense_off, routed_off, router_off, 0)

#include "Common.hlsli"

#define PA_THREADS 256

groupshared float s_partial[PA_THREADS / 4];

// Wave-width-agnostic sum of squares across the whole threadgroup.
float group_rms_inv(float acc, uint tid, uint n, float eps) {
    const uint lane_count = WaveGetLaneCount();
    const uint num_waves = PA_THREADS / lane_count;
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

[numthreads(PA_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint D            = gp0.x;
    const uint attn_off     = gp0.y;
    const uint hidden_off   = gp0.z;
    const uint out_hid_off  = gp0.w;
    const float eps         = asfloat(gp1.x);
    const uint w_post_off   = gp1.y;
    const uint w_pre_off    = gp1.z;
    const uint w_pre2_off   = gp1.w;
    const uint dense_off    = gp2.x;
    const uint routed_off   = gp2.y;
    const uint router_off   = gp2.z;

    // 1. RMS of the attention output.
    float acc = 0.0f;
    for (uint i = tid; i < D; i += PA_THREADS) {
        const float v = f32_load(g_in0, attn_off + i * 4);
        acc += v * v;
    }
    const float attn_inv = group_rms_inv(acc, tid, D, eps);

    // 2. Residual add, accumulating the new hidden's sum of squares as we go.
    acc = 0.0f;
    for (uint i = tid; i < D; i += PA_THREADS) {
        const float a = f32_load(g_in0, attn_off + i * 4) * attn_inv *
                        bf16_load(g_in2, w_post_off + i * 2);
        const float h = f32_load(g_in1, hidden_off + i * 4) + a;
        f32_store(g_out0, out_hid_off + i * 4, h);
        acc += h * h;
    }
    AllMemoryBarrierWithGroupSync();
    const float hidden_inv = group_rms_inv(acc, tid, D, eps);

    // 3. The three views.
    for (uint i = tid; i < D; i += PA_THREADS) {
        const float h = f32_load(g_out0, out_hid_off + i * 4) * hidden_inv;
        f32_store(g_out1, dense_off  + i * 4, h * bf16_load(g_in3, w_pre_off  + i * 2));
        f32_store(g_out2, routed_off + i * 4, h * bf16_load(g_in4, w_pre2_off + i * 2));
        f32_store(g_out3, router_off + i * 4, h);
    }
}
