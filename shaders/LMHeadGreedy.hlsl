// Fused tied-LM-head GEMV + partial argmax. Pass 1 of 2.
//
// The greedy path never materializes the 262,144-wide logit vector. Each wave reduces one
// vocabulary row to a scalar, the threadgroup takes the max, and only (value, index) is
// written -- 8 bytes per threadgroup instead of 1 MB of logits, with no CPU readback of the
// distribution at all.
//
// Softcap is deliberately NOT applied here: 30*tanh(z/30) is strictly monotonic, so it
// cannot change which row is the maximum. The sampled path still needs real logits and
// keeps using GemvInt4 + Softcap.
//
// Ties resolve to the lowest row index, matching the CPU reference and the Metal original.
//
//   t0 weights   t1 scales   t2 biases   t3 x (normed hidden)
//   u0 summaries: float2 per threadgroup -- (best_value, asfloat(best_row))
//
//   gp0 = (rows, in_dim, w_byte_off, s_byte_off)
//   gp1 = (b_byte_off, x_byte_off, out_byte_off, row_base)

#include "Common.hlsli"

#define LMH_THREADS 512
#define LMH_MAX_WAVES (LMH_THREADS / 32)

groupshared float s_val[LMH_MAX_WAVES];
groupshared uint  s_idx[LMH_MAX_WAVES];

[numthreads(LMH_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rows     = gp0.x;
    const uint in_dim   = gp0.y;
    const uint w_off    = gp0.z;
    const uint s_off    = gp0.w;
    const uint b_off    = gp1.x;
    const uint x_off    = gp1.y;
    const uint out_off  = gp1.z;
    const uint row_base = gp1.w;

    // Wave width is queried, not assumed: RDNA 3 runs this as Wave64, so a 512-thread group
    // is 8 waves, not 16.
    const uint lane_count = WaveGetLaneCount();
    const uint waves      = LMH_THREADS / lane_count;
    const uint wave_id    = tid / lane_count;
    const uint lane       = tid % lane_count;

    const uint row = row_base + gid.x * waves + wave_id;

    float best = -1e30f;
    uint  best_row = 0;
    if (row < rows) {
        const float partial = gemv_int4_row_lane(g_in0, w_off, g_in1, s_off, g_in2, b_off,
                                                 g_in3, x_off, row, in_dim, lane);
        best = WaveActiveSum(partial);
        best_row = row;
    }

    if (lane == 0) {
        s_val[wave_id] = best;
        s_idx[wave_id] = best_row;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float bv = s_val[0];
        uint  bi = s_idx[0];
        for (uint w = 1; w < waves; ++w) {
            // Strictly greater, so an equal value keeps the earlier (lower) row.
            if (s_val[w] > bv) { bv = s_val[w]; bi = s_idx[w]; }
        }
        f32_store(g_out0, out_off + gid.x * 8u,      bv);
        f32_store(g_out0, out_off + gid.x * 8u + 4u, asfloat(bi));
    }
}
