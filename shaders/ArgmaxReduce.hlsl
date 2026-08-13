// Reduces LMHeadGreedy's per-threadgroup (value, index) summaries to a single token ID.
// Pass 2 of 2.
//
// One threadgroup walks all the summaries. At 262,144 rows over 512-thread groups there are
// at most 32,768 of them, so this is a few hundred microseconds and the only thing the CPU
// ever reads back is 4 bytes.
//
//   t0 summaries (float2 per entry)
//   u0 token id (uint)
//
//   gp0 = (count, in_byte_off, out_byte_off, 0)

#include "Common.hlsli"

#define AR_THREADS 256
#define AR_MAX_WAVES (AR_THREADS / 32)

groupshared float s_val[AR_MAX_WAVES];
groupshared uint  s_idx[AR_MAX_WAVES];

[numthreads(AR_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint count   = gp0.x;
    const uint in_off  = gp0.y;
    const uint out_off = gp0.z;

    const uint lane_count = WaveGetLaneCount();
    const uint waves      = AR_THREADS / lane_count;
    const uint wave_id    = tid / lane_count;
    const uint lane       = tid % lane_count;

    float best = -1e30f;
    uint  best_row = 0xFFFFFFFFu;
    for (uint i = tid; i < count; i += AR_THREADS) {
        const float v = f32_load(g_in0, in_off + i * 8u);
        const uint  r = asuint(f32_load(g_in0, in_off + i * 8u + 4u));
        // Ties go to the lower row index.
        if (v > best || (v == best && r < best_row)) { best = v; best_row = r; }
    }

    // Wave-level reduction first, then across waves through groupshared.
    const float wave_best = WaveActiveMax(best);
    // Among lanes holding the winning value, keep the smallest row index.
    const uint wave_row = WaveActiveMin(best == wave_best ? best_row : 0xFFFFFFFFu);
    if (lane == 0) {
        s_val[wave_id] = wave_best;
        s_idx[wave_id] = wave_row;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float bv = s_val[0];
        uint  bi = s_idx[0];
        for (uint w = 1; w < waves; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_idx[w] < bi)) {
                bv = s_val[w];
                bi = s_idx[w];
            }
        }
        g_out0.Store(out_off, bi);
    }
}
