// y = x / sqrt(mean(x^2) + eps), optionally scaled by a BF16 weight.
//
// The weight is applied as plain `w`, NOT `1 + w` -- the offset is already baked into the
// checkpoint. Using 1+w here produces fluent but wrong output, which is the hardest kind of
// bug to find, so it is asserted by the GPU-vs-CPU diff rather than left to inspection.
//
//   t0 x   t1 weight (BF16, optional)
//   u0 out (FP32)
//
//   gp0 = (n, x_byte_off, w_byte_off, out_byte_off)
//   gp1 = (has_weight, eps_bits, 0, 0)
//
// One threadgroup of 256 for the whole vector.

#include "Common.hlsli"

#define RMS_THREADS 256

// Sized for the narrowest possible wave (RMS_THREADS / 4). The actual wave width is queried
// at runtime -- RDNA 3 runs this threadgroup as Wave64, not Wave32, so assuming 32 here made
// each wave's total land in two slots and doubled the sum of squares. That produced output
// scaled by exactly 1/sqrt(2), which is subtle enough to look like a plausible activation.
groupshared float s_partial[RMS_THREADS / 4];

[numthreads(RMS_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint n          = gp0.x;
    const uint x_off      = gp0.y;
    const uint w_off      = gp0.z;
    const uint out_off    = gp0.w;
    const bool has_weight = (gp1.x != 0u);
    const float eps       = asfloat(gp1.y);

    float acc = 0.0f;
    for (uint i = tid; i < n; i += RMS_THREADS) {
        const float v = f32_load(g_in0, x_off + i * 4);
        acc += v * v;
    }

    const uint lane_count = WaveGetLaneCount();
    const uint num_waves  = RMS_THREADS / lane_count;
    const uint wave_id    = tid / lane_count;

    acc = WaveActiveSum(acc);
    if (WaveIsFirstLane()) s_partial[wave_id] = acc;
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float total = 0.0f;
        for (uint w = 0; w < num_waves; ++w) total += s_partial[w];
        s_partial[0] = rsqrt(total / float(n) + eps);
    }
    GroupMemoryBarrierWithGroupSync();

    const float inv = s_partial[0];
    for (uint i = tid; i < n; i += RMS_THREADS) {
        float v = f32_load(g_in0, x_off + i * 4) * inv;
        if (has_weight) v *= bf16_load(g_in1, w_off + i * 2);
        f32_store(g_out0, out_off + i * 4, v);
    }
}
