// Affine INT8 matrix-vector product. Used only by the routers, which the checkpoint
// quantizes to 8 bits while everything else is 4-bit (config.json: the per-layer
// "router.proj" overrides carry bits: 8).
//
//   t0 weights   t1 scales   t2 biases   t3 x
//   u0 out (FP32)
//
//   gp0 = (rows, in_dim, w_byte_off, s_byte_off)
//   gp1 = (b_byte_off, x_byte_off, out_byte_off, row_base)

#include "Common.hlsli"

[numthreads(GEMV_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint rows     = gp0.x;
    const uint in_dim   = gp0.y;
    const uint w_off    = gp0.z;
    const uint s_off    = gp0.w;
    const uint b_off    = gp1.x;
    const uint x_off    = gp1.y;
    const uint out_off  = gp1.z;
    const uint row_base = gp1.w;

    const uint row = gid.x + row_base;
    if (row >= rows) return;

    const float partial = gemv_int8_row_lane(g_in0, w_off, g_in1, s_off, g_in2, b_off,
                                             g_in3, x_off, row, in_dim, lane);
    const float total = WaveActiveSum(partial);
    if (lane == 0) {
        f32_store(g_out0, out_off + row * 4, total);
    }
}
