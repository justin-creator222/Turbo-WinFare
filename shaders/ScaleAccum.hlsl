// out[i] = accumulate ? (out[i] + scale * in[i]) : (scale * in[i])
//
// Reduces the eight routed-expert outputs into h2, weighted by their routing weights. The
// first expert overwrites (which also zeroes the buffer) and the rest accumulate, so no
// separate clear pass is needed.
//
// The routing weight is read from a buffer rather than a root constant because it is
// produced on the GPU by RouterTopK and never round-trips through the CPU.
//
//   t0 in   t1 weights (FP32 array; index selected by gp1.y)
//   u0 out
//
//   gp0 = (n, in_off, out_off, w_off)
//   gp1 = (accumulate, weight_index, 0, 0)

#include "Common.hlsli"

#define SA_THREADS 256

[numthreads(SA_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n          = gp0.x;
    const uint in_off     = gp0.y;
    const uint out_off    = gp0.z;
    const uint w_off      = gp0.w;
    const bool accumulate = (gp1.x != 0u);
    const uint w_index    = gp1.y;

    const uint i = gid.x;
    if (i >= n) return;

    const float scale = f32_load(g_in1, w_off + w_index * 4);
    const float v = scale * f32_load(g_in0, in_off + i * 4);
    f32_store(g_out0, out_off + i * 4, accumulate ? (f32_load(g_out0, out_off + i * 4) + v) : v);
}
