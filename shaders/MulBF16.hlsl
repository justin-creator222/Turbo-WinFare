// out[i] = in[i] * bf16(w[i]) * scale
//
// Builds the router's input: rmsnorm_no_scale(hidden) * router.scale / sqrt(D). The
// reference folds 1/sqrt(D) into router.scale once at load time; here it is passed as the
// constant so the resident weights stay untouched.
//
//   t0 in   t1 w (BF16)
//   u0 out
//
//   gp0 = (n, in_off, w_off, out_off)
//   gp1 = (scale_bits, 0, 0, 0)

#include "Common.hlsli"

#define MUL_THREADS 256

[numthreads(MUL_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n       = gp0.x;
    const uint in_off  = gp0.y;
    const uint w_off   = gp0.z;
    const uint out_off = gp0.w;
    const float scale  = asfloat(gp1.x);

    const uint i = gid.x;
    if (i >= n) return;

    f32_store(g_out0, out_off + i * 4,
              f32_load(g_in0, in_off + i * 4) * bf16_load(g_in1, w_off + i * 2) * scale);
}
