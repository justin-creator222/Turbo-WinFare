// act[i] = gelu_pytorch_tanh(gate[i]) * up[i]
//
// Shared by the dense MLP and every routed expert -- both are GeGLU with the same
// activation, differing only in width (2112 vs 704).
//
//   t0 gate   t1 up
//   u0 act (FP32)
//
//   gp0 = (n, gate_byte_off, up_byte_off, out_byte_off)

#include "Common.hlsli"

#define GEGLU_THREADS 256

[numthreads(GEGLU_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n       = gp0.x;
    const uint g_off   = gp0.y;
    const uint u_off   = gp0.z;
    const uint out_off = gp0.w;

    const uint i = gid.x;
    if (i >= n) return;

    const float g = f32_load(g_in0, g_off + i * 4);
    const float u = f32_load(g_in1, u_off + i * 4);
    f32_store(g_out0, out_off + i * 4, gelu_tanh(g) * u);
}
