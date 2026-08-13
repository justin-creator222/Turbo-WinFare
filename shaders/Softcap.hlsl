// Final logit softcapping: z' = cap * tanh(z / cap), with cap = 30 for this model.
//
// Applied to the LM head output only. There is deliberately NO router-logit softcap --
// applying one there is a documented difference in this architecture.
//
// Argmax and sampling stay on the CPU: temperature/top-p/top-k need the full distribution
// anyway, and the readback is 1 MB per token over UMA.
//
//   t0 logits (FP32)
//   u0 out (FP32; may alias t0's buffer)
//
//   gp0 = (n, in_off, out_off, cap_bits)

#include "Common.hlsli"

#define SOFTCAP_THREADS 256

[numthreads(SOFTCAP_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint n       = gp0.x;
    const uint in_off  = gp0.y;
    const uint out_off = gp0.z;
    const float cap    = asfloat(gp0.w);

    const uint i = gid.x;
    if (i >= n) return;

    const float z = f32_load(g_in0, in_off + i * 4);
    f32_store(g_out0, out_off + i * 4, (cap > 0.0f) ? cap * tanh(z / cap) : z);
}
