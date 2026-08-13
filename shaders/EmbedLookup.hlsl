// Dequantizes one row of the embedding table and scales it by sqrt(hidden_size).
//
// The scale is not optional: Gemma multiplies the embedding by sqrt(D) before the first
// layer. The same matrix is reused as the LM head (tie_word_embeddings), but there it is
// consumed by GemvInt4 rather than here.
//
//   t0 weights   t1 scales   t2 biases
//   u0 hidden (FP32)
//
//   gp0 = (token, in_dim, w_byte_off, s_byte_off)
//   gp1 = (b_byte_off, out_byte_off, scale_bits, 0)

#include "Common.hlsli"

#define EMBED_THREADS 256

[numthreads(EMBED_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    const uint token   = gp0.x;
    const uint in_dim  = gp0.y;
    const uint w_off   = gp0.z;
    const uint s_off   = gp0.w;
    const uint b_off   = gp1.x;
    const uint out_off = gp1.y;
    const float scale  = asfloat(gp1.z);

    const uint groups    = in_dim / GROUP_SIZE;
    const uint row_bytes = in_dim / 2;
    const uint w_row = w_off + token * row_bytes;
    const uint s_row = s_off + token * groups * 2;
    const uint b_row = b_off + token * groups * 2;

    for (uint c = tid; c < in_dim; c += EMBED_THREADS) {
        const uint g = c / GROUP_SIZE;
        const float s = bf16_load(g_in1, s_row + g * 2);
        const float b = bf16_load(g_in2, b_row + g * 2);
        // Low nibble first: element c lives in byte c/2, low half when c is even.
        const uint byte_off = w_row + (c >> 1);
        const uint word = g_in0.Load(byte_off & ~3u);
        const uint byte = (word >> ((byte_off & 3u) * 8)) & 0xFFu;
        const uint q = ((c & 1u) == 0u) ? (byte & 0x0Fu) : (byte >> 4);
        f32_store(g_out0, out_off + c * 4, (float(q) * s + b) * scale);
    }
}
