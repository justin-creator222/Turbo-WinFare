// Copyright 2026 justin-creator222. Licensed under the Apache License, Version 2.0.
//
// These kernels are original HLSL, but the numerics they implement -- the forward-pass
// constants, normalization conventions, and dispatch order -- are derived from the Metal
// implementation in TurboFieldfare (https://github.com/drumih/turbo-fieldfare), copyright
// Andrey Mikhaylov, Apache-2.0. See NOTICE in the repository root.
//
// Shared helpers and binding convention for the Turbo-WinFare compute kernels.
//
// BINDING CONVENTION
// ------------------
// Everything is a raw byte-address buffer. The previous kernels declared a mix of
// StructuredBuffer<float16_t> (stride 2) and Texture2D<uint> while the host bound every
// resource as a stride-4 structured buffer, so the views silently disagreed with the
// declarations. Raw buffers remove that entire class of bug: the shader does explicit
// Load()/Store() at byte offsets and the host only has to get the offsets right.
//
// Activations are FP32. This is a correctness-first port -- FP16 packing is a later
// optimization, and mixing it in now would make GPU-vs-CPU diffs ambiguous.
//
//   t0..t7  ByteAddressBuffer     inputs (weights, scales, biases, activations)
//   u0..u3  RWByteAddressBuffer   outputs
//   b0      16 root constants     see each kernel for its layout
//
// QUANTIZATION
// ------------
// MLX affine: w = q * scale + bias, group_size 64 along the input dimension, BF16 scale
// and bias. 4-bit values are packed low-nibble-first (element c is in byte c/2, low nibble
// when c is even); 8-bit values (routers only) are one byte each.

#ifndef GTURBO_COMMON_HLSLI
#define GTURBO_COMMON_HLSLI

static const uint GROUP_SIZE = 64;

// The number of threads in a GEMV threadgroup -- NOT an assumption about the hardware wave
// width. RDNA 3 may execute this as a single Wave64 with half the lanes inactive, in which
// case WaveActiveSum still reduces only the active lanes and the result is unchanged. Any
// kernel that needs the real wave width must query WaveGetLaneCount() at runtime.
static const uint GEMV_THREADS = 32;
static const uint WAVE_SIZE = GEMV_THREADS;

ByteAddressBuffer   g_in0 : register(t0);
ByteAddressBuffer   g_in1 : register(t1);
ByteAddressBuffer   g_in2 : register(t2);
ByteAddressBuffer   g_in3 : register(t3);
ByteAddressBuffer   g_in4 : register(t4);
ByteAddressBuffer   g_in5 : register(t5);

RWByteAddressBuffer g_out0 : register(u0);
RWByteAddressBuffer g_out1 : register(u1);
RWByteAddressBuffer g_out2 : register(u2);
RWByteAddressBuffer g_out3 : register(u3);

cbuffer Params : register(b0) {
    uint4 gp0;
    uint4 gp1;
    uint4 gp2;
    uint4 gp3;
};

// Reads a BF16 value at an arbitrary 2-byte-aligned offset.
float bf16_load(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint half_bits = ((byte_off & 2u) != 0u) ? (word >> 16) : (word & 0xFFFFu);
    return asfloat(half_bits << 16);
}

float f32_load(ByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load(byte_off));
}

// Overload for reading back a value this threadgroup just wrote. Callers must place an
// AllMemoryBarrierWithGroupSync() between the store and the load.
float f32_load(RWByteAddressBuffer buf, uint byte_off) {
    return asfloat(buf.Load(byte_off));
}

void f32_store(RWByteAddressBuffer buf, uint byte_off, float v) {
    buf.Store(byte_off, asuint(v));
}

// gelu_pytorch_tanh -- the activation named in the checkpoint's config.json.
float gelu_tanh(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCubic = 0.044715f;
    return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + kCubic * x * x * x)));
}

// One wave's dot product of an affine-quantized 4-bit row against an FP32 vector.
// Each lane walks a strided subset of the groups; the caller reduces with WaveActiveSum.
//
// The stride is WaveGetLaneCount(), NOT a compile-time constant. A fixed stride of 32 is
// correct only when the caller's threadgroup is exactly 32 threads; a 512-thread group on
// RDNA 3 (Wave64) passes lanes 0..63, and lanes 32..63 would then re-process groups 32..63
// and double-count them. That produced a *lower* maximum in the greedy LM head rather than
// an obviously broken one.
float gemv_int4_row_lane(ByteAddressBuffer W, uint w_base,
                         ByteAddressBuffer S, uint s_base,
                         ByteAddressBuffer B, uint b_base,
                         ByteAddressBuffer X, uint x_base,
                         uint row, uint in_dim, uint lane)
{
    const uint groups = in_dim / GROUP_SIZE;
    const uint row_bytes = in_dim / 2;
    const uint w_row = w_base + row * row_bytes;
    const uint s_row = s_base + row * groups * 2;
    const uint b_row = b_base + row * groups * 2;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += WaveGetLaneCount()) {
        const float s = bf16_load(S, s_row + g * 2);
        const float b = bf16_load(B, b_row + g * 2);
        const uint gw = w_row + g * (GROUP_SIZE / 2);   // 32 bytes per group
        const uint gx = x_base + g * GROUP_SIZE * 4;    // FP32 activations

        float dot = 0.0f;
        float sum = 0.0f;
        // MEASURED: replacing these scalar loads with two Load4 per w iteration is SLOWER on
        // RDNA 3 -- 7.3-7.8 tok/s against 8.6-9.0 for this version, across three variants
        // (local array, explicit scalars, and with `precise`). DXC already merges these into
        // wide fetches; hoisting all 8 values by hand only inflates register pressure. Do
        // not "optimize" this without measuring.
        [unroll]
        for (uint w = 0; w < 8; ++w) {
            const uint packed = W.Load(gw + w * 4);     // 8 nibbles = 8 elements
            [unroll]
            for (uint n = 0; n < 4; ++n) {
                const uint byte = (packed >> (n * 8)) & 0xFFu;
                const uint e = w * 8 + n * 2;
                const float x0 = f32_load(X, gx + (e + 0) * 4);
                const float x1 = f32_load(X, gx + (e + 1) * 4);
                dot += float(byte & 0x0Fu) * x0;
                dot += float(byte >> 4) * x1;
                sum += x0 + x1;
            }
        }
        acc += s * dot + b * sum;
    }
    return acc;
}

// Same, for 8-bit rows (routers). One byte per element.
float gemv_int8_row_lane(ByteAddressBuffer W, uint w_base,
                         ByteAddressBuffer S, uint s_base,
                         ByteAddressBuffer B, uint b_base,
                         ByteAddressBuffer X, uint x_base,
                         uint row, uint in_dim, uint lane)
{
    const uint groups = in_dim / GROUP_SIZE;
    const uint w_row = w_base + row * in_dim;
    const uint s_row = s_base + row * groups * 2;
    const uint b_row = b_base + row * groups * 2;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += WaveGetLaneCount()) {
        const float s = bf16_load(S, s_row + g * 2);
        const float b = bf16_load(B, b_row + g * 2);
        const uint gw = w_row + g * GROUP_SIZE;
        const uint gx = x_base + g * GROUP_SIZE * 4;

        float dot = 0.0f;
        float sum = 0.0f;
        // See the note in gemv_int4_row_lane: Load4 measured slower here too.
        [unroll]
        for (uint w = 0; w < 16; ++w) {
            const uint packed = W.Load(gw + w * 4);     // 4 bytes = 4 elements
            [unroll]
            for (uint n = 0; n < 4; ++n) {
                const float xv = f32_load(X, gx + (w * 4 + n) * 4);
                dot += float((packed >> (n * 8)) & 0xFFu) * xv;
                sum += xv;
            }
        }
        acc += s * dot + b * sum;
    }
    return acc;
}

#endif // GTURBO_COMMON_HLSLI
