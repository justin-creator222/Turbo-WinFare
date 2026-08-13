// RDNA 3 Wave32 Optimized Int4 Dequantized Matrix-Vector Multiplication
#pragma waveSize 32

Texture2D<uint> g_WeightTable : register(t0);
StructuredBuffer<float16_t> g_Scales : register(t1);
StructuredBuffer<float16_t> g_Biases : register(t2);
StructuredBuffer<float16_t> g_Input  : register(t3);
RWStructuredBuffer<float16_t> g_Output : register(u0);

cbuffer KernelParams : register(b0) {
    uint K;            // Hidden dimension D (e.g. 2816)
    uint N;            // Output dimension (e.g. 1056)
    float16_t outScale;
    uint pad0;
};

[numthreads(32, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    uint row = groupID.x;
    if (row >= N) return;

    float acc = 0.0f;
    uint itemsPerThread = K / 32;
    uint offset = threadID * itemsPerThread;

    for (uint i = 0; i < itemsPerThread; i += 8) {
        uint kIdx = offset + i;
        uint packed = g_WeightTable.Load(int3(kIdx / 8, row, 0));
        
        // Group size 32 -> 1 scale/bias per 32 elements
        uint groupIdx = kIdx / 32;
        float16_t scale = g_Scales[row * (K / 32) + groupIdx];
        float16_t bias  = g_Biases[row * (K / 32) + groupIdx];

        [unroll]
        for (uint b = 0; b < 8; ++b) {
            uint w_int4 = (packed >> (b * 4)) & 0x0F;
            float16_t w_fp16 = (float16_t)w_int4 * scale + bias;
            acc += (float)w_fp16 * (float)g_Input[kIdx + b];
        }
    }

    // Parallel Wave32 Reduction across SIMD lanes
    float waveSum = WaveActiveSum(acc);
    if (threadID == 0) {
        g_Output[row] = (float16_t)(waveSum * (float)outScale);
    }
}
