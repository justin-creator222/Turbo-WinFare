// RDNA 3 Wave32 Optimized Int8 Dequantized Matrix-Vector Multiplication for Shared Expert
#pragma waveSize 32

Texture2D<uint> g_WeightTable : register(t0);
StructuredBuffer<float16_t> g_Scales : register(t1);
StructuredBuffer<float16_t> g_Biases : register(t2);
StructuredBuffer<float16_t> g_Input  : register(t3);
RWStructuredBuffer<float16_t> g_Output : register(u0);

cbuffer KernelParams : register(b0) {
    uint K;
    uint N;
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

    for (uint i = 0; i < itemsPerThread; i += 4) {
        uint kIdx = offset + i;
        uint packed = g_WeightTable.Load(int3(kIdx / 4, row, 0));
        
        uint groupIdx = kIdx / 64; // Int8 group size 64
        float16_t scale = g_Scales[row * (K / 64) + groupIdx];
        float16_t bias  = g_Biases[row * (K / 64) + groupIdx];

        [unroll]
        for (uint b = 0; b < 4; ++b) {
            int w_int8 = (int)((packed >> (b * 8)) & 0xFF) - 128;
            float16_t w_fp16 = (float16_t)w_int8 * scale + bias;
            acc += (float)w_fp16 * (float)g_Input[kIdx + b];
        }
    }

    float waveSum = WaveActiveSum(acc);
    if (threadID == 0) {
        g_Output[row] = (float16_t)(waveSum * (float)outScale);
    }
}
