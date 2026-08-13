// Fused LM Head GEMV + Parallel Greedy Argmax Shader
#pragma waveSize 32

StructuredBuffer<float16_t> g_WeightTable : register(t0);
StructuredBuffer<float16_t> g_Scales      : register(t1);
StructuredBuffer<float16_t> g_Biases      : register(t2);
StructuredBuffer<float16_t> g_Input       : register(t3);
RWStructuredBuffer<uint> g_OutToken       : register(u0);

cbuffer KernelParams : register(b0) {
    uint D;          // 2816
    uint vocabSize;  // 256000
    float softcap;   // 30.0f
    uint weightOffset;
};

groupshared float s_maxLogits[256];
groupshared uint  s_bestTokens[256];

[numthreads(256, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    float threadMax = -1e30f;
    uint threadBest = 0;

    for (uint row = threadID; row < vocabSize; row += 256) {
        float acc = 0.0f;
        uint baseIdx = weightOffset + row * D;
        for (uint d = 0; d < D; ++d) {
            acc += (float)g_WeightTable[baseIdx + d] * (float)g_Input[d];
        }

        float capped = (softcap > 0.0f) ? (tanh(acc / softcap) * softcap) : acc;
        if (capped > threadMax) {
            threadMax = capped;
            threadBest = row;
        }
    }

    s_maxLogits[threadID] = threadMax;
    s_bestTokens[threadID] = threadBest;
    GroupMemoryBarrierWithGroupSync();

    if (threadID == 0) {
        float groupMax = -1e30f;
        uint groupBest = 0;
        for (uint i = 0; i < 256; ++i) {
            if (s_maxLogits[i] > groupMax) {
                groupMax = s_maxLogits[i];
                groupBest = s_bestTokens[i];
            }
        }
        g_OutToken[0] = groupBest;
    }
}
