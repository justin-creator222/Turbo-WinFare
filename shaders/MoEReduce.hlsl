// Fused MoE Token-Major Reduction Shader
#pragma waveSize 32

StructuredBuffer<float16_t> g_SharedOut : register(t0); // [D]
StructuredBuffer<float16_t> g_RoutedOut : register(t1); // [topK * D]
StructuredBuffer<float16_t> g_RouterWeights : register(t2); // [topK]
RWStructuredBuffer<float16_t> g_Output : register(u0); // [D]

cbuffer KernelParams : register(b0) {
    uint D;          // 2816
    uint topK;       // 8
    float layerScalar;
    uint pad0;
};

[numthreads(32, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    uint itemsPerThread = D / 32;
    uint offset = threadID * itemsPerThread;

    for (uint i = 0; i < itemsPerThread; ++i) {
        uint d = offset + i;
        float sharedVal = (float)g_SharedOut[d];
        
        float routedVal = 0.0f;
        for (uint k = 0; k < topK; ++k) {
            float w = (float)g_RouterWeights[k];
            float exVal = (float)g_RoutedOut[k * D + d];
            routedVal += w * exVal;
        }

        float totalVal = (sharedVal + routedVal) * layerScalar;
        g_Output[d] = (float16_t)totalVal;
    }
}
