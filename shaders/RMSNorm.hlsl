// Fused RMSNorm Compute Shader
#pragma waveSize 32

StructuredBuffer<float16_t> g_Input   : register(t0);
StructuredBuffer<float16_t> g_Weight  : register(t1);
RWStructuredBuffer<float16_t> g_Output : register(u0);

cbuffer KernelParams : register(b0) {
    uint D;          // Hidden size (e.g. 2816)
    float eps;       // 1e-6f
    uint weightOffset;
    uint pad0;
};

[numthreads(32, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    uint itemsPerThread = D / 32;
    uint offset = threadID * itemsPerThread;
    
    float sumSq = 0.0f;
    for (uint i = 0; i < itemsPerThread; ++i) {
        float val = (float)g_Input[offset + i];
        sumSq += val * val;
    }

    float waveSumSq = WaveActiveSum(sumSq);
    float rmsInv = rsqrt(waveSumSq / (float)D + eps);

    for (uint j = 0; j < itemsPerThread; ++j) {
        uint idx = offset + j;
        float val = (float)g_Input[idx];
        float weight = (float)g_Weight[weightOffset + idx];
        g_Output[idx] = (float16_t)(val * rmsInv * weight);
    }
}
