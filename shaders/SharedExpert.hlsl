// Shared Expert MLP Shader (Gate/Up/Down + GeLU Activation)
#pragma waveSize 32

StructuredBuffer<float16_t> g_GateUp : register(t0); // [2112 + 2112]
RWStructuredBuffer<float16_t> g_Output : register(u0);

cbuffer KernelParams : register(b0) {
    uint F; // 2112
    uint weightOffset;
    uint pad0;
    uint pad1;
};

// GeLU (PyTorch Tanh Approximation)
float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

[numthreads(32, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    uint idx = groupID.x * 32 + threadID;
    if (idx >= F) return;

    float gate = (float)g_GateUp[weightOffset + idx];
    float up   = (float)g_GateUp[weightOffset + F + idx];
    float act  = gelu_tanh(gate) * up;
    
    g_Output[idx] = (float16_t)act;
}
