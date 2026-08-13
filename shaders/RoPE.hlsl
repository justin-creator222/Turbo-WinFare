// Rotary Position Embedding (RoPE) Compute Shader for Gemma 4
#pragma waveSize 32

RWStructuredBuffer<float16_t> g_Q : register(u0);
RWStructuredBuffer<float16_t> g_K : register(u1);

cbuffer KernelParams : register(b0) {
    uint position;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    float theta;
    uint rotatedPairs;
    uint isFull;
    uint ringCapacity;
};

[numthreads(32, 1, 1)]
void main(uint32_t threadID : SV_GroupIndex, uint32_t3 groupID : SV_GroupID) {
    uint headIdx = groupID.x;
    bool isQHead = (groupID.y == 0);

    if (isQHead && headIdx >= numQHeads) return;
    if (!isQHead && headIdx >= numKVHeads) return;

    uint qHeadOffset = headIdx * headDim;
    uint physPos = (ringCapacity > 0) ? (position % ringCapacity) : position;
    uint kHeadOffset = physPos * numKVHeads * headDim + headIdx * headDim;

    // Process half of rotated pairs (each thread handles pair)
    if (threadID < rotatedPairs) {
        uint pairIdx = threadID;
        float freq = 1.0f / pow(theta, (2.0f * (float)pairIdx) / (float)headDim);
        float val = (float)position * freq;
        float cosVal = cos(val);
        float sinVal = sin(val);

        if (isQHead) {
            uint i1 = qHeadOffset + pairIdx;
            uint i2 = qHeadOffset + pairIdx + rotatedPairs;
            float q1 = (float)g_Q[i1];
            float q2 = (float)g_Q[i2];
            g_Q[i1] = (float16_t)(q1 * cosVal - q2 * sinVal);
            g_Q[i2] = (float16_t)(q1 * sinVal + q2 * cosVal);
        } else {
            uint i1 = kHeadOffset + pairIdx;
            uint i2 = kHeadOffset + pairIdx + rotatedPairs;
            float k1 = (float)g_K[i1];
            float k2 = (float)g_K[i2];
            g_K[i1] = (float16_t)(k1 * cosVal - k2 * sinVal);
            g_K[i2] = (float16_t)(k1 * sinVal + k2 * cosVal);
        }
    }
}
