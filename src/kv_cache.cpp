#include "gturbo/kv_cache.hpp"
#include <windows.h>
#include <memoryapi.h>
#include <algorithm>

namespace gturbo {

KVCacheManager::KVCacheManager(std::shared_ptr<D3D12Context> ctx,
                               int num_layers,
                               int num_kv_heads,
                               int num_full_kv_heads,
                               int head_dim,
                               int full_head_dim,
                               int max_context,
                               int sliding_window,
                               const std::vector<int>& full_layer_mask)
    : ctx_(ctx), num_layers_(num_layers), num_kv_heads_(num_kv_heads),
      num_full_kv_heads_(num_full_kv_heads), head_dim_(head_dim),
      full_head_dim_(full_head_dim), max_context_(max_context),
      sliding_window_(sliding_window), full_layer_mask_(full_layer_mask) {
}

KVCacheManager::~KVCacheManager() {}

bool KVCacheManager::is_full_layer(int layer) const {
    return (layer < static_cast<int>(full_layer_mask_.size())) && (full_layer_mask_[layer] != 0);
}

// Sliding-window layers only ever attend to the last `sliding_window` positions, so they
// need exactly that many slots regardless of total context -- position p overwrites p-W,
// which is precisely the entry falling out of the window. Full-attention layers see the
// whole history and scale with max_context.
//
// This used to return a hardcoded `640`, ignoring the sliding_window_ the class already
// stores. That is why the KV cache scaled with context on all 30 layers.
int KVCacheManager::layer_capacity(int layer) const {
    return is_full_layer(layer) ? max_context_ : std::min(max_context_, sliding_window_);
}

int KVCacheManager::physical_slot(int layer, int position) const {
    int cap = layer_capacity(layer);
    return (cap > 0) ? (position % cap) : position;
}

size_t KVCacheManager::layer_kv_bytes(int layer) const {
    bool is_full = is_full_layer(layer);
    int capacity = layer_capacity(layer);
    int heads = is_full ? num_full_kv_heads_ : num_kv_heads_;
    int dim = is_full ? full_head_dim_ : head_dim_;
    // FP32: the GPU path stores activations and KV as float. (This said 2 bytes, from when
    // the engine was going to use FP16 -- it never did.)
    return static_cast<size_t>(capacity) * heads * dim * 4;
}

void KVCacheManager::initialize() {
    k_buffers_.resize(num_layers_);
    v_buffers_.resize(num_layers_);

    for (int l = 0; l < num_layers_; ++l) {
        size_t bytes = layer_kv_bytes(l);
        std::string k_name = "KV_K_Layer_" + std::to_string(l);
        std::string v_name = "KV_V_Layer_" + std::to_string(l);
        
        k_buffers_[l] = ctx_->create_uma_buffer(bytes, k_name);
        v_buffers_[l] = ctx_->create_uma_buffer(bytes, v_name);
    }
}

// Zeroing or discarding the slots is unnecessary: a slot is always written by QKVEpilogue
// earlier in the same command list than the Attention dispatch that reads it, and reads are
// bounded by n_pos, so stale bytes from a previous generation are never observed. The old
// implementation called DiscardVirtualMemory on a mapped D3D12 resource, which is not a
// supported way to manage a committed resource's pages.
void KVCacheManager::reset() {
    current_position_ = 0;
}

void KVCacheManager::advance_position(int count) {
    current_position_ += count;
}

uint64_t KVCacheManager::total_memory_bytes() const {
    uint64_t total = 0;
    for (int l = 0; l < num_layers_; ++l) {
        total += static_cast<uint64_t>(layer_kv_bytes(l)) * 2ULL; // K and V buffers
    }
    return total;
}

} // namespace gturbo
