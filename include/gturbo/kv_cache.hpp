#pragma once

#include "gturbo/d3d12_context.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace gturbo {

class KVCacheManager {
public:
    KVCacheManager(std::shared_ptr<D3D12Context> ctx,
                   int num_layers,
                   int num_kv_heads,
                   int num_full_kv_heads,
                   int head_dim,
                   int full_head_dim,
                   int max_context,
                   int sliding_window,
                   const std::vector<int>& full_layer_mask);
    ~KVCacheManager();

    void initialize();
    void reset();

    void advance_position(int count = 1);
    int current_position() const { return current_position_; }
    // Resume at a known position instead of from zero. Used to re-enter a cache whose slots
    // were filled by an earlier request (prompt-cache reuse); the caller is responsible for
    // having actually written those slots.
    void set_position(int position) { current_position_ = position; }

    ID3D12Resource* key_buffer(int layer) const { return k_buffers_[layer].Get(); }
    ID3D12Resource* value_buffer(int layer) const { return v_buffers_[layer].Get(); }

    size_t layer_kv_bytes(int layer) const;
    int layer_capacity(int layer) const;
    int physical_slot(int layer, int position) const;
    bool is_full_layer(int layer) const;
    uint64_t total_memory_bytes() const;

private:
    std::shared_ptr<D3D12Context> ctx_;
    int num_layers_;
    int num_kv_heads_;
    int num_full_kv_heads_;
    int head_dim_;
    int full_head_dim_;
    int max_context_;
    int sliding_window_;
    std::vector<int> full_layer_mask_;

    int current_position_{0};
    std::vector<ComPtr<ID3D12Resource>> k_buffers_;
    std::vector<ComPtr<ID3D12Resource>> v_buffers_;
};

} // namespace gturbo
