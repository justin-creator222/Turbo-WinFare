#include "gturbo/format.hpp"
#include "gturbo/manifest.hpp"
#include "gturbo/packed_experts.hpp"
#include "gturbo/resident_index.hpp"
#include "gturbo/kv_cache.hpp"
#include "gturbo/runner.hpp"
#include "gturbo/d3d12_context.hpp"
#include "gturbo/pipeline.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] Running gturbo binary format & optimization unit tests...\n";

    // Test 1: Math checked add and multiply
    assert(gturbo::checked_add(100, 200, "test") == 300);
    assert(gturbo::checked_multiply(50, 40, "test") == 2000);
    
    bool overflow_caught = false;
    try {
        gturbo::checked_add(UINT64_MAX - 10, 20, "test_overflow");
    } catch (const gturbo::GTurboFormatError&) {
        overflow_caught = true;
    }
    assert(overflow_caught);
    std::cout << "  [PASS] Checked arithmetic tests passed.\n";

    // Test 2: Manifest parsing round-trip.
    // from_json_string used to ignore its argument entirely and return a hardcoded struct;
    // these assertions exist so that can never silently come back.
    {
        bool empty_rejected = false;
        try {
            gturbo::GTurboManifestV1::from_json_string("");
        } catch (const gturbo::GTurboFormatError&) {
            empty_rejected = true;
        }
        assert(empty_rejected);

        bool garbage_rejected = false;
        try {
            gturbo::GTurboManifestV1::from_json_string("{\"magic\": \"NOPE\"}");
        } catch (const gturbo::GTurboFormatError&) {
            garbage_rejected = true;
        }
        assert(garbage_rejected);
        std::cout << "  [PASS] Manifest rejects empty and malformed JSON.\n";
    }

    gturbo::GTurboManifestV1 manifest;
    manifest.model_id = "gemma-4-26b-a4b";
    manifest.expert_stride = 3358720;
    manifest.flags["streamingPresent"] = true;
    manifest.arch.full_attention_layer_mask.assign(30, 0);
    for (int i = 5; i < 30; i += 6) manifest.arch.full_attention_layer_mask[i] = 1;
    manifest.quant = gturbo::ManifestQuantV1();
    manifest.validate();

    std::string manifest_json = manifest.to_json_string();
    assert(manifest_json.find("GTURBO") != std::string::npos);

    gturbo::GTurboManifestV1 reparsed = gturbo::GTurboManifestV1::from_json_string(manifest_json);
    assert(reparsed.expert_stride == 3358720);
    assert(reparsed.arch.hidden_size == 2816);
    assert(reparsed.arch.moe_intermediate_size == 704);
    assert(reparsed.arch.num_kv_heads == 8);
    assert(reparsed.arch.num_full_kv_heads == 2);
    assert(reparsed.arch.head_dim == 256);
    assert(reparsed.arch.full_head_dim == 512);
    assert(reparsed.arch.vocab_size == 262144);
    assert(reparsed.arch.attention_k_eq_v == true);
    assert(reparsed.arch.full_attention_layer_mask.size() == 30);
    // Full attention on layers 5, 11, 17, 23, 29 -- five of thirty, not "every other one".
    int full_count = 0;
    for (size_t i = 0; i < reparsed.arch.full_attention_layer_mask.size(); ++i) {
        if (reparsed.arch.full_attention_layer_mask[i]) {
            assert(i % 6 == 5);
            ++full_count;
        }
    }
    assert(full_count == 5);
    std::cout << "  [PASS] Manifest JSON round-trips architecture and layer mask.\n";

    // Test 3: Packed Experts Layout round-trip on the compact schema.
    gturbo::PackedExpertsLayoutV1 layout;
    layout.expert_stride = 3358720;
    {
        uint64_t off = 0;
        const char* roles[] = {"gate_proj", "up_proj", "down_proj"};
        for (const char* role : roles) {
            for (const char* part : {"weight", "scales", "biases"}) {
                gturbo::SubTensorV1 t;
                t.offset = off;
                t.size = (std::string(part) == "weight") ? 991232 : 61952;
                t.dtype = (std::string(part) == "weight") ? 0 : 1;
                t.shape = {704, 352};
                layout.expert_block[std::string(role) + "." + part] = t;
                off += t.size;
            }
        }
        assert(off == 3345408); // raw block, before alignment padding
    }
    for (int l = 0; l < 30; ++l) {
        char buf[32];
        snprintf(buf, sizeof(buf), "layer_%02d.bin", l);
        layout.layers.push_back(gturbo::LayerV1{l, buf});
    }
    layout.cross_validate(manifest);

    gturbo::PackedExpertsLayoutV1 relayout =
        gturbo::PackedExpertsLayoutV1::from_json_string(layout.to_json_string());
    assert(relayout.expert_stride == 3358720);
    assert(relayout.layers.size() == 30);
    assert(relayout.expert_block.size() == 9);
    // Expert 3's gate weights start at 3 strides in, at block offset 0.
    assert(relayout.sub_tensor_offset(3, "gate_proj.weight") == 3ULL * 3358720);
    assert(relayout.sub_tensor_offset(0, "up_proj.weight") == 1115136);
    std::cout << "  [PASS] Packed Experts layout round-trips and resolves offsets.\n";

    // Test 4: Resident Index Binary Header Codec
    gturbo::ResidentIndexHeaderV1 header{81920, 1361346560, 597};
    uint8_t header_bytes[24];
    gturbo::ResidentIndexCodec::write_header(header_bytes, header);
    
    gturbo::ResidentIndexHeaderV1 decoded = gturbo::ResidentIndexCodec::decode_header(header_bytes, 24);
    assert(decoded.index_size == header.index_size);
    assert(decoded.resident_size == header.resident_size);
    assert(decoded.entry_count == header.entry_count);
    std::cout << "  [PASS] Resident Index binary header codec passed.\n";

    // Test 5: SWA ring-buffer KV cache.
    //
    // Real Gemma 4 geometry: full attention on layers 5, 11, 17, 23, 29; sliding window 1024;
    // 8 KV heads at head_dim 256 for SWA, 2 heads at 512 for full.
    std::vector<int> full_layer_mask(30, 0);
    for (int i = 5; i < 30; i += 6) full_layer_mask[i] = 1;

    auto dummy_ctx = std::make_shared<gturbo::D3D12Context>();
    dummy_ctx->initialize(false);
    const int kMaxContext = 4096, kWindow = 1024;
    gturbo::KVCacheManager kv_mgr(dummy_ctx, 30, /*kv_heads*/8, /*full_kv_heads*/2,
                                  /*head_dim*/256, /*full_head_dim*/512,
                                  kMaxContext, kWindow, full_layer_mask);

    // Full-attention layers keep the whole history and scale with context.
    assert(kv_mgr.is_full_layer(5) == true);
    assert(kv_mgr.layer_capacity(5) == kMaxContext);
    assert(kv_mgr.physical_slot(5, 4000) == 4000);      // linear, no wrap below max_context

    // Sliding-window layers ring at `sliding_window` REGARDLESS of context length. This
    // previously returned a hardcoded 640 that ignored sliding_window entirely, which is
    // why the KV cache used to scale with context on all 30 layers.
    assert(kv_mgr.is_full_layer(0) == false);
    assert(kv_mgr.layer_capacity(0) == kWindow);
    assert(kv_mgr.physical_slot(0, 5) == 5);            // before the first wrap
    assert(kv_mgr.physical_slot(0, kWindow) == 0);      // exactly at the wrap
    assert(kv_mgr.physical_slot(0, kWindow + 5) == 5);  // position p overwrites p - window

    // Sizing: FP32, and only full layers grow with context.
    assert(kv_mgr.layer_kv_bytes(0) == static_cast<size_t>(kWindow) * 8 * 256 * 4);
    assert(kv_mgr.layer_kv_bytes(5) == static_cast<size_t>(kMaxContext) * 2 * 512 * 4);
    // 25 SWA + 5 full, K and V each.
    const uint64_t expect_total =
        (25ull * static_cast<uint64_t>(kWindow) * 8 * 256 * 4 +
          5ull * static_cast<uint64_t>(kMaxContext) * 2 * 512 * 4) * 2ull;
    assert(kv_mgr.total_memory_bytes() == expect_total);
    std::cout << "  [PASS] SWA ring-buffer capacity, wrap indexing and FP32 sizing passed ("
              << (expect_total / (1024 * 1024)) << " MB at " << kMaxContext << " ctx).\n";

    // Test 6: DirectX 12 Context & Compute Shader Integration
    try {
        auto info = dummy_ctx->query_memory_info();
        assert(info.total_system_ram_bytes > 0);
        std::cout << "  [PASS] D3D12 Context initialized on: " << dummy_ctx->adapter_name() << "\n";
        
        gturbo::ComputePipelineManager pipeline_mgr(dummy_ctx);
        pipeline_mgr.initialize_pipelines(gturbo::ComputePipelineManager::kLegacyDescriptorCapacity);
        std::cout << "  [PASS] Compute shader pipelines initialized.\n";
    } catch (const std::exception& ex) {
        std::cout << "  [NOTE] GPU compute test skipped: " << ex.what() << "\n";
    }

    std::cout << "[TEST SUCCESS] All format and optimization unit tests PASSED!\n";
    return 0;
}
