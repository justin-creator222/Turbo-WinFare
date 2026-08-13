#!/usr/bin/env python3
"""
Unit tests for the streaming/scatter core of convert_hf_to_gturbo.py.

These run in under a second against synthetic data. The scatter path in particular is
worth testing offline: it interleaves 128 expert slices out of one sequential read, and a
misplaced offset there would silently corrupt 12 GB of expert weights in a way that only
shows up as bad model output much later.

    python tools/test_convert_streaming.py
"""

import io
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_hf_to_gturbo as C


class FakeSource:
    """Serves deterministic bytes, chunked awkwardly to exercise chunk-boundary handling."""

    def __init__(self, blobs, chunk=7):
        self.blobs = blobs
        self.chunk = chunk
        self.requests = 0

    def read_all(self, name):
        return self.blobs[name]

    def open_range(self, name, offset, length):
        self.requests += 1
        data = self.blobs[name][offset:offset + length]
        assert len(data) == length, f"{name}: short range {len(data)} != {length}"
        for i in range(0, len(data), self.chunk):
            yield data[i:i + self.chunk]


def check(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)
    print(f"  [PASS] {msg}")


def test_stream_into():
    blob = bytes((i * 31 + 7) & 0xFF for i in range(1000))
    src = FakeSource({"s": blob}, chunk=13)
    prog = C.Progress(300, "t")
    prog.advance = lambda n: None

    with tempfile.TemporaryFile() as f:
        f.truncate(4096)
        C.stream_into(src, f, "s", 100, 300, 2048, prog)
        f.seek(2048)
        got = f.read(300)
    check(got == blob[100:400], "stream_into copies the exact source range")


def test_stream_scatter_uniform():
    """The real shape: one source tensor split into N expert slices at a fixed stride."""
    n_experts, slice_bytes, stride, block_off = 128, 97, 512, 40
    total = n_experts * slice_bytes
    blob = bytes((i * 17 + 3) & 0xFF for i in range(total))
    src = FakeSource({"s": blob}, chunk=11)  # deliberately not a divisor of slice_bytes
    prog = C.Progress(total, "t")
    prog.advance = lambda n: None

    offsets = [e * stride + block_off for e in range(n_experts)]
    with tempfile.TemporaryFile() as f:
        f.truncate(n_experts * stride)
        C.stream_scatter(src, f, "s", 0, total, slice_bytes, offsets, prog)
        bad = 0
        for e in range(n_experts):
            f.seek(offsets[e])
            if f.read(slice_bytes) != blob[e * slice_bytes:(e + 1) * slice_bytes]:
                bad += 1
    check(bad == 0, f"stream_scatter places all {n_experts} slices correctly")


def test_stream_scatter_detects_mismatch():
    blob = b"x" * 100
    src = FakeSource({"s": blob}, chunk=8)
    prog = C.Progress(100, "t")
    prog.advance = lambda n: None
    # 100 bytes cannot fill 3 slices of 40.
    with tempfile.TemporaryFile() as f:
        f.truncate(1000)
        try:
            C.stream_scatter(src, f, "s", 0, 100, 40, [0, 100, 200], prog)
        except RuntimeError:
            check(True, "stream_scatter rejects a partial final slice")
            return
    check(False, "stream_scatter should have rejected a partial final slice")


def test_index_roundtrip():
    """Encode an index, then decode it with the C++ decoder's exact validation rules."""
    entries = [
        C.ResidentEntry("model.embed_tokens.weight", C.DTYPE_U32, [262144, 352],
                        ("s", 0, 369098752), ("s", 0, 23068672), ("s", 0, 23068672)),
        C.ResidentEntry("model.layers.0.self_attn.q_norm.weight", C.DTYPE_BF16, [256],
                        ("s", 0, 512), None, None),
        C.ResidentEntry("model.norm.weight", C.DTYPE_BF16, [2816],
                        ("s", 0, 5632), None, None),
    ]
    index_size, resident_size = C.plan_resident(entries)
    region = C.build_resident_index(entries, index_size, resident_size)

    check(len(region) == index_size, "index region is exactly index_size bytes")
    check(index_size % C.ALIGNMENT == 0, "index_size is 16 KB aligned")

    isz, rsz, cnt = struct.unpack("<QQQ", region[:24])
    check((isz, rsz, cnt) == (index_size, resident_size, len(entries)),
          "header round-trips (index_size, resident_size, entry_count)")

    table_end = C.RESIDENT_HEADER_BYTES + cnt * C.RESIDENT_ENTRY_BYTES
    for i, e in enumerate(entries):
        off = C.RESIDENT_HEADER_BYTES + i * C.RESIDENT_ENTRY_BYTES
        ent = region[off:off + C.RESIDENT_ENTRY_BYTES]
        name_off, name_len, dtype, reserved = struct.unpack("<IHBB", ent[:8])
        # These are exactly the checks in ResidentIndexCodec::decode_region.
        assert reserved == 0, "reserved byte must be zero"
        assert name_off >= table_end, "name must live past the entry table"
        assert name_off + name_len <= isz, "name must live inside the index"
        assert region[name_off:name_off + name_len].decode() == e.name
        f_off, f_size = struct.unpack("<QQ", ent[8:24])
        assert (f_off, f_size) == (e.offset, e.size), (e.name, f_off, f_size)
        shape = struct.unpack("<4I", ent[24:40])
        assert list(shape) == (list(e.shape) + [0, 0, 0, 0])[:4]
        sc_off, sc_sz, bi_off, bi_sz = struct.unpack("<QQQQ", ent[40:72])
        assert sc_sz == (e.scale_src[2] if e.scale_src else 0)
        assert bi_sz == (e.bias_src[2] if e.bias_src else 0)
    check(True, "every entry decodes under the C++ validation rules")


def test_alignment_invariants():
    entries = [
        C.ResidentEntry(f"t{i}", C.DTYPE_BF16, [3], ("s", 0, 5632 + i * 111),
                        ("s", 0, 64) if i % 2 else None,
                        ("s", 0, 64) if i % 3 == 0 else None)
        for i in range(40)
    ]
    index_size, resident_size = C.plan_resident(entries)
    regions = []
    for e in entries:
        check_off = [(e.offset, e.size)]
        if e.scale_src:
            check_off.append((e.scale_offset, e.scale_src[2]))
        if e.bias_src:
            check_off.append((e.bias_offset, e.bias_src[2]))
        for off, size in check_off:
            assert off % C.ALIGNMENT == 0, f"{e.name} region @ {off} not aligned"
            regions.append((off, size))
    regions.sort()
    prev_end = index_size
    for off, size in regions:
        assert off >= prev_end, f"overlap at {off} (previous ended {prev_end})"
        prev_end = off + size
    check(prev_end <= resident_size, "no overlaps; everything fits in resident_size")


def test_manifest_arch_from_config():
    """The manifest must be derived from the checkpoint, never hardcoded."""
    cfg = {
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "text_config": {
            "hidden_size": 2816, "intermediate_size": 2112, "moe_intermediate_size": 704,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "num_global_key_value_heads": 2, "head_dim": 256, "global_head_dim": 512,
            "vocab_size": 262144, "sliding_window": 1024,
            "final_logit_softcapping": 30.0, "rms_norm_eps": 1e-6,
            "num_hidden_layers": 30, "num_experts": 128, "top_k_experts": 8,
            "tie_word_embeddings": True, "attention_k_eq_v": True,
            "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
                "sliding_attention": {"rope_theta": 10000.0}},
            "layer_types": ["full_attention" if i in (5, 11, 17, 23, 29)
                            else "sliding_attention" for i in range(30)],
        },
    }
    m = C.build_manifest(cfg, 3358720, 30, 128, {}, "abc123")
    a = m["arch"]
    expect = {
        "hiddenSize": 2816, "moeIntermediateSize": 704, "numKVHeads": 8,
        "numFullKVHeads": 2, "headDim": 256, "fullHeadDim": 512, "vocabSize": 262144,
        "fullRopeTheta": 1000000.0, "ropeTheta": 10000.0, "partialRotaryFactor": 0.25,
        "attentionKEqV": True, "topKExperts": 8,
    }
    for k, v in expect.items():
        assert a[k] == v, f"arch.{k}: got {a[k]}, want {v}"
    mask = a["fullAttentionLayerMask"]
    assert [i for i, x in enumerate(mask) if x] == [5, 11, 17, 23, 29], mask
    assert m["quant"]["router"]["weightBits"] == 8, "routers are 8-bit"
    assert m["quant"]["routedExpert"]["weightBits"] == 4
    assert m["quant"]["routedExpert"]["groupSize"] == 64
    check(True, "manifest arch matches the checkpoint (mask = layers 5,11,17,23,29)")


def main():
    print("[TEST] convert_hf_to_gturbo streaming/layout tests")
    for fn in (test_stream_into, test_stream_scatter_uniform,
               test_stream_scatter_detects_mismatch, test_index_roundtrip,
               test_alignment_invariants, test_manifest_arch_from_config):
        fn()
    print("[TEST SUCCESS] all converter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
