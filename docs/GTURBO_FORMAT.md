# The `.gturbo` bundle format

A `.gturbo` "model" is a directory, not a single file. Turbo-WinFare reads and writes the same
layout as the Swift/Metal reference, so a bundle produced by either implementation works in the
other.

Bundles are never committed to this repository — see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Layout

```
gemma-4-26b-a4b.gturbo/
  manifest.json                    arch + per-file size and sha256
  resident.bin                     16 KB index region, then all non-expert weights
  tokenizer/
    tokenizer.json                 HF BPE, 262,144 entries
    tokenizer_config.json
    chat_template.jinja            shipped, never read (see docs/ARCHITECTURE.md)
  packed_experts/
    layout.json                    one expertBlock description, not 3,840 copies
    layer_00.bin .. layer_29.bin   128 blocks of 3,358,720 B each
```

## Alignment

**Every sub-tensor and every expert block starts on a 16 KB boundary**
(`GTurboFormatV1::ALIGNMENT_BYTES`, [../include/gturbo/format.hpp](../include/gturbo/format.hpp)),
so reads land on sector boundaries and can be issued as unbuffered DMA straight into UMA memory.

This is why `resident.bin` is ~7.5 MB larger than the reference's `model_weights.bin`: the
payload and index totals are identical, but each `scales` / `biases` sub-region is aligned here
rather than packed tight.

## Quantization

Weights are **MLX affine 4-bit, group 64, BF16 scale + BF16 bias**; routers are 8-bit. They are
copied through from the source checkpoint **unchanged** — there is no requantization step in
`tools/convert_hf_to_gturbo.py`.

Dequantization is `w = q * scale + bias`, packed low-nibble-first. (An earlier version of this
codebase documented a "Q4_K_M group 32" scheme; no such thing exists in this model, and any code
written against that description is wrong.)

## Resident index

`resident.bin` opens with a 16 KB index region decoded by `ResidentIndexCodec`
([../src/resident_index.cpp](../src/resident_index.cpp)). Tensor names are the HuggingFace names
with the `language_model.` prefix stripped, e.g. `model.layers.0.self_attn.q_proj.weight`.

A quantized tensor is a **single** index entry whose `scale_offset` and `bias_offset` point at
its `.scales` and `.biases` regions — not three separate entries.

Multimodal tensors (`vision_tower`, `embed_vision`) are excluded; this is a text-only engine.

## Expert blocks

`layout.json` describes the nine sub-tensor offsets **once**, under `expertBlock`, rather than
enumerating all 3,840 (layer, expert) pairs. That is why it is 3.4 KB instead of 6.9 MB. A
block's file offset is `expert_index * expert_stride`.

> **`expert_stride` is 3,358,720 bytes.** An earlier version of this codebase hardcoded 692,224
> in several places. Any code computing an expert offset from that value reads a completely
> different expert, and the model keeps producing fluent text while doing it.

## Verification against the reference

The converter's output was checked field by field against a bundle produced by the Swift
reference:

| | Turbo-WinFare | Reference |
|---|---|---|
| expert stride | 3,358,720 B | 3,358,720 B |
| `packed_experts/` total | 12,897,484,800 B | 12,897,484,800 B |
| resident payload + index | 1,353,689,148 + 81,920 | 1,353,771,068 |
| resident index sha256 | `bf198c9f…c850b13` | `bf198c9f…c850b13` |

## Building a bundle

```powershell
python tools/convert_hf_to_gturbo.py --output gemma-4-26b-a4b.gturbo
```

The converter streams: source tensors are read in fixed-size chunks and scattered directly into
the output files, so peak memory stays at a few MB regardless of the 15 GB input. It never holds
a whole shard or tensor in memory.

`tools/test_convert_streaming.py` (registered with CTest as `test_convert`) exercises the layout
and scatter paths offline against synthetic data. It guards the expert scatter in particular,
where a wrong offset would silently corrupt 12 GB of weights.
