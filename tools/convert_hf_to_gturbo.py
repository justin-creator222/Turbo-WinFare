#!/usr/bin/env python3
"""
Repacks the pinned Gemma 4 26B-A4B checkpoint into a .gturbo bundle.

    python tools/convert_hf_to_gturbo.py --output gemma-4-26b-a4b.gturbo

Source (pinned, see SupportedModelSource.swift:3-11 in the reference implementation):
    mlx-community/gemma-4-26b-a4b-it-4bit @ 0d77464eeb233a2da68ebf9d7dc4edaac7db956d

WHAT THIS DOES NOT DO
---------------------
It does not quantize. The source is already MLX affine 4-bit (group 64, BF16 scale + BF16
bias; routers are 8-bit) and every quantized value is copied through byte-for-byte. The
previous version of this file invented a "Q4_K_M group 32" scheme, never read its --input,
and synthesized weights from `(l*1000 + e*10 + k%255)` -- producing a bundle that loaded
cleanly and generated nothing but zeros.

It also never holds a whole shard or tensor in memory. Source tensors are streamed in
CHUNK_BYTES pieces and scattered directly into the output files, so peak RSS stays flat at
a few MB regardless of the 15 GB input. This mirrors the reference's hard invariant
(docs/SYSTEM_DESIGN.md:432-433).

OUTPUT LAYOUT
-------------
    <out>.gturbo/
      manifest.json                    arch + per-file size/sha256
      resident.bin                     16 KB index region, then all non-expert weights
      tokenizer/                       tokenizer.json et al, copied verbatim
      packed_experts/
        layout.json                    expert block description (compact)
        layer_00.bin .. layer_29.bin   128 fixed-stride expert blocks each

Every tensor and every expert block starts on a 16 KB boundary so the engine can issue
unbuffered (FILE_FLAG_NO_BUFFERING) NVMe reads straight into UMA memory. That padding is
why resident.bin comes out ~7.5 MB larger than the reference's model_weights.bin: the
payload and index totals are identical (1,353,689,148 + 81,920 = 1,353,771,068), but we
also align each scales/biases sub-region rather than packing them tight.

Resident tensor names are the HuggingFace names with the "language_model." prefix stripped,
matching the reference repacker (RepackPlanner.swift:109). Multimodal tensors (vision_tower,
embed_vision) are excluded -- this is a text-only engine.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned source
# ---------------------------------------------------------------------------

REPO_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
HF_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"

INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"

# Copied verbatim into <out>.gturbo/tokenizer/. The first two are mandatory.
TOKENIZER_FILES = [
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("special_tokens_map.json", False),
    ("chat_template.jinja", False),
    ("chat_template.json", False),
    ("generation_config.json", False),
]

# ---------------------------------------------------------------------------
# Format constants -- must match include/gturbo/format.hpp
# ---------------------------------------------------------------------------

ALIGNMENT = 16384                 # GTurboFormatV1::ALIGNMENT_BYTES
RESIDENT_HEADER_BYTES = 24        # GTurboFormatV1::RESIDENT_HEADER_BYTES
RESIDENT_ENTRY_BYTES = 72         # GTurboFormatV1::RESIDENT_ENTRY_BYTES
RESIDENT_INDEX_MAX_BYTES = 16 * 1024 * 1024

# DType enum in include/gturbo/format.hpp
DTYPE_U32, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32 = 0, 1, 2, 3
DTYPE_FROM_SAFETENSORS = {"U32": DTYPE_U32, "BF16": DTYPE_BF16,
                          "F16": DTYPE_FP16, "F32": DTYPE_FP32}
ITEMSIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}

CHUNK_BYTES = 4 * 1024 * 1024     # streaming granularity; bounds peak memory
ROUTED_MARKER = ".experts.switch_glu."
EXPERT_ROLES = ("gate_proj", "up_proj", "down_proj")
EXPERT_PARTS = ("weight", "scales", "biases")

# Order of the 9 sub-tensors inside one expert block.
EXPERT_BLOCK_ORDER = [(r, p) for r in EXPERT_ROLES for p in EXPERT_PARTS]


def align_up(n, a=ALIGNMENT):
    return (n + a - 1) // a * a


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024.0


# ---------------------------------------------------------------------------
# Byte sources
# ---------------------------------------------------------------------------

class RemoteSource:
    """Reads byte ranges from the pinned HuggingFace revision over HTTP."""

    def __init__(self, token=None):
        self.token = token or os.environ.get("HF_TOKEN")

    def _headers(self, extra=None):
        h = {"User-Agent": "turbo-winfare/convert"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    def read_all(self, name):
        req = urllib.request.Request(f"{HF_BASE}/{name}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(
                    f"Access denied for {name} (HTTP {e.code}).\n"
                    f"{REPO_ID} is a gated repository. Accept the license at\n"
                    f"    https://huggingface.co/{REPO_ID}\n"
                    "then set HF_TOKEN (or pass --token) to a read token.")
            raise

    def open_range(self, name, offset, length):
        """Yields chunks covering [offset, offset+length). Retries on interruption."""
        remaining = length
        pos = offset
        attempts = 0
        while remaining > 0:
            end = pos + remaining - 1
            req = urllib.request.Request(
                f"{HF_BASE}/{name}",
                headers=self._headers({"Range": f"bytes={pos}-{end}"}))
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    while remaining > 0:
                        chunk = r.read(min(CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        yield chunk
                        pos += len(chunk)
                        remaining -= len(chunk)
                attempts = 0
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as ex:
                attempts += 1
                if attempts > 5:
                    raise RuntimeError(
                        f"Giving up on {name} at byte {pos} after 5 retries: {ex}")
                wait = min(30, 2 ** attempts)
                print(f"\n    [retry {attempts}/5 in {wait}s at byte {pos:,}: {ex}]",
                      flush=True)
                time.sleep(wait)


class LocalSource:
    """Reads byte ranges from an already-downloaded snapshot directory."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise SystemExit(f"--input is not a directory: {self.root}")

    def read_all(self, name):
        p = self.root / name
        if not p.exists():
            raise SystemExit(f"Missing {name} in {self.root}")
        return p.read_bytes()

    def open_range(self, name, offset, length):
        with open(self.root / name, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise RuntimeError(f"Short read on {name} at {offset}")
                yield chunk
                remaining -= len(chunk)


# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------

def load_safetensors_headers(source, shards):
    """Returns {tensor_name: (shard, dtype, shape, abs_start, nbytes)}."""
    merged = {}
    for shard in shards:
        # 8-byte little-endian header length, then that many bytes of JSON.
        raw_len = b"".join(source.open_range(shard, 0, 8))
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > 64 * 1024 * 1024:
            raise RuntimeError(f"{shard}: implausible header length {header_len}")
        raw_hdr = b"".join(source.open_range(shard, 8, header_len))
        hdr = json.loads(raw_hdr)
        hdr.pop("__metadata__", None)
        data_start = 8 + header_len
        for name, meta in hdr.items():
            begin, end = meta["data_offsets"]
            merged[name] = (shard, meta["dtype"], meta["shape"],
                            data_start + begin, end - begin)
        print(f"    {shard}: {len(hdr)} tensors, header {header_len:,} B")
    return merged


def layer_index_of(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


SLOT_RANK = [
    (".self_attn.q_proj.", 0), (".self_attn.k_proj.", 1), (".self_attn.v_proj.", 2),
    (".self_attn.o_proj.", 3), (".self_attn.q_norm.", 4), (".self_attn.k_norm.", 5),
    (".router.proj.", 6), (".router.scale", 7), (".router.per_expert_scale", 8),
    (".mlp.gate_proj.", 9), (".mlp.up_proj.", 10), (".mlp.down_proj.", 11),
    (".input_layernorm.", 12), (".post_attention_layernorm.", 13),
    (".pre_feedforward_layernorm.", 14), (".pre_feedforward_layernorm_2.", 15),
    (".post_feedforward_layernorm.", 16), (".post_feedforward_layernorm_1.", 17),
    (".post_feedforward_layernorm_2.", 18), (".layer_scalar", 19),
]


def resident_sort_key(name):
    """Embedding first, then per-layer groups in slot order, then the final norm.

    Mirrors RepackPlanner.lmResidentOrdering / slotRank in the reference.
    """
    if name == "model.embed_tokens.weight":
        return (0, 0, 0, name)
    if name == "model.norm.weight":
        return (3, 0, 0, name)
    li = layer_index_of(name)
    if li is not None:
        for needle, rank in SLOT_RANK:
            if needle in name or name.endswith(needle.rstrip(".")):
                return (1, li, rank, name)
        return (1, li, 100, name)
    return (2, 0, 0, name)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class ResidentEntry:
    """One logical weight: a payload plus optional quantization scales/biases."""

    __slots__ = ("name", "dtype", "shape", "src", "scale_src", "bias_src",
                 "offset", "scale_offset", "bias_offset")

    def __init__(self, name, dtype, shape, src, scale_src, bias_src):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.src = src               # (shard, abs_start, nbytes)
        self.scale_src = scale_src   # or None
        self.bias_src = bias_src     # or None
        self.offset = 0
        self.scale_offset = 0
        self.bias_offset = 0

    @property
    def size(self):
        return self.src[2]


def group_resident(tensors):
    """Collapses {x.weight, x.scales, x.biases} triples into single entries.

    Entry names are the source names with "language_model." stripped and nothing else
    changed, so "model.layers.0.self_attn.q_proj.weight" stays exactly that and the bare
    scalars ("...layer_scalar", "...router.scale") keep their own names.
    """
    stems = {}
    for full, (shard, dt, shape, start, nbytes) in tensors.items():
        short = full[len("language_model."):]
        for suffix, part in ((".scales", "scales"), (".biases", "biases")):
            if short.endswith(suffix):
                stems.setdefault(short[: -len(suffix)], {})[part] = (
                    shard, start, nbytes)
                break
        else:
            # Either "<stem>.weight" or a bare scalar with no suffix at all.
            stem = short[: -len(".weight")] if short.endswith(".weight") else short
            stems.setdefault(stem, {})["payload"] = (
                shard, start, nbytes, dt, shape, short)

    entries = []
    for stem, parts in sorted(stems.items()):
        if "payload" not in parts:
            raise RuntimeError(f"{stem}: scales/biases present with no weight tensor")
        shard, start, nbytes, dt, shape, name = parts["payload"]
        entries.append(ResidentEntry(
            name=name,
            dtype=DTYPE_FROM_SAFETENSORS[dt],
            shape=shape,
            src=(shard, start, nbytes),
            scale_src=parts.get("scales"),
            bias_src=parts.get("biases"),
        ))
    entries.sort(key=lambda e: resident_sort_key(e.name))
    return entries


def plan_resident(entries):
    """Assigns 16 KB-aligned offsets. Returns (index_size, total_size)."""
    names_bytes = sum(len(e.name.encode()) for e in entries)
    raw_index = RESIDENT_HEADER_BYTES + len(entries) * RESIDENT_ENTRY_BYTES + names_bytes
    index_size = align_up(raw_index)
    if index_size > RESIDENT_INDEX_MAX_BYTES:
        raise RuntimeError(f"Resident index {index_size:,} B exceeds the 16 MB cap")

    cursor = index_size
    for e in entries:
        e.offset = cursor
        cursor = align_up(cursor + e.size)
        if e.scale_src:
            e.scale_offset = cursor
            cursor = align_up(cursor + e.scale_src[2])
        if e.bias_src:
            e.bias_offset = cursor
            cursor = align_up(cursor + e.bias_src[2])
    return index_size, cursor


def build_resident_index(entries, index_size, resident_size):
    """Serializes the index region: header, fixed-size entry table, string table."""
    table_end = RESIDENT_HEADER_BYTES + len(entries) * RESIDENT_ENTRY_BYTES
    table = bytearray()
    strings = bytearray()

    for e in entries:
        raw = e.name.encode()
        name_offset = table_end + len(strings)
        strings += raw
        shape4 = (list(e.shape) + [0, 0, 0, 0])[:4]
        table += struct.pack(
            "<IHBB", name_offset, len(raw), e.dtype, 0)
        table += struct.pack("<QQ", e.offset, e.size)
        table += struct.pack("<4I", *shape4)
        table += struct.pack("<QQQQ",
                             e.scale_offset, e.scale_src[2] if e.scale_src else 0,
                             e.bias_offset, e.bias_src[2] if e.bias_src else 0)

    assert len(table) == len(entries) * RESIDENT_ENTRY_BYTES, len(table)
    header = struct.pack("<QQQ", index_size, resident_size, len(entries))
    region = header + bytes(table) + bytes(strings)
    if len(region) > index_size:
        raise RuntimeError("Index region overflowed its reserved size")
    return region.ljust(index_size, b"\0")


def plan_experts(tensors, num_layers, num_experts):
    """Computes the per-expert block layout and validates it is uniform.

    Source routed tensors are expert-major -- shape [num_experts, K, M] -- so expert e's
    slice of each is one contiguous range. The output interleaves them: every expert gets
    all nine sub-tensors together, so a single read fills a UMA slot.
    """
    block, cursor = {}, 0
    per_layer_src = {}

    for layer in range(num_layers):
        for role, part in EXPERT_BLOCK_ORDER:
            full = (f"language_model.model.layers.{layer}"
                    f".experts.switch_glu.{role}.{part}")
            if full not in tensors:
                raise RuntimeError(f"Missing routed expert tensor: {full}")
            shard, dt, shape, start, nbytes = tensors[full]
            if shape[0] != num_experts:
                raise RuntimeError(
                    f"{full}: leading dim {shape[0]} != num_experts {num_experts}")
            if nbytes % num_experts:
                raise RuntimeError(f"{full}: {nbytes} not divisible by {num_experts}")
            slice_bytes = nbytes // num_experts

            key = f"{role}.{part}"
            if layer == 0:
                block[key] = {"offset": cursor, "size": slice_bytes,
                              "dtype": DTYPE_FROM_SAFETENSORS[dt],
                              "shape": list(shape[1:])}
                cursor += slice_bytes
            elif block[key]["size"] != slice_bytes:
                raise RuntimeError(
                    f"{full}: expert slice {slice_bytes} differs from layer 0 "
                    f"({block[key]['size']}); non-uniform experts are unsupported")
            per_layer_src.setdefault(layer, {})[key] = (shard, start, slice_bytes)

    return block, align_up(cursor), cursor, per_layer_src


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.done = total, label, 0
        self.start = time.time()
        self.last = 0.0

    def advance(self, n):
        self.done += n
        now = time.time()
        if now - self.last < 0.5 and self.done < self.total:
            return
        self.last = now
        elapsed = max(now - self.start, 1e-6)
        rate = self.done / elapsed / (1024 * 1024)
        pct = 100.0 * self.done / self.total if self.total else 100.0
        eta = (self.total - self.done) / (self.done / elapsed) if self.done else 0
        sys.stdout.write(
            f"\r    {self.label}: {pct:5.1f}%  {human(self.done)} / {human(self.total)}"
            f"  {rate:6.1f} MB/s  ETA {int(eta // 60):02d}:{int(eta % 60):02d}   ")
        sys.stdout.flush()

    def finish(self):
        self.advance(0)
        sys.stdout.write("\n")


def stream_into(source, out, shard, src_start, nbytes, dst_offset, progress):
    """Copies one contiguous source range to one contiguous output offset."""
    out.seek(dst_offset)
    written = 0
    for chunk in source.open_range(shard, src_start, nbytes):
        out.write(chunk)
        written += len(chunk)
        progress.advance(len(chunk))
    if written != nbytes:
        raise RuntimeError(
            f"{shard}: wrote {written} of {nbytes} bytes at src {src_start}")


def stream_scatter(source, out, shard, src_start, total_bytes,
                   slice_bytes, dst_offsets, progress):
    """Streams one source range and scatters fixed-size slices to many output offsets.

    Used for routed experts: a single sequential read of, say, a 127 MB gate_proj.weight
    is split into 128 slices written at stride intervals. This is what keeps the request
    count at ~1,300 instead of ~35,000.
    """
    idx, in_slice = 0, 0
    out.seek(dst_offsets[0])
    consumed = 0

    for chunk in source.open_range(shard, src_start, total_bytes):
        view = memoryview(chunk)
        while view:
            room = slice_bytes - in_slice
            take = min(room, len(view))
            out.write(view[:take])
            view = view[take:]
            in_slice += take
            consumed += take
            progress.advance(take)
            if in_slice == slice_bytes:
                idx += 1
                in_slice = 0
                if idx < len(dst_offsets):
                    out.seek(dst_offsets[idx])
                elif view:
                    raise RuntimeError(f"{shard}: more data than {len(dst_offsets)} slices")

    if consumed != total_bytes:
        raise RuntimeError(f"{shard}: scattered {consumed} of {total_bytes} bytes")
    if idx != len(dst_offsets):
        raise RuntimeError(f"{shard}: filled {idx} of {len(dst_offsets)} slices")


def sha256_file(path, progress=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK_BYTES)
            if not b:
                break
            h.update(b)
            if progress:
                progress.advance(len(b))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(cfg, expert_stride, num_layers, num_experts, files, index_sha):
    tc = cfg.get("text_config", cfg)
    rope = tc.get("rope_parameters", {})
    full_rope = rope.get("full_attention", {})
    swa_rope = rope.get("sliding_attention", {})
    layer_types = tc["layer_types"]

    quant = cfg.get("quantization", {})
    group = quant.get("group_size", 64)

    def slot(bits):
        return {"weightBits": bits, "scheme": "mlx_affine",
                "scaleType": "BF16", "biasType": "BF16", "groupSize": group}

    return {
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": {"streamingPresent": True, "turboQuantKV": False,
                  "aneSharedExpert": False},
        "modelID": "gemma-4-26b-a4b",
        "sourceRepo": REPO_ID,
        "sourceRevision": REVISION,
        "sourceSnapshotHash": index_sha,
        "numLayers": num_layers,
        "expertsPerLayer": num_experts,
        "expertStride": expert_stride,
        "arch": {
            "hiddenSize": tc["hidden_size"],
            "ffnIntermediate": tc["intermediate_size"],
            "moeIntermediateSize": tc["moe_intermediate_size"],
            "numHeads": tc["num_attention_heads"],
            "numKVHeads": tc["num_key_value_heads"],
            "numFullKVHeads": tc["num_global_key_value_heads"],
            "headDim": tc["head_dim"],
            "fullHeadDim": tc["global_head_dim"],
            "vocabSize": tc["vocab_size"],
            "slidingWindow": tc["sliding_window"],
            "finalLogitSoftcap": tc["final_logit_softcapping"],
            "ropeTheta": swa_rope.get("rope_theta", 10000.0),
            "fullRopeTheta": full_rope.get("rope_theta", 1000000.0),
            "partialRotaryFactor": full_rope.get("partial_rotary_factor", 0.25),
            "rmsNormEps": tc["rms_norm_eps"],
            "numLayers": num_layers,
            "numExperts": num_experts,
            "topKExperts": tc["top_k_experts"],
            "tieWordEmbeddings": tc["tie_word_embeddings"],
            "attentionKEqV": tc["attention_k_eq_v"],
            "hiddenActivation": tc["hidden_activation"],
            # 1 = full attention, 0 = sliding-window. Derived from the checkpoint, not
            # assumed: for this model it is layers 5, 11, 17, 23, 29.
            "fullAttentionLayerMask": [
                1 if t == "full_attention" else 0 for t in layer_types],
        },
        "quant": {
            "embedding": slot(quant.get("bits", 4)),
            "attention": slot(quant.get("bits", 4)),
            "router": slot(8),
            "sharedExpert": slot(quant.get("bits", 4)),
            "routedExpert": slot(quant.get("bits", 4)),
        },
        "files": files,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(source, out_dir, skip_hash=False, resume=False):
    out = Path(out_dir)
    partial = out.with_name(out.name + ".partial")

    if out.exists() and not resume:
        raise SystemExit(
            f"{out} already exists. Move it aside, or pass --resume to reuse "
            f"{partial} if a previous run was interrupted.")
    packed = partial / "packed_experts"
    packed.mkdir(parents=True, exist_ok=True)
    (partial / "tokenizer").mkdir(parents=True, exist_ok=True)

    # --- 1. Source metadata -------------------------------------------------
    print("[1/6] Reading checkpoint metadata...")
    cfg = json.loads(source.read_all(CONFIG_FILE))
    index_raw = source.read_all(INDEX_FILE)
    index_sha = hashlib.sha256(index_raw).hexdigest()
    index = json.loads(index_raw)
    print(f"    index sha256 {index_sha}")

    tc = cfg.get("text_config", cfg)
    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    print(f"    {num_layers} layers, {num_experts} experts/layer, "
          f"hidden {tc['hidden_size']}, vocab {tc['vocab_size']}")

    shards = sorted(set(index["weight_map"].values()))
    tensors = load_safetensors_headers(source, shards)

    # --- 2. Classify --------------------------------------------------------
    lm = {k: v for k, v in tensors.items() if k.startswith("language_model.")}
    routed = {k: v for k, v in lm.items() if ROUTED_MARKER in k}
    resident_src = {k: v for k, v in lm.items() if ROUTED_MARKER not in k}
    excluded = len(tensors) - len(lm)
    print(f"    {len(resident_src)} resident, {len(routed)} routed, "
          f"{excluded} multimodal (excluded)")

    # --- 3. Plan ------------------------------------------------------------
    print("[2/6] Planning layout...")
    entries = group_resident(resident_src)
    index_size, resident_size = plan_resident(entries)
    block, expert_stride, block_raw, per_layer_src = plan_experts(
        tensors, num_layers, num_experts)
    layer_bytes = expert_stride * num_experts

    print(f"    resident.bin   {human(resident_size)} "
          f"({len(entries)} tensors, {human(index_size)} index)")
    print(f"    expert block   {block_raw:,} B raw -> {expert_stride:,} B aligned")
    print(f"    per layer file {human(layer_bytes)}  x{num_layers} = "
          f"{human(layer_bytes * num_layers)}")

    # --- 4. Resident --------------------------------------------------------
    print("[3/6] Writing resident.bin...")
    resident_path = partial / "resident.bin"
    prog = Progress(resident_size - index_size, "resident")
    with open(resident_path, "r+b" if resident_path.exists() else "wb") as f:
        f.truncate(resident_size)
        f.seek(0)
        f.write(build_resident_index(entries, index_size, resident_size))
        for e in entries:
            stream_into(source, f, e.src[0], e.src[1], e.src[2], e.offset, prog)
            if e.scale_src:
                stream_into(source, f, e.scale_src[0], e.scale_src[1],
                            e.scale_src[2], e.scale_offset, prog)
            if e.bias_src:
                stream_into(source, f, e.bias_src[0], e.bias_src[1],
                            e.bias_src[2], e.bias_offset, prog)
    prog.finish()

    # --- 5. Experts ---------------------------------------------------------
    print("[4/6] Writing packed_experts/...")
    # Payload bytes per layer = every expert's nine sub-tensors, excluding stride padding.
    payload_per_layer = block_raw * num_experts
    prog = Progress(payload_per_layer * num_layers, "experts")

    for layer in range(num_layers):
        path = packed / f"layer_{layer:02d}.bin"
        marker = packed / f"layer_{layer:02d}.done"
        # A layer file is only trusted on resume if it was fully written last time. The
        # marker is what proves that -- correct size alone does not, since the file is
        # truncated to its final length before any bytes are streamed in.
        if resume and marker.exists() and path.exists() and path.stat().st_size == layer_bytes:
            prog.advance(payload_per_layer)
            continue
        with open(path, "wb") as f:
            f.truncate(layer_bytes)
            for role, part in EXPERT_BLOCK_ORDER:
                key = f"{role}.{part}"
                shard, start, slice_bytes = per_layer_src[layer][key]
                offsets = [e * expert_stride + block[key]["offset"]
                           for e in range(num_experts)]
                stream_scatter(source, f, shard, start, slice_bytes * num_experts,
                               slice_bytes, offsets, prog)
            f.flush()
            os.fsync(f.fileno())
        marker.touch()
    prog.finish()
    for m in packed.glob("*.done"):
        m.unlink()

    # --- 6. Tokenizer + manifest -------------------------------------------
    print("[5/6] Copying tokenizer...")
    for name, required in TOKENIZER_FILES:
        try:
            data = source.read_all(name)
        except (urllib.error.HTTPError, SystemExit) as ex:
            if required:
                raise SystemExit(f"Required tokenizer file {name} unavailable: {ex}")
            print(f"    (skipped optional {name})")
            continue
        (partial / "tokenizer" / name).write_bytes(data)
        print(f"    {name} ({len(data):,} B)")

    print("[6/6] Writing layout.json + manifest.json...")
    layout = {
        "expertStride": expert_stride,
        "numLayers": num_layers,
        "expertsPerLayer": num_experts,
        # Every expert block is byte-identical in structure, so the sub-tensor offsets are
        # described once instead of being enumerated 3,840 times (the old layout.json was
        # 6.9 MB of repetition).
        "expertBlock": {f"{r}.{p}": block[f"{r}.{p}"] for r, p in EXPERT_BLOCK_ORDER},
        "layers": [{"layer": i, "file": f"layer_{i:02d}.bin"}
                   for i in range(num_layers)],
    }
    (packed / "layout.json").write_text(json.dumps(layout, indent=2))

    files = {}
    hash_targets = [("resident.bin", partial / "resident.bin"),
                    ("packed_experts/layout.json", packed / "layout.json")]
    hash_targets += [(f"packed_experts/layer_{i:02d}.bin",
                      packed / f"layer_{i:02d}.bin") for i in range(num_layers)]

    if skip_hash:
        print("    (--skip-hash: recording sizes only)")
        for rel, path in hash_targets:
            files[rel] = {"size": path.stat().st_size, "sha256": ""}
    else:
        total = sum(p.stat().st_size for _, p in hash_targets)
        prog = Progress(total, "sha256  ")
        for rel, path in hash_targets:
            files[rel] = {"size": path.stat().st_size,
                          "sha256": sha256_file(path, prog)}
        prog.finish()

    manifest = build_manifest(cfg, expert_stride, num_layers, num_experts,
                              files, index_sha)
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Promote only once everything is on disk, so an interrupted run never leaves a
    # directory that looks complete.
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)

    total_bytes = sum(f["size"] for f in files.values())
    print(f"\nDone: {out}  ({human(total_bytes)})")
    print(f"  resident.bin        {human(files['resident.bin']['size'])}")
    print(f"  packed_experts/     {human(layer_bytes * num_layers)}")
    print(f"  expert stride       {expert_stride:,} B")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="gemma-4-26b-a4b.gturbo",
                    help="output .gturbo directory")
    ap.add_argument("--input", default=None,
                    help="local snapshot directory (default: stream from HuggingFace)")
    ap.add_argument("--token", default=None,
                    help="HuggingFace read token (or set HF_TOKEN)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse a previous interrupted <output>.partial")
    ap.add_argument("--skip-hash", action="store_true",
                    help="record file sizes without sha256 (faster, weaker manifest)")
    args = ap.parse_args()

    source = LocalSource(args.input) if args.input else RemoteSource(args.token)
    print(f"Source: {args.input if args.input else REPO_ID + ' @ ' + REVISION[:12]}")
    print(f"Output: {args.output}\n")

    try:
        convert(source, args.output, skip_hash=args.skip_hash, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
