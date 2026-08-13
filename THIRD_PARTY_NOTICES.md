# Third-party software and model terms

Turbo-WinFare's own source and documentation are licensed under the
[Apache License 2.0](LICENSE). That license applies to this repository's source only. It does
not relicense model weights, the upstream work this is derived from, or any third-party
component listed below.

This file records the dependency review performed on 2026-08-13. It is an attribution aid, not
legal advice. Anyone distributing a compiled build must also preserve the license and NOTICE
material required by the exact component versions included in that build.

## Upstream work

Turbo-WinFare is a derivative work of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey Mikhaylov, licensed under
the Apache License 2.0. The Swift/Metal sources are **not** vendored here; the port is an
independent C++23 / Direct3D 12 implementation that retains the upstream `.gturbo` format and
forward-pass semantics. See [NOTICE](NOTICE) for the attribution and the notice of modification
required by Apache-2.0 §4(b).

## Model weights

**Model weights are not included in this repository, and must not be redistributed with
releases of it.**

[`tools/convert_hf_to_gturbo.py`](tools/convert_hf_to_gturbo.py) downloads the pinned revision
`0d77464eeb233a2da68ebf9d7dc4edaac7db956d` of
[`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)
and repacks it locally into a `.gturbo` bundle on your machine. Its model card describes it as
an Apache-2.0 quantization of Google's Gemma 4 26B-A4B instruction checkpoint; Google publishes
Gemma 4 under the [Apache License 2.0](https://ai.google.dev/gemma/apache_2).

Downloaded weights, and any `.gturbo` bundle built from them, remain a separate work governed by
their source terms. `.gturbo` directories are excluded by [`.gitignore`](.gitignore) for exactly
this reason.

The bundle also contains the checkpoint's `tokenizer.json`, `tokenizer_config.json`, and
`chat_template.jinja`, copied through unchanged and governed by the same terms.

## Runtime dependencies

The engine links only against components shipped with Windows:

| Component | Source | Terms |
| --- | --- | --- |
| `d3d12`, `dxgi`, `d3dcompiler`, `dxguid`, `ws2_32` | Windows SDK import libraries | Microsoft Windows SDK license; system components, not redistributed here |
| `dxcompiler.dll`, `dxil.dll` | [microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler) release assets | DXC is LLVM/University of Illinois Open Source License with Microsoft additions; `dxil.dll` is a Microsoft-signed binary redistributable under its own terms. Fetched at build time by [`tools/download_toolchain.py`](tools/download_toolchain.py); **never committed to this repository**. Check the exact release's terms before redistributing them in a binary package. |

No third-party source is vendored into the engine. [`CMakeLists.txt`](CMakeLists.txt) declares no
external package dependency beyond the system libraries above; every file under `src/`,
`include/`, `shaders/`, `gui/`, `tools/`, and `tests/` is original to this project.

## Build-time tools

These are used to build the software and are not linked into or distributed with it:

| Tool | Source | Terms |
| --- | --- | --- |
| w64devkit (MinGW-w64 GCC toolchain) | [skeeto/w64devkit](https://github.com/skeeto/w64devkit) | Unlicense for the distribution; bundled GCC/binutils are GPL, with the GCC Runtime Library Exception covering compiled output |
| CMake | Kitware | BSD-3-Clause |
| Ninja | ninja-build | Apache-2.0 |
| Python 3 | Python Software Foundation | PSF License |

`tools/convert_hf_to_gturbo.py` and `tools/download_toolchain.py` use only the Python standard
library -- no third-party Python packages are required to fetch the toolchain or build a bundle.

## Binary distribution

If you publish a compiled release of Turbo-WinFare:

* Include [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
* Include the license text for the exact DXC release whose `dxcompiler.dll` / `dxil.dll` you
  ship, and confirm that release permits redistribution.
* Do **not** include model weights or a prebuilt `.gturbo` bundle.
