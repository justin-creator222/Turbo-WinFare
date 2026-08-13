"""
Fetches the build/runtime toolchain this project needs:

  * w64devkit  -> C:\\w64devkit          (g++ / ninja host toolchain)
  * DXC        -> build/dxcompiler.dll   (Shader Model 6.6 HLSL compiler)
                  build/dxil.dll         (signing library, required to create PSOs)

DXC is not optional. Every compute kernel in shaders/ relies on SM 6.6 wave intrinsics
(WaveActiveSum / WaveActiveMax / WaveReadLaneFirst) for its cross-lane reductions, and the
engine deliberately has no cs_5_0 fallback -- the old one #define'd those intrinsics to
identity, which silently reduced every wave reduction to a single lane's partial value.

Usage:
    python tools/download_toolchain.py            # everything that is missing
    python tools/download_toolchain.py --dxc      # just DXC
    python tools/download_toolchain.py --w64devkit
    python tools/download_toolchain.py --force    # re-download even if present
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
W64DEVKIT_DIR = Path("C:/w64devkit")
DXC_DEST_DIRS = [REPO_ROOT / "build", REPO_ROOT]
DXC_FILES = ("dxcompiler.dll", "dxil.dll")

UA = {"User-Agent": "turbo-winfare-toolchain"}


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _download(url):
    print(f"    downloading {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _latest_release_asset(repo, predicate):
    """Returns (asset_name, download_url) for the first asset matching predicate."""
    data = _get_json(f"https://api.github.com/repos/{repo}/releases/latest")
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if predicate(name):
            return name, asset["browser_download_url"]
    raise RuntimeError(f"No matching asset in latest release of {repo}")


def install_w64devkit(force=False):
    gxx = W64DEVKIT_DIR / "bin" / "g++.exe"
    if gxx.exists() and not force:
        print(f"[skip] w64devkit already present at {W64DEVKIT_DIR}")
        return

    print("[w64devkit] resolving latest release...")
    name, url = _latest_release_asset("skeeto/w64devkit", lambda n: n.endswith(".zip"))
    print(f"[w64devkit] {name}")
    blob = _download(url)

    print(f"[w64devkit] extracting to {W64DEVKIT_DIR}...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        # The archive contains a top-level w64devkit/ directory.
        zf.extractall(W64DEVKIT_DIR.parent)

    if not gxx.exists():
        raise RuntimeError(f"Extraction finished but {gxx} is missing")
    print(f"[w64devkit] OK -> {gxx}")
    print("           NOTE: add C:\\w64devkit\\bin to PATH, or g++ cannot find 'as'.")


def install_dxc(force=False):
    already = [d for d in DXC_DEST_DIRS
               if all((d / f).exists() for f in DXC_FILES)]
    if already and not force:
        print(f"[skip] DXC already present in {already[0]}")
        return

    print("[dxc] resolving latest DirectXShaderCompiler release...")
    name, url = _latest_release_asset(
        "microsoft/DirectXShaderCompiler",
        lambda n: n.lower().startswith("dxc_") and n.lower().endswith(".zip"),
    )
    print(f"[dxc] {name}")
    blob = _download(url)

    # The archive lays binaries out under bin/x64/.
    wanted = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for entry in zf.namelist():
            base = os.path.basename(entry).lower()
            if base in DXC_FILES and "/x64/" in entry.replace("\\", "/").lower():
                wanted[base] = zf.read(entry)

        missing = set(DXC_FILES) - set(wanted)
        if missing:
            raise RuntimeError(f"DXC archive did not contain x64 {sorted(missing)}")

    for dest in DXC_DEST_DIRS:
        dest.mkdir(parents=True, exist_ok=True)
        for fname, data in wanted.items():
            out = dest / fname
            out.write_bytes(data)
            print(f"[dxc] wrote {out} ({len(data):,} bytes)")

    print("[dxc] OK -- shaders will now compile as cs_6_6.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w64devkit", action="store_true", help="install only w64devkit")
    ap.add_argument("--dxc", action="store_true", help="install only DXC")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    # No selector means "everything".
    do_all = not (args.w64devkit or args.dxc)

    try:
        if do_all or args.w64devkit:
            install_w64devkit(force=args.force)
        if do_all or args.dxc:
            install_dxc(force=args.force)
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
