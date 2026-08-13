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
import subprocess
import sys
import tempfile
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


def _w64devkit_asset(name):
    """Picks the x64 build out of a release's asset list.

    w64devkit stopped publishing .zip assets after v1.23.0: v2.0.0 and v2.1.0 ship a
    plain self-extracting .exe, and v2.2.0 onward ship <name>.7z.exe. Matching only
    ".zip" therefore selected nothing and every fresh install died with "No matching
    asset in latest release". That went unnoticed for a long time because this
    function returns early whenever C:\\w64devkit already exists -- it only ever
    failed on a clean machine, which is exactly where CI runs.
    """
    n = name.lower()
    if n.endswith(".sig"):          # detached signatures, not archives
        return False
    if not n.startswith("w64devkit-x64-"):
        return False                # skip x86, source.tar, and fortran variants
    return n.endswith(".exe") or n.endswith(".zip")


def install_w64devkit(force=False):
    gxx = W64DEVKIT_DIR / "bin" / "g++.exe"
    if gxx.exists() and not force:
        print(f"[skip] w64devkit already present at {W64DEVKIT_DIR}")
        return

    print("[w64devkit] resolving latest release...")
    name, url = _latest_release_asset("skeeto/w64devkit", _w64devkit_asset)
    print(f"[w64devkit] {name}")
    blob = _download(url)

    # Both archive kinds contain a single top-level w64devkit/ directory, so they
    # extract into the parent of W64DEVKIT_DIR.
    parent = W64DEVKIT_DIR.parent
    print(f"[w64devkit] extracting to {W64DEVKIT_DIR}...")

    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(parent)
    else:
        installer = Path(tempfile.gettempdir()) / name
        installer.write_bytes(blob)
        try:
            # 7-Zip self-extractor: -y assumes yes, -o sets the output directory.
            #
            # The return code is deliberately ignored. This SFX exits 2 even on a
            # complete and perfectly working extraction, and prints nothing at all
            # to stdout or stderr, so the exit status carries no signal. Success is
            # decided below by whether g++.exe actually landed and runs.
            subprocess.run([str(installer), "-y", f"-o{parent}"],
                           check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        finally:
            installer.unlink(missing_ok=True)

    if not gxx.exists():
        raise RuntimeError(
            f"Extraction finished but {gxx} is missing.\n"
            f"    If {parent} is not writable by this account, re-run from an\n"
            f"    elevated shell, or extract {name} by hand so that {gxx} exists."
        )

    # Verify it actually runs -- an extracted-but-broken toolchain otherwise only
    # shows up much later, as a confusing CMake compiler-check failure.
    probe = subprocess.run([str(gxx), "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"{gxx} exists but failed to run:\n{probe.stderr.strip()}")
    version = probe.stdout.splitlines()[0] if probe.stdout else "unknown version"

    print(f"[w64devkit] OK -> {gxx} ({version})")
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
