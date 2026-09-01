# Build assets/turbo-winfare.ico from the project logo.
#
# The generated .ico is committed, so a normal build never runs this -- it only needs
# re-running when the logo artwork changes.
#
#   powershell -ExecutionPolicy Bypass -File tools/make_icon.ps1
#
# Source is docs/assets/logo.png. It used to be read from gui/, where it was a
# byte-identical duplicate that nothing in the GUI referenced -- 575 KB copied into
# build/gui on every build for no one.
# Note the file is really a JPEG (it starts with the JFIF magic), which
# is why this decodes through System.Drawing rather than a PNG reader.
#
# Two things the artwork forces on us:
#
#   * The logo sits on a large dark canvas with an ambient glow, and only the middle ~75% is
#     the actual squircle tile. Shipping the whole canvas would waste most of a 16x16 icon on
#     empty background, so CROP_* below is the measured bounding box of the glowing border.
#   * That border is a rounded rect of radius ~170 at the cropped scale. We cut the corners
#     to transparent along the same curve; a mismatched radius reads as a doubled corner.
#
# Downsampling is a plain box filter over the exact source rectangle for each destination
# pixel. Graphics.DrawImage's bicubic samples only a 4x4 neighbourhood, which for a 764 -> 16
# reduction throws away ~99% of the source and aliases badly.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$repo = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $repo 'docsssets\logo.png'
$out  = Join-Path $repo 'assets\turbo-winfare.ico'

# Measured bounding box of the squircle border within the 1024x1024 artwork, and the corner
# radius of that border, in the cropped coordinate space.
$CROP_X = 130
$CROP_Y = 130
$CROP_S = 764
$RADIUS = 170.0

# Every entry is an uncompressed DIB, including 256x256. Windows itself accepts PNG-compressed
# entries, but plenty of icon consumers still assume a BITMAPINFOHEADER and render a PNG payload
# as noise -- .NET Framework's own System.Drawing.Icon is one of them, which also means an
# all-DIB file is one we can actually read back and verify. The cost is ~390 KB instead of
# ~240 KB, which is nothing next to the executable.
$SIZES = @(16, 20, 24, 32, 40, 48, 64, 128, 256)

# ---------------------------------------------------------------- source pixels

$bmp = [System.Drawing.Bitmap]::FromFile($src)
if ($bmp.Width -lt ($CROP_X + $CROP_S) -or $bmp.Height -lt ($CROP_Y + $CROP_S)) {
    throw "Logo is $($bmp.Width)x$($bmp.Height); crop box $CROP_X,$CROP_Y +$CROP_S does not fit."
}
$rect = New-Object System.Drawing.Rectangle 0, 0, $bmp.Width, $bmp.Height
$data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                      [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$stride = $data.Stride
$srcPix = New-Object byte[] ($stride * $bmp.Height)
[System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $srcPix, 0, $srcPix.Length)
$bmp.UnlockBits($data)
$bmp.Dispose()

# ---------------------------------------------------------------- per-size rendering

# Coverage of one destination pixel by the rounded rect, sampled 4x4. Straight (not
# premultiplied) alpha -- that is what both the ICO DIB and the PNG entries expect.
function Get-CornerAlpha([int]$px, [int]$py, [double]$n, [double]$r) {
    $inner = $n - $r          # centre coordinate of the far corner arcs
    $hits = 0
    for ($sy = 0; $sy -lt 4; $sy++) {
        $y = $py + ($sy + 0.5) / 4.0
        # Distance from the nearest corner-arc centre, along each axis.
        $dy = 0.0
        if ($y -lt $r) { $dy = $r - $y } elseif ($y -gt $inner) { $dy = $y - $inner }
        for ($sx = 0; $sx -lt 4; $sx++) {
            $x = $px + ($sx + 0.5) / 4.0
            $dx = 0.0
            if ($x -lt $r) { $dx = $r - $x } elseif ($x -gt $inner) { $dx = $x - $inner }
            if ([Math]::Sqrt($dx * $dx + $dy * $dy) -le $r) { $hits++ }
        }
    }
    return [int][Math]::Round(255.0 * $hits / 16.0)
}

# Box-filter the crop down to n x n and apply the rounded-corner alpha. Returns BGRA,
# top-down, stride n*4.
function New-Rendition([int]$n) {
    $px = New-Object byte[] ($n * $n * 4)
    $r  = $RADIUS * $n / $CROP_S
    for ($dy = 0; $dy -lt $n; $dy++) {
        $y0 = $CROP_Y + [int][Math]::Floor($dy * $CROP_S / $n)
        $y1 = $CROP_Y + [int][Math]::Floor(($dy + 1) * $CROP_S / $n)
        if ($y1 -le $y0) { $y1 = $y0 + 1 }
        for ($dx = 0; $dx -lt $n; $dx++) {
            $x0 = $CROP_X + [int][Math]::Floor($dx * $CROP_S / $n)
            $x1 = $CROP_X + [int][Math]::Floor(($dx + 1) * $CROP_S / $n)
            if ($x1 -le $x0) { $x1 = $x0 + 1 }

            $b = 0; $g = 0; $rr = 0; $cnt = 0
            for ($sy = $y0; $sy -lt $y1; $sy++) {
                $row = $sy * $stride
                for ($sx = $x0; $sx -lt $x1; $sx++) {
                    $i = $row + $sx * 4
                    $b += $srcPix[$i]; $g += $srcPix[$i + 1]; $rr += $srcPix[$i + 2]
                    $cnt++
                }
            }
            $o = ($dy * $n + $dx) * 4
            $px[$o]     = [byte][int][Math]::Round($b / $cnt)
            $px[$o + 1] = [byte][int][Math]::Round($g / $cnt)
            $px[$o + 2] = [byte][int][Math]::Round($rr / $cnt)
            $px[$o + 3] = [byte](Get-CornerAlpha $dx $dy $n $r)
        }
    }
    return ,$px
}

# ---------------------------------------------------------------- ICO entry encoding

# BITMAPINFOHEADER + bottom-up BGRA + an all-zero AND mask. biHeight is doubled because the
# header describes the XOR and AND bitmaps together, even though the mask is unused at 32bpp.
function ConvertTo-DibEntry([byte[]]$px, [int]$n) {
    $maskStride = [int][Math]::Floor(($n + 31) / 32) * 4
    $ms = New-Object System.IO.MemoryStream
    $w  = New-Object System.IO.BinaryWriter $ms
    $w.Write([uint32]40); $w.Write([int32]$n); $w.Write([int32]($n * 2))
    $w.Write([uint16]1);  $w.Write([uint16]32)
    $w.Write([uint32]0);  $w.Write([uint32]($n * $n * 4))
    $w.Write([uint32]0);  $w.Write([uint32]0); $w.Write([uint32]0); $w.Write([uint32]0)
    for ($y = $n - 1; $y -ge 0; $y--) { $w.Write($px, $y * $n * 4, $n * 4) }
    $w.Write((New-Object byte[] ($maskStride * $n)))
    $w.Flush()
    return ,$ms.ToArray()
}

# ---------------------------------------------------------------- assemble

$payloads = @()
foreach ($n in $SIZES) {
    Write-Host "  rendering ${n}x${n}"
    $payloads += , (ConvertTo-DibEntry (New-Rendition $n) $n)
}

New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
$fs = [System.IO.File]::Create($out)
$w  = New-Object System.IO.BinaryWriter $fs
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]$SIZES.Count)

$offset = 6 + 16 * $SIZES.Count
for ($i = 0; $i -lt $SIZES.Count; $i++) {
    $n = $SIZES[$i]
    # 256 is stored as 0; the field is a single byte.
    $dim = if ($n -ge 256) { 0 } else { $n }
    $w.Write([byte]$dim); $w.Write([byte]$dim); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([uint16]1);  $w.Write([uint16]32)
    $w.Write([uint32]$payloads[$i].Length); $w.Write([uint32]$offset)
    $offset += $payloads[$i].Length
}
foreach ($p in $payloads) { $w.Write($p) }
$w.Flush(); $fs.Close()

Write-Host "wrote $out ($((Get-Item $out).Length) bytes, $($SIZES.Count) sizes)"
