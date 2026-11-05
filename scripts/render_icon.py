"""Render VisionLab icons with signed-distance anti-aliasing."""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

C0 = np.array([216, 200, 255], dtype=np.float32) / 255.0
C1 = np.array([139, 92, 246], dtype=np.float32) / 255.0
C2 = np.array([59, 7, 100], dtype=np.float32) / 255.0
WHITE = np.array([248, 250, 252], dtype=np.float32) / 255.0
CYAN = np.array([34, 211, 238], dtype=np.float32) / 255.0
PUPIL = np.array([15, 23, 42], dtype=np.float32) / 255.0
IRIS_EDGE = np.array([76, 29, 149], dtype=np.float32) / 255.0


def length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y)


def cover(sd: np.ndarray, aa: float) -> np.ndarray:
    return np.clip(0.5 - sd / max(aa, 1e-4), 0.0, 1.0)


def sd_round_box(x: np.ndarray, y: np.ndarray, hx: float, hy: float, r: float) -> np.ndarray:
    ax = np.abs(x) - hx + r
    ay = np.abs(y) - hy + r
    return length(np.maximum(ax, 0.0), np.maximum(ay, 0.0)) + np.minimum(np.maximum(ax, ay), 0.0) - r


def sd_circle(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    return length(x, y) - r


def sd_segment(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    pax, pay = px - ax, py - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay + 1e-8
    h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
    return length(pax - bax * h, pay - bay * h)


def blend(dst: np.ndarray, color: np.ndarray, a: np.ndarray) -> None:
    a3 = a[..., None]
    dst[..., :3] = dst[..., :3] * (1.0 - a3) + color * a3
    dst[..., 3] = dst[..., 3] * (1.0 - a) + a


def paint_mark(size: int, rounded: bool) -> Image.Image:
    s = float(size)
    aa = 1.35 if size < 48 else 1.8
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Pixel centers
    x = xx + 0.5 - s / 2.0
    y = yy + 0.5 - s / 2.0
    half = s / 2.0 - 0.5

    t = np.clip(((xx + 0.5) * 0.55 + (yy + 0.5) * 0.45) / s, 0.0, 1.0)
    mid = t < 0.45
    u = np.where(mid, t / 0.45, (t - 0.45) / 0.55)
    grad = np.empty((size, size, 3), dtype=np.float32)
    grad[mid] = (C0 + (C1 - C0) * u[..., None])[mid]
    grad[~mid] = (C1 + (C2 - C1) * u[..., None])[~mid]

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., :3] = grad
    if rounded:
        tile = cover(sd_round_box(x, y, half, half, s * 0.25), aa)
        rgba[..., 3] = tile
        rgba[..., :3] *= tile[..., None]
    else:
        rgba[..., 3] = 1.0

    # Inner rim
    rim = cover(np.abs(sd_round_box(x, y, half - s * 0.035, half - s * 0.035, s * 0.20)) - s * 0.012, aa)
    blend(rgba, WHITE, rim * 0.22 * rgba[..., 3])

    inset = s * 0.22
    arm = s * 0.155
    rad = s * 0.028
    hx = half
    corners = [
        ((-hx + inset, -hx + inset), (-hx + inset, -hx + inset + arm), (-hx + inset + arm, -hx + inset)),
        ((hx - inset, -hx + inset), (hx - inset, -hx + inset + arm), (hx - inset - arm, -hx + inset)),
        ((-hx + inset, hx - inset), (-hx + inset, hx - inset - arm), (-hx + inset + arm, hx - inset)),
        ((hx - inset, hx - inset), (hx - inset, hx - inset - arm), (hx - inset - arm, hx - inset)),
    ]
    bracket = np.full((size, size), 1e6, dtype=np.float32)
    for origin, v, hseg in corners:
        bracket = np.minimum(bracket, sd_segment(x, y, origin[0], origin[1], v[0], v[1]) - rad)
        bracket = np.minimum(bracket, sd_segment(x, y, origin[0], origin[1], hseg[0], hseg[1]) - rad)
    blend(rgba, WHITE, cover(bracket, aa) * rgba[..., 3])

    # Eye ring (ellipse in scaled space)
    rx, ry = s * 0.24, s * 0.138
    eye = np.abs(length(x / rx, y / ry) - 1.0) * min(rx, ry) - s * 0.018
    blend(rgba, WHITE, cover(eye, aa) * rgba[..., 3])

    iris_r = s * 0.105
    ix = x - s * 0.02
    iy = y - s * 0.025
    iris_d = length(ix, iy) / iris_r
    iris_a = cover(sd_circle(x, y, iris_r), aa) * rgba[..., 3]
    iris_col = IRIS_EDGE + (CYAN - IRIS_EDGE) * np.clip(1.0 - iris_d, 0.0, 1.0)[..., None]
    iris_col = iris_col + (WHITE - iris_col) * (np.clip(1.0 - iris_d * 1.6, 0.0, 1.0) ** 2)[..., None]
    blend(rgba, iris_col, iris_a)

    blend(rgba, PUPIL, cover(sd_circle(x, y, s * 0.038), aa) * rgba[..., 3])
    blend(
        rgba,
        WHITE,
        cover(sd_circle(x - s * 0.032, y + s * 0.03, s * 0.02), aa) * rgba[..., 3],
    )

    out = np.clip(rgba, 0, 1)
    px = np.empty((size, size, 4), dtype=np.uint8)
    px[..., :3] = (out[..., :3] * 255.0 + 0.5).astype(np.uint8)
    px[..., 3] = (out[..., 3] * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(px, "RGBA")


def png_bytes(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def write_ico(path: Path, images: list[Image.Image]) -> None:
    blobs = [png_bytes(im.convert("RGBA")) for im in images]
    sizes = [im.size[0] for im in images]
    offset = 6 + 16 * len(blobs)
    entries = b""
    body = b""
    for s, blob in zip(sizes, blobs):
        wb = 0 if s >= 256 else s
        entries += struct.pack("<BBBBHHII", wb, wb, 0, 0, 1, 32, len(blob), offset)
        body += blob
        offset += len(blob)
    path.write_bytes(struct.pack("<HHH", 0, 1, len(blobs)) + entries + body)


def main() -> None:
    def export(size: int, rounded: bool) -> Image.Image:
        return paint_mark(size * 2, rounded).resize((size, size), Image.Resampling.LANCZOS)

    export(512, True).save(ASSETS / "app-icon.png", format="PNG", optimize=True)
    export(256, False).save(ASSETS / "app-icon-win.png", format="PNG", optimize=True)
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    write_ico(ASSETS / "app.ico", [export(s, False) for s in sizes])
    print("wrote icons", sizes)


if __name__ == "__main__":
    main()
