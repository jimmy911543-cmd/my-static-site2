# -*- coding: utf-8 -*-
"""Prepare web images: resize travel photos, knock out studio-white ikebana backgrounds."""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(r"c:\Users\jimmy\OneDrive\Desktop\ling")
OUT = ROOT / "site" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

KNOCKOUT = {
    "花.jfif": "flower-grass.png",
    "花1.jfif": "flower-celosia.png",
    "夏至風動.jpg": "flower-summer.png",
    "S__127606789_0.jpg": "flower-arc.png",
    "S__127606790_0.jpg": "flower-purple.png",
    "S__127606792_0.jpg": "flower-pair.png",
    "S__127606793_0.jpg": "flower-fern.png",
    "S__127606794_0.jpg": "flower-gloriosa.png",
    "S__127606796_0.jpg": "flower-amaranth.png",
    "494039927_9699584523465071_3509318905873663089_n.jpg": "flower-iris.png",
}

PHOTOS = {
    "新年.jpg": "flower-newyear.jpg",
    "S__127606791_0.jpg": "flower-bird.jpg",
    "S__127606795_0.jpg": "flower-tropic.jpg",
    "S__229318670.jpg": "flower-field.jpg",
    "簡玲照片.jpg": "travel-boat.jpg",
    "20230917_222418+0800-669949_0.jpg": "travel-tea.jpg",
    "36937371_2199282186752754_7288049516895272960_n_0.jpg": "travel-cliff.jpg",
    "75366345_2643233965764048_2966974965833793536_n_0.jpg": "travel-stone.jpg",
    "UNADJUSTEDNONRAW_thumb_1e1d.jpg": "travel-sea.jpg",
}


def limit(im, max_side):
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return im


def _bg_color(rgb):
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0).astype(np.int16)
    lum = border.mean(axis=1)
    sat = border.max(axis=1) - border.min(axis=1)
    sample = border[(lum > np.percentile(lum, 65)) & (sat < 28)]
    if len(sample) < 20:
        sample = border
    return np.median(sample, axis=0)


def _flood(near):
    h, w = near.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    def push(y, x):
        if 0 <= y < h and 0 <= x < w and mask[y, x] == 0 and near[y, x]:
            mask[y, x] = 1
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)
    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)
    return mask


def knockout(im, hard=10, punch_holes=False):
    rgb = np.asarray(im).astype(np.int16)
    h, w, _ = rgb.shape
    bg = _bg_color(rgb)
    delta = np.abs(rgb - bg).max(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    near = (delta <= hard) & (sat <= 16)
    mask = _flood(near)
    if punch_holes:
        mask[(delta <= hard - 2) & (sat <= 14)] = 1

    alpha = np.full((h, w), 255, np.uint8)
    alpha[mask == 1] = 0
    # Nibble only the pale fringe that touches the removed background.
    zero = Image.fromarray(np.where(alpha == 0, 255, 0).astype(np.uint8), "L")
    touch = np.asarray(zero.filter(ImageFilter.MaxFilter(3))) > 0
    lum = rgb.mean(axis=2)
    fringe = touch & (alpha > 0) & (sat <= 18) & ((lum >= 236) | (delta <= hard + 6))
    alpha[fringe] = 0
    alpha_im = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=0.45))
    rgba = im.convert("RGBA")
    rgba.putalpha(alpha_im)
    ys, xs = np.where(np.asarray(alpha_im) > 18)
    if len(xs) == 0:
        return rgba
    pad = 24
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )
    return rgba.crop(box)


def cut_portrait(src, dest):
    im = Image.open(src).convert("RGB")
    im.thumbnail((1200, 1800), Image.Resampling.LANCZOS)
    rgb = np.asarray(im).astype(np.int16)
    h, w, _ = rgb.shape
    bg = _bg_color(rgb)
    delta = np.abs(rgb - bg).max(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    lum = rgb.mean(axis=2)
    # Studio backdrop shifts from light gray to a darker gray; both are neutral.
    bg_like = (sat <= 13) & (lum >= 148) & (lum <= 242)
    alpha = np.where(bg_like, 0, 255).astype(np.uint8)
    for _ in range(4):
        zero = alpha == 0
        padded = np.pad(zero, 1, constant_values=False)
        touch = np.zeros_like(zero)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                touch |= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        rim = touch & (alpha > 0) & (sat <= 16) & (lum >= 150) & (lum <= 242)
        neighbors = np.zeros((h, w), dtype=np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbors += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        crumbs = (alpha > 0) & (neighbors >= 6)
        alpha[rim | crumbs] = 0
    alpha_im = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=0.6))
    rgba = im.convert("RGBA")
    rgba.putalpha(alpha_im)
    ys, xs = np.where(np.asarray(alpha_im) > 24)
    pad = 16
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad),
        min(h, int(ys.max()) + pad),
    )
    cut = rgba.crop(box)
    cut.save(dest, optimize=True)
    print(f"portrait {cut.size} bg {bg}")


def main():
    for src, dest in KNOCKOUT.items():
        im = limit(Image.open(ROOT / src), 1500)
        path = OUT / dest.replace(".png", ".jpg")
        im.save(path, quality=86, optimize=True)
        print(f"flower {path.name} {im.size}")

    for src, dest in PHOTOS.items():
        im = limit(Image.open(ROOT / src), 1600)
        path = OUT / dest
        im.save(path, quality=84, optimize=True)
        print(f"jpg {dest} {im.size}")

    cut_portrait(ROOT / "3O5A0170_0.jpg", OUT / "portrait.png")


if __name__ == "__main__":
    main()
