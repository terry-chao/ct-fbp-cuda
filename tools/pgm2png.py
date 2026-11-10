#!/usr/bin/env python3
"""PGM -> PNG 转换工具（开发辅助，需 Pillow）。

用法:
  python tools/pgm2png.py input.pgm [output.png] [--window vmin vmax]

--window 指定显示窗口（体模内部结构对比度只有 1~3%，窄窗口才能看清），
缺省为图像自身的 [min, max]。
"""
import sys

import numpy as np
from PIL import Image


def read_pgm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P2"):
            sys.exit(f"not a PGM file: {path}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        if magic == b"P5":
            dtype = np.uint8 if maxval < 256 else np.dtype(">u2")
            data = np.frombuffer(f.read(), dtype=dtype)
        else:
            data = np.array(f.read().split(), dtype=int)
        return data.reshape(h, w).astype(np.float64) / maxval


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    src = args[0]
    dst = args[1] if len(args) > 1 else src.rsplit(".", 1)[0] + ".png"

    img = read_pgm(src)
    if "--window" in sys.argv:
        i = sys.argv.index("--window")
        vmin, vmax = float(sys.argv[i + 1]), float(sys.argv[i + 2])
    else:
        vmin, vmax = img.min(), img.max()
    win = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    Image.fromarray((win * 255 + 0.5).astype(np.uint8)).save(dst)
    print(f"{src} -> {dst}  window=[{vmin:g}, {vmax:g}]")


if __name__ == "__main__":
    main()
