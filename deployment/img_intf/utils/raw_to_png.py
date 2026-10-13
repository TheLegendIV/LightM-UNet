"""Render board class-map raws as colour PNGs.

Per <case>.raw in --raw-dir (as written by send_image.py / collect_results.py) writes <case>_board.png.
With --images-dir / --labels-dir also writes <case>_panel.png: input | board | ground truth | mismatch.
With --images-dir and --overlay also writes <case>_overlay.png: the class map blended over the input
(background stays transparent).

  python deployment/img_intf/utils/raw_to_png.py [--raw-dir D] [--png-dir D] [--images-dir D] [--labels-dir D] [--overlay] [--alpha A] [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IFACE = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = IFACE / "output" / "s12_512_bilinear_mixed68" / "raw"
SIZE = 512

# class id -> RGB (0 = background)
PALETTE = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25]], dtype=np.uint8)
MISMATCH_RGB = (255, 0, 255)


def colorize(cls: np.ndarray) -> np.ndarray:
    return PALETTE[np.minimum(cls, len(PALETTE) - 1)]


def overlay(gray: np.ndarray, cls: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """(H, W) uint8 gray + (H, W) class ids -> (H, W, 3) uint8 with classes > 0 blended in colour."""
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    fg = cls > 0
    rgb[fg] = (1.0 - alpha) * rgb[fg] + alpha * colorize(cls)[fg].astype(np.float32)
    return rgb.astype(np.uint8)


def titled(rgb: np.ndarray, title: str) -> Image.Image:
    img = Image.fromarray(rgb)
    canvas = Image.new("RGB", (img.width, img.height + 18), (32, 32, 32))
    canvas.paste(img, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="board outputs <case>.raw (case = image stem)")
    ap.add_argument("--png-dir", help="default: sibling 'png' of --raw-dir")
    ap.add_argument("--images-dir", help="source PNGs <case>.png, for the panel")
    ap.add_argument("--labels-dir", help="ground-truth PNGs <case>.png (the _0000 suffix is stripped)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overlay", action="store_true", help="also write <case>_overlay.png (needs --images-dir)")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity of the class colours")
    a = ap.parse_args()

    raw_dir = Path(a.raw_dir)
    png_dir = Path(a.png_dir) if a.png_dir else raw_dir.parent / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    raws = sorted(raw_dir.glob("*.raw"))[: a.limit]
    if not raws:
        raise FileNotFoundError(f"no board outputs in {raw_dir}")

    for raw in raws:
        stem = raw.stem
        board = np.fromfile(raw, dtype=np.uint8).reshape(SIZE, SIZE)
        board_rgb = colorize(board)
        Image.fromarray(board_rgb).save(png_dir / f"{stem}_board.png")

        src = Path(a.images_dir) / f"{stem}.png" if a.images_dir else None
        if a.overlay and src and src.is_file():
            gray = np.asarray(Image.open(src).convert("L"))
            Image.fromarray(overlay(gray, board, a.alpha)).save(png_dir / f"{stem}_overlay.png")

        gt_path = Path(a.labels_dir) / f"{stem[:-5] if stem.endswith('_0000') else stem}.png" if a.labels_dir else None
        if not (src and src.is_file() and gt_path and gt_path.is_file()):
            continue
        inp = np.asarray(Image.open(src).convert("L"))
        gt = np.asarray(Image.open(gt_path))
        bad = board != gt
        gt_rgb = colorize(gt)
        diff_rgb = (gt_rgb // 3).copy()
        diff_rgb[bad] = MISMATCH_RGB
        tiles = [titled(np.repeat(inp[..., None], 3, axis=2), "input"), titled(board_rgb, "board"),
                 titled(gt_rgb, "ground truth"), titled(diff_rgb, f"mismatch: {int(bad.sum())} px")]
        panel = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
        x = 0
        for t in tiles:
            panel.paste(t, (x, 0))
            x += t.width + 4
        panel.save(png_dir / f"{stem}_panel.png")

    print(f"wrote PNGs for {len(raws)} case(s) to {png_dir}")
    print("classes: 0 black, 1 red, 2 green, 3 blue, 4 yellow; mismatch magenta")


if __name__ == "__main__":
    main()
