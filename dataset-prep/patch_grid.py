from typing import List, Tuple

Box = Tuple[int, int, int, int]


def split_boundaries(size: int, n: int) -> List[int]:
    """n+1 nearly-equal boundaries spanning [0, size]. Adjacent boundaries
    differ by at most 1px when size is not evenly divisible by n."""
    return [round(i * size / n) for i in range(n + 1)]


def layer1_boxes(width: int, height: int, cols: int, rows: int) -> List[List[Box]]:
    """rows x cols grid of boxes tiling the full image with no gaps/overlaps."""
    xs = split_boundaries(width, cols)
    ys = split_boundaries(height, rows)
    return [
        [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for c in range(cols)]
        for r in range(rows)
    ]


def layer2_boxes(width: int, height: int, cols: int, rows: int) -> List[List[Box]]:
    """(rows-1) x (cols-1) equally-sized boxes, one centered on each internal
    layer-1 grid vertex, so every layer-1 border is straddled by a patch.
    Boxes are clamped to stay inside the image, which can shift a center by
    a few pixels near non-uniform layer-1 boundaries."""
    xs = split_boundaries(width, cols)
    ys = split_boundaries(height, rows)
    patch_w = round(width / cols)
    patch_h = round(height / rows)

    boxes = []
    for r in range(rows - 1):
        cy = ys[r + 1]
        y0 = max(0, min(cy - patch_h // 2, height - patch_h))
        row_boxes = []
        for c in range(cols - 1):
            cx = xs[c + 1]
            x0 = max(0, min(cx - patch_w // 2, width - patch_w))
            row_boxes.append((x0, y0, x0 + patch_w, y0 + patch_h))
        boxes.append(row_boxes)
    return boxes
