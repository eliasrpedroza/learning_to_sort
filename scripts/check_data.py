"""
Double check the dataset looks trainable b4 any ML

What it does:
  1. Prints how many crops you have for each session x shape.
  2. Flags crop files that are missing, and crops with no depth reading.
  3. Prints the pixel-area range per shape (a crop ~2x the usual area is
     probably two parts merged into one blob).
  4. Saves picture sheets so you can eyeball bad crops (hands, merged parts):
       data/review/overview.png  -> rows = shapes, columns = sessions, 4 random crops each
       data/review/<shape>.png   -> every crop of that shape, numbered
       data/review/<shape>.txt   -> number -> crop file name (to find a bad one)
"""
import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
REVIEW = DATA / "review"
CLASSES = ["square", "cylinder", "hex", "triangle"]
# each crop is shrunk to fit in a tile x tile square in the sheets
TILE = 80
# crops per row in the per shape sheets
COLS = 12


def crop_path(row):
    """Find the crop image for one labels.csv row."""
    p = Path(row["crop"])
    if p.suffix == "":
        p = p.with_suffix(".png")
    for cand in (p, DATA / p, DATA / "crops" / row["shape"] / p.name):
        if cand.exists():
            return cand
    return None


def tile(img, label=None):
    """Shrink a crop to fit a tile x tile black square, can with a number in the corner."""
    h, w = img.shape[:2]
    s = TILE / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    out = np.zeros((TILE, TILE, 3), np.uint8)
    y0 = (TILE - small.shape[0]) // 2
    x0 = (TILE - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    if label is not None:
        cv2.putText(out, str(label), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def sheet(tiles, cols):
    """Arrange tiles into a grid image."""
    tiles = list(tiles)
    blank = np.zeros((TILE, TILE, 3), np.uint8)
    while len(tiles) % cols:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    with open(DATA / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("labels.csv has no rows yet.")
        return

    shapes = [c for c in CLASSES if any(r["shape"] == c for r in rows)]
    # any unexpected labels
    shapes += sorted({r["shape"] for r in rows} - set(shapes))
    # in the order you collected them
    sessions = list(dict.fromkeys(r["session"] for r in rows))

    # 1. counts table
    counts = Counter((r["session"], r["shape"]) for r in rows)
    name_w = max(len("session"), *(len(s) for s in sessions))
    print(f"\n{'session':<{name_w}}  " + "  ".join(f"{s:>8}" for s in shapes) + "     total")
    for sess in sessions:
        line = [counts[(sess, s)] for s in shapes]
        print(f"{sess:<{name_w}}  " + "  ".join(f"{n:>8}" for n in line) + f"  {sum(line):>8}")
    totals = [sum(counts[(sess, s)] for sess in sessions) for s in shapes]
    print(f"{'total':<{name_w}}  " + "  ".join(f"{n:>8}" for n in totals) + f"  {len(rows):>8}\n")

    # 2. missing files and crops with no depth
    missing = [r["crop"] for r in rows if crop_path(r) is None]
    no_depth = [r["crop"] for r in rows if float(r["median_depth_mm"]) == 0]
    print(f"missing crop files: {len(missing)}")
    for m in missing[:10]:
        print("   ", m)
    print(f"crops with no depth reading (median_depth_mm = 0): {len(no_depth)}")
    for m in no_depth[:10]:
        print("   ", m)

    # 3. area range per shape
    print("\npixel area per shape (min / median / max):")
    for s in shapes:
        a = np.array([float(r["area_px"]) for r in rows if r["shape"] == s])
        print(f"   {s:<9} {a.min():7.0f} / {np.median(a):7.0f} / {a.max():7.0f}")

    # 4. picture sheets
    REVIEW.mkdir(exist_ok=True)
    random.seed(0)

    fig, axes = plt.subplots(len(shapes), len(sessions),
                             figsize=(2.4 * len(sessions), 2.4 * len(shapes)), squeeze=False)
    for i, s in enumerate(shapes):
        for j, sess in enumerate(sessions):
            ax = axes[i][j]
            ax.axis("off")
            cell = [r for r in rows if r["shape"] == s and r["session"] == sess and crop_path(r)]
            if not cell:
                continue
            pick = random.sample(cell, min(4, len(cell)))
            imgs = [tile(cv2.imread(str(crop_path(r)))) for r in pick]
            ax.imshow(cv2.cvtColor(sheet(imgs, 2), cv2.COLOR_BGR2RGB))
            if i == 0:
                ax.set_title(sess, fontsize=9)
            if j == 0:
                ax.text(-0.08, 0.5, s, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(REVIEW / "overview.png", dpi=150)
    plt.close(fig)

    for s in shapes:
        mine = [r for r in rows if r["shape"] == s and crop_path(r)]
        if not mine:
            continue
        tiles = [tile(cv2.imread(str(crop_path(r))), k) for k, r in enumerate(mine)]
        cv2.imwrite(str(REVIEW / f"{s}.png"), sheet(tiles, COLS))
        with open(REVIEW / f"{s}.txt", "w") as f:
            for k, r in enumerate(mine):
                f.write(f"{k}\t{r['session']}\t{crop_path(r).name}\n")

    print(f"\nsaved picture sheets to {REVIEW}/")


if __name__ == "__main__":
    main()
