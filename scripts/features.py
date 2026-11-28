"""
Turn each part's mask into a short row of numbers, aka its features.

Saves data/features.npz containing:
  X       (n_crops, n_features)  the feature table: one row per crop, one column per feature
  y       (n_crops,)             the answer for each row as a number: 0=square 1=cylinder 2=hex 3=triangle
  groups  (n_crops,)             session name of each row (used later to split folds by lighting)
  names   feature names          (column labels for X)
  crops   crop file names        (to trace a weird row back to its picture)
and data/review/features.png: one box plot per feature, split by shape.
"""
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["square", "cylinder", "hex", "triangle"]
FEATURE_NAMES = ["area_true", "perimeter_true", "circularity", "n_vertices",
                 "extent", "aspect", "solidity", "circle_fill"]


def compute_features(mask, depth_mm):
    """mask: black/white image of ONE part (255 = part). depth_mm: how far away the part is.
    Returns a list of numbers, in the same order as FEATURE_NAMES."""

    # The part's outline = a list of (x, y) points around its edge
    # CHAIN_APPROX_NONE keeps every edge point (more acc perimeter than _SIMPLE that I used in
    # contours_live.py, but takes more memory)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # biggest outline in case of any stray specks
    c = max(contours, key=cv2.contourArea)

    # pixels inside the outline
    area = cv2.contourArea(c)
    # length of the outline in pixels (true is for closed loop)
    perim = cv2.arcLength(c, True)
    # dist in meters
    z = depth_mm / 1000

    # 1. SIZE, correct for dist -> (for ex: twice as far -> half as wide and half as tall -> 1/4 the pixel
    #    area, multiplying by z^2 undoes that)
    area_true = area * z**2

    # 2. PERIMETER, correct for dist -> a length (not an area) only shrinks by half when the part is twice
    #    as far, so multiply by depth
    perimeter_true = perim * z

    # 3. ROUNDNESS -> 1.0 = perfect circle, a square is ~0.79, a triangle ~0.60
    circularity = 4 * np.pi * area / perim**2

    # 4. CORNERS. approxPolyDP replaces the wiggly outline w as few straight lines as possible,
    #    allowing the simplified shape to stray up to 2% (0.02) of the perimeter from the real edge
    approx = cv2.approxPolyDP(c, 0.02 * perim, True)
    # approx is a ls of the simplified shape's corner points
    n_vertices = len(approx)

    # 5. EXTENT: how much of the tightest (potentially rotated) rectangle around the part it fills
    # center, (width, height), rotation
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
    # part area / rectangle area (for ex: a square fills ~all of it, a triangle about half)
    extent = area / (rw*rh)

    # 6. ASPECT -> short side of that rectangle / long side (ex: 1.0 = as wide as it is long)
    aspect = min(rw, rh) / max (rw, rh)

    # 7. SOLIDITY -> how much of its rubber band-ish outline the part fills, the convex hull is the shape
    # a rubber band would make if stretched around the part
    hull = cv2.convexHull(c)
    # part area / hull area
    solidity = area / cv2.contourArea(hull)

    # 8. CIRCLE FILL: how much of the smallest circle around the part it fills (kinda like extent)
    # center and radius
    (ex, ey), r = cv2.minEnclosingCircle(c)
    # part area / circle area.
    circle_fill = area / (np.pi * r**2)

    return [area_true, perimeter_true, circularity, n_vertices,
            extent, aspect, solidity, circle_fill]


def file_for(row, col):
    """Find the file named in one labels.csv column."""
    p = Path(row[col])
    for cand in (p, DATA / p, DATA / "crops" / row["shape"] / p.name):
        if cand.exists():
            return cand
    raise FileNotFoundError(row[col])


def main():
    with open(DATA / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    X, y, groups, crops = [], [], [], []
    for r in rows:
        mask = cv2.imread(str(file_for(r, "mask")), cv2.IMREAD_GRAYSCALE)
        # make sure it's pure black/white
        mask = (mask > 127).astype(np.uint8) * 255
        X.append(compute_features(mask, float(r["median_depth_mm"])))
        y.append(CLASSES.index(r["shape"]))
        groups.append(r["session"])
        crops.append(r["crop"])

    X = np.array(X, dtype=float)
    y = np.array(y)
    np.savez(DATA / "features.npz", X=X, y=y, groups=np.array(groups),
             names=np.array(FEATURE_NAMES), crops=np.array(crops))
    print(f"feature table X: {X.shape[0]} crops x {X.shape[1]} features  ->  data/features.npz\n")

    # avg of each feature per shape
    print(f"{'feature':<16}" + "".join(f"{c:>10}" for c in CLASSES))
    for j, name in enumerate(FEATURE_NAMES):
        print(f"{name:<16}" + "".join(f"{X[y == k, j].mean():>10.3f}" for k in range(len(CLASSES))))

    # one box plot per feature, if the 4 boxes don't overlap, that feature separates the shapes well
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for j, ax in enumerate(axes.flat):
        ax.boxplot([X[y == k, j] for k in range(len(CLASSES))])
        ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
        ax.set_title(FEATURE_NAMES[j])
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    (DATA / "review").mkdir(exist_ok=True)
    fig.savefig(DATA / "review" / "features.png", dpi=120)
    print("\nsaved box plots to data/review/features.png")


if __name__ == "__main__":
    main()
