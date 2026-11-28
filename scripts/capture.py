"""
Capture labeled crops of the parts: this builds the dataset.

How to use (one shape per run):
  1. Run it, type which shape is on the mat and the session number or lighting description.
  2. Clear the mat and press Enter: it photographs the empty mat (brightness and depth).
  3. Put the 3 copies of that shape on the mat, spread out (not touching), but still within
     yellow region. A pixel counts as a part if it is at least BRIGHT_DIFF brigher than the
     empty reference mat.(Tried just depth at first but was too noisy at a distance)
  4. Press SPACE to save. Move and rotate the parts, press SPACE again. ~9 times.
  5. Press q to quit. Run again for the next shape.

Saves under data/:
  frames/<id>_color.png, <id>_depth.npy               full frame
  crops/<shape>/<id>_p<k>.png, _mask.png, _depth.npy  one set per part (PAD px border)
  labels.csv                                          one row per part (appended, never overwritten)
<id> = s<session>_<shape>_<HHMMSS run stamp>_<shot number>
"""
import csv
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs

# Settings
# a pixel is part if it's at least this much closer than the empty mat
THRESH_MM = 8
# dont consider blobs smaller then this many pixels as parts
MIN_AREA = 300
# extra pixels kept around each part when cropping
PAD = 10
# a pixel is a part of it is much brighter than the empty mat
BRIGHT_DIFF = 40
CLASSES = ["square", "cylinder", "hex", "triangle"]

# The mat region to look at, as 4 corners (x, y) going clockwise
ROI_CORNERS = [(85,122), (607, 122), (639, 479), (0, 479)]

# send data to learning_to_sort/data
DATA = Path(__file__).resolve().parent.parent / "data"

# What's on the mat this run?
shape = input(f"Which shape is on the mat? {CLASSES}: ").strip()
while shape not in CLASSES:
    shape = input(f"Please type one of {CLASSES}: ").strip()
session = input("Session number (new lighting or setup = new number): ").strip()
run_stamp = time.strftime("%H%M%S")

# Folders and the labels file
crop_dir = DATA / "crops" / shape
frame_dir = DATA / "frames"
# make the folders if they don't exist yet
crop_dir.mkdir(parents=True, exist_ok=True)
frame_dir.mkdir(parents=True, exist_ok=True)

labels_path = DATA / "labels.csv"
is_new_file = not labels_path.exists()
# "a" = append, helps to never erase earlier runs
labels_file = open(labels_path, "a", newline="")
writer = csv.writer(labels_file)
if is_new_file:
    writer.writerow(["crop", "mask", "depth", "shape", "session", "frame",
                     "x", "y", "w", "h", "area_px", "median_depth_mm"])

# A mask for our region of interest
roi = np.zeros((480, 640), np.uint8)
roi_polygon = np.array(ROI_CORNERS, np.int32)
cv2.fillPoly(roi, [roi_polygon], 255)

# Camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)


def get_frames():
    """Return one aligned (color_image, depth_image) pair as numpy arrays (copied)."""
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame and color_frame:
            return np.array(color_frame.get_data()), np.array(depth_frame.get_data())


try:
    # Photograph the empty mat
    input("Clear the mat, then press Enter...")
    for _ in range(30):
        get_frames()
    empty_shots = []
    empty_gray_shots = []
    for _ in range(30):
        color, depth = get_frames()
        empty_shots.append(depth)
        # brightness only, 0-255
        empty_gray_shots.append(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    empty = np.median(np.stack(empty_shots), axis=0).astype(np.int32)
    empty_gray = np.median(np.stack(empty_gray_shots), axis=0).astype(np.int32)
    print(f"Empty mat captured. Put the {shape}s on the mat. SPACE = save, q = quit.")

    # how many photos and crops saved this run
    shot = 0
    saved = 0
    while True:
        color, depth = get_frames()

        # Find the parts
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.int32)
        # how much brighter than the empty mat
        brighter = gray - empty_gray
        part_mask = brighter >= BRIGHT_DIFF
        mask = part_mask.astype(np.uint8) * 255
        # ignore outside the mat region
        mask = cv2.bitwise_and(mask, roi)
        # remove specks then fill small holes with CLOSE
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_AREA:
                parts.append(cnt)

        # Draw on a copy, so that the saved images stay clean
        view = color.copy()
        # yellow for the region we look in
        cv2.polylines(view, [roi_polygon], True, (0, 255, 255), 1)
        for cnt in parts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(view, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 0), 1)
        status = f"{shape} | session {session} | parts in view: {len(parts)} | saved: {saved}"
        cv2.putText(view, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("capture", view)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # anything but SPACE: keep watching
        if key != ord(" "):
            continue

        # when 'space' is pressed -> save the full images, then one crop per part
        frame_id = f"s{session}_{shape}_{run_stamp}_{shot:03d}"
        cv2.imwrite(str(frame_dir / f"{frame_id}_color.png"), color)
        np.save(frame_dir / f"{frame_id}_depth.npy", depth)

        # i = 0, 1, 2, ... one per part
        for i, cnt in enumerate(parts):
            x, y, w, h = cv2.boundingRect(cnt)

            # Pad extra pixels on every side, but never past the image edges (0 .. 640 across, 0 .. 480 down).
            # Hint: max(..., 0) and min(..., 640) keep a number inside the image.
            x0 = max(x - PAD, 0)
            y0 = max(y - PAD, 0)
            x1 = min(x + w + PAD, 640)
            y1 = min(y + h + PAD, 480)

            # The part's own mask: draw this contour filled in with a thickness -1
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)

            # Median depth of this part's pixels that have a reading, the depth[condition] picks
            # out only the pixels where the condition is True
            part_depths = depth[(blob > 0) & (depth > 0)]
            if part_depths.size > 0:
                median_depth = int(np.median(part_depths))
            else:
                median_depth = 0       # no depth reading on this part

            # Cut the crops out -> image[rows, columns] = image[y0:y1, x0:x1]
            name = f"{frame_id}_p{i}"
            cv2.imwrite(str(crop_dir / f"{name}.png"), color[y0:y1, x0:x1])
            cv2.imwrite(str(crop_dir / f"{name}_mask.png"), blob[y0:y1, x0:x1])
            np.save(crop_dir / f"{name}_depth.npy", depth[y0:y1, x0:x1])

            writer.writerow([f"crops/{shape}/{name}.png", f"crops/{shape}/{name}_mask.png",
                             f"crops/{shape}/{name}_depth.npy", shape, session, frame_id,
                             x0, y0, x1 - x0, y1 - y0, int(cv2.contourArea(cnt)), median_depth])

        # Write the rows to disk now, in case of a crash
        labels_file.flush()
        shot += 1
        saved += len(parts)
        print(f"Saved {len(parts)} crops from {frame_id} (this run: {saved})")
finally:
    pipeline.stop()
    labels_file.close()
    cv2.destroyAllWindows()
