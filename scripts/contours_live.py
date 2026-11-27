"""
Find parts on the mat and outline them live.

1. Run it with the mat empty, this will take a ref depth image of the empty mat.
2. Put parts on the mat and watch the two windows (for mask and countours).
3. Can quit once all looks good.
"""

import cv2
import numpy as np
import pyrealsense2 as rs


# camera setup (same as align-depth2color.py)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)


def get_frames():
    """Return one aligned (color_image, depth_image) pair as numpy arrays."""
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if both arrived ... if not loop to try again
        if depth_frame and color_frame:
            # .get_data hands over the raw pixel numbers stored inside a frame (w/out unnnecessary
            # info like timestamp)
            return (np.array(color_frame.get_data()),      # shape is 480, 640, 3
                    np.array(depth_frame.get_data()))      # ^ 480, 640


try:
    # take reference image of the empty mat
    input("Clear the mat, then press Enter...")
    # throw away 30 frames so the camera's exposure settles
    for _ in range(30):
        get_frames()

    # Take 30 depth images of the empty mat and keep them in a ls
    empty_shots = []
    for _ in range(30):
        # get_frames hands us both color and depth frames, we only want depth frames
        color, depth = get_frames()
        empty_shots.append(depth)

    # Stack the ls into one array of shape (30, 480, 640) so that we have 30 images on top of each other
    empty_stack = np.stack(empty_shots)
    # For each pixel, take the median of its 30 readings. axis=0 means along the first dim, aka the 30 images
    # The result returns one image of shape (480, 640). Median ignores bad readings better than an ave would
    empty = np.median(empty_stack, axis=0)
    # Store it as int32 so later subtractions can go negative (uint16 cant)
    empty = empty.astype(np.int32)
    print("Empty (reference) mat captured. Put parts on the mat. Hit q in a window to quit.")

    while True:
        color, depth = get_frames()
        # how much closer than the empty mat, in mm
        diff = empty - depth.astype(np.int32)

        # Mask pixels where the new depth image has a depth > 0 AND that's depth  is >= 10 than that of the ref
        # (empty) image, if both of these conditions are True, there is a part there.
        part_mask = (diff >= 10) & (depth > 0)

        # OpenCV needs a black (0) and white (255) image not True/False
        mask = part_mask.astype(np.uint8) * 255

        # Opening shrink every white blob, then grows it back.
        # Specks smaller than the 5x5 brush vanish, real parts return to their size.
        brush = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)

        # Get the outer contours of each part. RETR_EXTERNAL is the contour retrievel mode, it return's only the
        # outline of the blob. CHAIN... removes redundant points / select points that are endpoints of a straight
        # line and only keeps those for memory's sake (can still create outline w/out)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore blobs too small to be a part, to do this pick a cutoff in pixels
            if area < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(color, [cnt], -1, (0, 255, 0), 2)             # green outline
            cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 1)   # blue bounding box
            cv2.putText(color, f"{area:.0f} px", (x, y - 5),               # area w no decimals
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("mask", mask)
        cv2.imshow("contours", color)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
