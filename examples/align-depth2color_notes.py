## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 RealSense, Inc. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# My (elias) notes are depicted between the --- lines or XXX (for single lines)

# -----------------------------------------------------------------------------
# XXX: this example shows
#   - the camera's color image with everything further than 1 m greyed out

#       - I will do smth similar for my proj, except will take photos of an
#         empty mat, and compare that empty mat's depth to frames with the
#         objects on the mat
#           - have to do it this way bc the camera will not be directly over-
#             ahead, so there is no one const cut-off depth

#   - how to start the camera, grab frames in a loop, and rm the background
# -----------------------------------------------------------------------------
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
# XXX carries frames (single snapshots) from the camera to my code
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream different resolutions
#    of color and depth streams
# XXX which streams I want and at what size and speed
config = rs.config()

# XXX this block only verifies that camera has a color sensor -> quits if not
# ----------------------------------------------------------------------------
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
# ----------------------------------------------------------------------------

# XXX
# Filling in configs. Which stream, width, height, format, frames/second
#   - z16: indicates that each depth pixel is a whole number, is a raw dist
#     reading, not yet converted to meters
#   - bgr8: each color pixel is three #s from 0 to 255 in blue, green, red
#     order, this OpenCV's convention
# NOTE peep how when running dis script you get a window with these 2 streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
# XXX 'start' streaming w our conifg
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
# XXX 'depth_scale' converts raw depth readings to meters
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#     clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
# XXX converts the meters into raw units, will be comparing raw pixels later in the code.
#        Also better to convert this one meter into raw units rather than a whole bunch of pixels to meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
# XXX ^ Need to align bc the depth sensor and color camera sit side by side on the RealSense, so they see the
#   scene slightly differently. 'rs.align' remaps the depth image so every pixel lines up with the color image.

# Streaming loop
try:
    while True:
        # XXX Pauses until the next frameset (one color + one depth frame) from the same instant arrive
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image         (ignore)

        # Align the depth frame to color frame
        # XXX returns the aligned version of the frameset
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        # XXX if the frmaeset is missing -> just continue/skip through loop and wait for the next frameset  ^^
        if not aligned_depth_frame or not color_frame:
            continue
        # XXX Convert the camera's frame objects into numpy arrays (the plain grids I'll do math on )
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        # XXX the color image has 3 numbers per pixel, but depth only has 1. 'dstack' stacks three copies of the depth image
        #   to match the shape of the color image, so that the next line can compare them number by number
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # XXX 'np.where(condition, A, B) means that for each value if the condition is true use A (grey), otherwise use B (color)
        #   So if too far "|" aka 'or' no reading the background becomes grey 153
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # XXX convertScaleAbs(..., alpha=0.03) multiplies each depth value by 0.03 and caps the result at 255, so depth fits the
        #   0-255 range the screen can show
        # XXX applyColorMap(..., JET) paints fake colors on that ^ where blue is near and red is far away
        # XXX 'hstack' glues the two images side by side into one horizontally
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

# XXX always shuts the camera off cleanly even if there is an error
finally:
    pipeline.stop()
