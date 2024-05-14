import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Check if the camera is successfully opened
if not pipeline:
    print("Failed to open camera")
    exit()

# Initialise timer and frame counter
time_start = time.perf_counter()
frame_count = 0
count= 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert depth and color frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Display color image
        cv2.imshow('Color Image', color_image)

        # Display depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        # Update the frame count
        frame_count += 1

        # Average FPS calculated every 60 frames and printed
        if frame_count >= 60:
            count= count + 1
            print(frame_count)
            time_end = time.perf_counter()
            cv2.imwrite('./imgs/depth_images/' + str(count) + '.tif', depth_colormap)
            cv2.imwrite('./imgs/rgb_images/' + str(count) + '.tif', color_image)
            cv2.imwrite('./imgs/depth_images/' + str(count) + '.png', depth_colormap)
            cv2.imwrite('./imgs/rgb_images/' + str(count) + '.png', color_image)
            fps = frame_count / (time_end - time_start)
            print(f'Average FPS: {fps:.2f}')
            frame_count = 0  # reset the frame counter
            time_start = time.perf_counter()  # reset timer

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

