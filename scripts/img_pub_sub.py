# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import cv2
# from cv_bridge import CvBridge
# import pyrealsense2 as rs
# import numpy as np
# import time

# class CameraPublisherSubscriber(Node):
#     def __init__(self):
#         super().__init__('camera_pubsub')

#         # Initialize the RealSense pipeline
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#         # Create ROS publisher for RGB and depth images
#         self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
#         self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

#         # Initialize CvBridge
#         self.bridge = CvBridge()

#         # Subscribe to RGB and depth image topics
#         self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
#         self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)


#     def publish_images(self):
#         try:
#             while rclpy.ok():
#         # Wait for a coherent pair of frames: depth and color
#                 # print("in try")              
#                 frames = self.pipeline.wait_for_frames()

#                 # print("got frames")
#                 depth_frame = frames.get_depth_frame()
#                 color_frame = frames.get_color_frame()

#                 # Convert depth and color frames to numpy arrays
#                 depth_image = np.asanyarray(depth_frame.get_data())
#                 color_image = np.asanyarray(color_frame.get_data())

#                 # Publish RGB image
#                 rgb_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
#                 self.rgb_pub.publish(rgb_msg)
#                 # print("published rgb")
#                 # Publish depth image
#                 depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
#                 self.depth_pub.publish(depth_msg)
#                 # print("published depth")

#         except KeyboardInterrupt:
#             print("in except")
#             pass

#         finally:
#             print("in finally")
#             # Stop streaming
#             self.pipeline.stop()

#     def save_image(self, image, filename_prefix):
#         # Generate a unique filename using timestamp
#         timestamp = int(time.time() * 1000)  # milliseconds
#         filename_tif = f'{filename_prefix}_{timestamp}.tif'
#         filename_png = f'{filename_prefix}_{timestamp}.png'

#         # Save the image
#         cv2.imwrite(filename_tif, image)
#         cv2.imwrite(filename_png, image)
#         self.get_logger().info(f'Saved image: {filename_tif}')

#     def rgb_callback(self, msg):
#         self.get_logger().info("in rgb callback")
#         # Convert ROS image to OpenCV format
#         rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#         # Save RGB image
#         self.save_image(rgb_image, '/home/nvidia/final_proj_ese650/src/img_pub_sub_pkg/imgs/rgb_image/')

#     def depth_callback(self, msg):
#         self.get_logger().info("in depth callback")
#         # Convert ROS image to OpenCV format
#         depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

#         # Save depth image
#         self.save_image(depth_image, '/home/nvidia/final_proj_ese650/src/img_pub_sub_pkg/imgs/depth_image/')

# def main(args=None):
#     rclpy.init(args=args)
#     camera_pubsub = CameraPublisherSubscriber()
#     camera_pubsub.publish_images()
#     rclpy.spin_once(camera_pubsub)
#     camera_pubsub.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import time
import threading

class CameraPublisherSubscriber(Node):
    def __init__(self):
        super().__init__('camera_pubsub')

        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 5)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 5)

        # Start streaming
        self.pipeline.start(self.config)

        # Create ROS publisher for RGB and depth images
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to RGB and depth image topics
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Flag to indicate if the node should continue running
        self.running = True

    def start_publishing_images(self):
        # Start a separate thread for publishing images
        threading.Thread(target=self.publish_images).start()

    def publish_images(self):
        try:
            while self.running:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # Convert depth and color frames to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Publish RGB image
                rgb_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
                self.rgb_pub.publish(rgb_msg)

                # Publish depth image
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
                self.depth_pub.publish(depth_msg)

        except KeyboardInterrupt:
            pass

        finally:
            # Stop streaming
            self.pipeline.stop()

    def save_image(self, image, filename_prefix):
        # Generate a unique filename using timestamp
        timestamp = int(time.time() * 1000)  # milliseconds
        filename_tif = f'{filename_prefix}_{timestamp}.tif'
        filename_png = f'{filename_prefix}_{timestamp}.png'

        # Save the image
        cv2.imwrite(filename_tif, image)
        cv2.imwrite(filename_png, image)
        self.get_logger().info(f'Saved image: {filename_tif}')

    def rgb_callback(self, msg):
        # Convert ROS image to OpenCV format
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Save RGB image
        self.save_image(rgb_image, '/home/nvidia/final_proj_ese650/src/img_pub_sub_pkg/imgs/rgb_image/')

    def depth_callback(self, msg):
        # Convert ROS image to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Save depth image
        self.save_image(depth_image, '/home/nvidia/final_proj_ese650/src/img_pub_sub_pkg/imgs/depth_image/')

def main(args=None):
    rclpy.init(args=args)
    camera_pubsub = CameraPublisherSubscriber()
    camera_pubsub.start_publishing_images()
    rclpy.spin(camera_pubsub)
    camera_pubsub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
