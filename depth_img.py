
'''
You need to install pyrealsense2 library for this
'''
# pip install pyrealsense2

from PIL import Image
import numpy as np

# Load your depth image
depth_image = Image.open('C:\\Users\\shrey\\Documents\\LIR_Project\\Autonomous-Navigation-for-Unstructured-Environments\\test_data\\data\\depth_image\\1.tif')

# Convert the image to grayscale
depth_image_gray = depth_image.convert('L')

# Convert the grayscale image to a numpy array
depth_array_gray = np.array(depth_image_gray)

# Check the shape of the array
print("Depth image shape:", depth_array_gray.shape)

# Convert the image to a numpy array
# depth_array = np.array(depth_image)

# Convert depth values from millimeters to meters
depth_array_meters = depth_array_gray * 0.001


# Normalize depth values between 0 and 1
normalized_depth = (depth_array_meters - np.min(depth_array_meters)) / (np.max(depth_array_meters) - np.min(depth_array_meters))

# Scale normalized depth to 0-255 range for grayscale
depth_gray = (normalized_depth * 255).astype(np.uint8)

# Create a PIL image from the grayscale array
depth_image_gray = Image.fromarray(depth_gray)

# Display grayscale depth image
depth_image_gray.show()
