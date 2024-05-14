'''
This script creates a CSV file that contains the paths to the RGB and depth images. The dataset is then created using the TraversabilityDataset class.
'''

# Import the necessary libraries
import os
import pandas as pd
import numpy as np
from PIL import Image

print('Libraries imported successfully')

# Define the paths to the RGB and depth images
data_path = r'C:\Users\deeks\Documents\Autonomous-Navigation-for-Unstructured-Environments\test_data\data'
rgb_image_path = os.path.join(data_path, 'rgb_image')
depth_image_path = os.path.join(data_path, 'depth_image')

# Get the list of RGB and depth images
rgb_images = os.listdir(rgb_image_path)
depth_images = os.listdir(depth_image_path)

# Retain images ending with .tif from the list
rgb_images = [image for image in rgb_images if image.endswith('.tif')]
depth_images = [image for image in depth_images if image.endswith('.tif')]

# Retain images with the same name in both lists
rgb_images = [image for image in rgb_images if image in depth_images]
depth_images = [image for image in depth_images if image in rgb_images]

# Sort the lists based on the numbers in the sames before the .tif


rgb_images.sort()
depth_images.sort()

# Print the number of RGB and depth images
print(f'Number of RGB images: {len(rgb_images)}')
print(f'Number of depth images: {len(depth_images)}')

# Print the number of RGB and depth images with the same name
print(f'Number of RGB and depth images with the same name: {len(set(rgb_images) & set(depth_images))}')  

# Create a DataFrame to store the paths to the RGB and depth images
data = {'rgb_image': [], 'depth_image': []}

# Iterate over the RGB images and depth images
for rgb_image, depth_image in zip(rgb_images, depth_images):
    data['rgb_image'].append(os.path.join("rgb_image", rgb_image))
    data['depth_image'].append(os.path.join("depth_img", depth_image))

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(data_path, 'data.csv'), index=False)

# Print the first few rows of the DataFrame
print('Verifying the data.csv file: ')
print(df.head(5))







    




