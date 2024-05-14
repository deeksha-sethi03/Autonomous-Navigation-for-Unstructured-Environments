'''
This script is used to merge the dataset collected in two different folders into 1 folder with the names of the images changed to avoid conflicts.
'''

# Import the necessary libraries
import os
import shutil

# Define the paths to the RGB and depth images
data_path = r'/home/deeksha/Desktop/Autonomous-Navigation-for-Unstructured-Environments/test_data/data'
data_path_1 = r'/home/deeksha/Desktop/Autonomous-Navigation-for-Unstructured-Environments/test_data/set1'
data_path_2 = r'/home/deeksha/Desktop/Autonomous-Navigation-for-Unstructured-Environments/test_data/set2'

# Get the list of RGB and depth images
rgb_images_1 = os.listdir(os.path.join(data_path_1, 'rgb_image'))
depth_images_1 = os.listdir(os.path.join(data_path_1, 'depth_image'))
rgb_images_2 = os.listdir(os.path.join(data_path_2, 'rgb_image'))
depth_images_2 = os.listdir(os.path.join(data_path_2, 'depth_image'))

# Retain images ending with .tif from the list
rgb_images_1 = [image for image in rgb_images_1 if image.endswith('.tif')]
depth_images_1 = [image for image in depth_images_1 if image.endswith('.tif')]
rgb_images_2 = [image for image in rgb_images_2 if image.endswith('.tif')]
depth_images_2 = [image for image in depth_images_2 if image.endswith('.tif')]

# Retain images with the same name in both lists
rgb_images_1 = [image for image in rgb_images_1 if image in depth_images_1]
depth_images_1 = [image for image in depth_images_1 if image in rgb_images_1]
rgb_images_2 = [image for image in rgb_images_2 if image in depth_images_2]
depth_images_2 = [image for image in depth_images_2 if image in rgb_images_2]

# Get length of the first set
length = len(rgb_images_2)
iter = length

# Rename the images in the second set
for i, (rgb_image, depth_image) in enumerate(zip(rgb_images_1, depth_images_1)):
    new_rgb_image = f'{iter+1}.tif'
    new_depth_image = f'{iter+1}.tif'
    iter += 1
    os.rename(os.path.join(data_path_1, 'rgb_image', rgb_image), os.path.join(data_path_1, 'rgb_image', new_rgb_image))
    os.rename(os.path.join(data_path_1, 'depth_image', depth_image), os.path.join(data_path_1, 'depth_image', new_depth_image))

# Get updated list of RGB and depth images names in the first set
rgb_images_1 = os.listdir(os.path.join(data_path_1, 'rgb_image'))
depth_images_1 = os.listdir(os.path.join(data_path_1, 'depth_image'))

# Retain images ending with .tif from the list
rgb_images_1 = [image for image in rgb_images_1 if image.endswith('.tif')]
depth_images_1 = [image for image in depth_images_1 if image.endswith('.tif')]

# Retain images with the same name in both lists
rgb_images_1 = [image for image in rgb_images_1 if image in depth_images_1]
depth_images_1 = [image for image in depth_images_1 if image in rgb_images_1]

# Print the number of RGB and depth images
print(f'Number of RGB images in set 2: {length}')
print(f'Number of depth images in set 2: {length}')
print(f'Number of RGB images in set 1: {iter-length}')
print(f'Number of depth images in set 1: {iter-length}')

# Print the number of RGB and depth images with the same name
print(f'Number of RGB and depth images with the same name in set 1: {len(set(rgb_images_1) & set(depth_images_1))}')
print(f'Number of RGB and depth images with the same name in set 2: {len(set(rgb_images_2) & set(depth_images_2))}')
print(f'Number of RGB and depth images with the same name in set 1 and set 2: {len(set(rgb_images_1) & set(rgb_images_2))}')

# # Print the number of RGB and depth images in the merged dataset
# print(f'Number of RGB images in the merged dataset: {len(os.listdir(os.path.join(data_path, "rgb_image")))}')
# print(f'Number of depth images in the merged dataset: {len(os.listdir(os.path.join(data_path, "depth_image")))}')

# Save all the new dataset images to a new folder
new_data_path = r'/home/deeksha/Desktop/Autonomous-Navigation-for-Unstructured-Environments/test_data/data'

# Create a new folder for RGB in the new folder
rgb_image_path = os.path.join(new_data_path, 'rgb_image')
if not os.path.exists(rgb_image_path):
    os.makedirs(rgb_image_path)

# Create a new folder for depth in the new folder
depth_image_path = os.path.join(new_data_path, 'depth_image')
if not os.path.exists(depth_image_path):
    os.makedirs(depth_image_path)

# Copy the images from the first set to the new folder
for rgb_image, depth_image in zip(rgb_images_1, depth_images_1):
    shutil.copy(os.path.join(data_path_1, 'rgb_image', rgb_image), os.path.join(new_data_path, 'rgb_image', rgb_image))
    shutil.copy(os.path.join(data_path_1, 'depth_image', depth_image), os.path.join(new_data_path, 'depth_image', depth_image))

# Copy the images from the second set to the new folder
for rgb_image, depth_image in zip(rgb_images_2, depth_images_2):
    shutil.copy(os.path.join(data_path_2, 'rgb_image', rgb_image), os.path.join(new_data_path, 'rgb_image', rgb_image))
    shutil.copy(os.path.join(data_path_2, 'depth_image', depth_image), os.path.join(new_data_path, 'depth_image', depth_image))


# Print the number of RGB and depth images in the new dataset
print(f'Number of RGB images in the new dataset: {len(os.listdir(os.path.join(new_data_path, "rgb_image")))}')
print(f'Number of depth images in the new dataset: {len(os.listdir(os.path.join(new_data_path, "depth_image")))}')
