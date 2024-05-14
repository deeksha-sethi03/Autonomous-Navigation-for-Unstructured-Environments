'''
This script is used to perform inference on the trained model. It takes the images from the car as the input and outputs the mu map. 
The mu map is then used to calculate the waypoints.
'''

# Import the necessary libraries
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.resnet_depth_unet import ResnetDepthUnet
from utils.dataloader import TraversabilityDataset
import os
import sys
import time

class Object(object):
    pass

params = Object()
# dataset parameters
params.data_path        = r'C:/Users/deeks/Documents/WayFAST/test_data/data'
params.csv_path         = os.path.join(params.data_path, 'data.csv')
params.preproc          = True  # Vertical flip augmentation
params.depth_mean       = 3.5235
params.depth_std        = 10.6645

# training parameters
params.seed             = 230
params.epochs           = 50
params.batch_size       = 16
params.learning_rate    = 1e-4
params.weight_decay     = 1e-5

# model parameters
params.pretrained = True
params.load_network_path = None 
params.input_size       = (424, 240)
params.output_size      = (424, 240)
params.output_channels  = 1
params.bottleneck_dim   = 256

class Inference:
    def __init__(self, model_path, rgb_image_path, depth_image_path, output_path):
        self.model_path = model_path
        self.rgb_image_path = rgb_image_path
        self.depth_image_path = depth_image_path
        self.output_path = output_path

    def dataset_creation(self):

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Create the dataset
        dataset = TraversabilityDataset(params, transform)

    def run(self):


        dataset = TraversabilityDataset(params, transform)

        # Load the model
        model = torch.load(self.model_path)
        model.eval()

        # Load the images
        rgb_image = Image.open(self.rgb_image_path)
        depth_image = Image.open(self.depth_image_path)

        # Preprocess the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        rgb_image = transform(rgb_image).unsqueeze(0)
        depth_image = transform(depth_image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(rgb_image, depth_image)

        # Save the output
        output = output.squeeze().cpu().numpy()
        np.save(self.output_path, output)


if __name__ == '__main__':
    model_path = r"C:\Users\deeks\Documents\Autonomous-Navigation-for-Unstructured-Environments\models\resnet_depth_unet.py"
    rgb_image_path = sys.argv[2]
    depth_image_path = sys.argv[3]
    output_path = sys.argv[4]

    inference = Inference(model_path, rgb_image_path, depth_image_path, output_path)
    inference.run()

    
