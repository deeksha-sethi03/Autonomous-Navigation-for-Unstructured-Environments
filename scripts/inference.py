'''
This script is used to perform inference on the trained model. It takes the images from the car as the input and outputs the mu map. 
The mu map is then used to calculate the waypoints.
'''

# Import the necessary libraries
import torch
import numpy as np
from PIL import Image
import os
import sys
import time
import cv2
sys.path.append(r'C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments')
from models.resnet_depth_unet import ResnetDepthUnet
from torchvision import transforms
from utils.dataloader import TraversabilityDataset
import matplotlib.pyplot as plt

class Object(object):
    pass

params = Object()
# dataset parameters
params.data_path        = r'C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\test_data\data'
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
params.input_size       = (424, 240)#(480, 640)
params.output_size      = (424, 240)
params.output_channels  = 1
params.bottleneck_dim   = 256

class Inference:
    def __init__(self, model_path, rgb_image_path, depth_image_path, output_path):
        self.model_path = model_path
        self.rgb_image_path = rgb_image_path
        self.depth_image_path = depth_image_path
        self.output_path = output_path

    # def dataset_creation(self):

    #     transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     # Create the dataset
    #     dataset = TraversabilityDataset(params, transform)

    def run(self):

        # dataset = TraversabilityDataset(params, transform)

        # Load the model
        model = ResnetDepthUnet(params).double()
        # model = torch.load(self.model_path)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        # Load the images
        rgb_image = np.array(Image.open(self.rgb_image_path))
        depth_image = np.array(Image.open(self.depth_image_path))

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        # because pytorch pretrained model uses RGB
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, params.input_size, interpolation = cv2.INTER_AREA)
        rgb_image = transform(rgb_image)

        # Convert depth to meters
        depth_image = cv2.resize(depth_image, params.input_size, interpolation = cv2.INTER_AREA)
        depth_image = np.uint16(depth_image)
        depth_image = depth_image*10**-3
        # Normalize depth image
        depth_image = (depth_image-params.depth_mean)/params.depth_std
        depth_image = np.expand_dims(depth_image, axis=2)
    
        depth_image = np.transpose(depth_image, (2, 0, 1))
        rgb_image= torch.from_numpy(np.expand_dims(rgb_image, axis=0)).double()
        depth_image= torch.from_numpy(np.expand_dims(depth_image, axis=0)).double()
                
        # rgb_image = transform(rgb_image).unsqueeze(0)
        # depth_image = transform(depth_image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(rgb_image, depth_image)

        # Save the output
        output = output.squeeze().cpu().numpy()
        np.save(output_path, output)
        print("saved")


if __name__ == '__main__':
    model_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\model-checkpoints\best_predictor_depth.pth"
    rgb_image_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\test_data\data\real_rgb_image\15.tif"
    depth_image_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\test_data\data\real_depth_image\15.tif"
    output_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\Output\15"

    inference = Inference(model_path, rgb_image_path, depth_image_path, output_path)
    inference.run()

    load_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\Output\we_did_it.npy"
    op= np.load(load_path)

    # Display the image
    plt.imshow(op, cmap='gray')  # Assuming it's a grayscale image, change the cmap as needed
    plt.axis('off')  # Turn off axis
    plt.show()
    
# import torch
# import numpy as np
# from PIL import Image
# import os
# import sys
# import time
# import cv2
# sys.path.append(r'C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments')
# from models.resnet_depth_unet import ResnetDepthUnet
# from torchvision import transforms
# from utils.dataloader import TraversabilityDataset
# import matplotlib.pyplot as plt

# class Object(object):
#     pass

# params = Object()
# # dataset parameters
# params.data_path        = r'C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\test_data\data'
# params.csv_path         = os.path.join(params.data_path, 'data.csv')
# params.preproc          = True  # Vertical flip augmentation
# params.depth_mean       = 3.5235
# params.depth_std        = 10.6645

# # model parameters
# params.pretrained = True
# params.input_size       = (424, 240)
# params.output_size      = (424, 240)
# params.output_channels  = 1
# params.bottleneck_dim   = 256

# class Inference:
#     def __init__(self, model_path, input_folder, output_folder):
#         self.model_path = model_path
#         self.input_folder = input_folder
#         self.output_folder = output_folder

#     def run(self):
#         # Load the model
#         model = ResnetDepthUnet(params).double()
#         model.load_state_dict(torch.load(self.model_path))
#         model.eval()

#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         for filename in os.listdir(self.input_folder):
#             if filename.endswith(".tif"):
#                 # Load the images
#                 rgb_image_path = os.path.join(self.input_folder, filename)
#                 depth_image_path = os.path.join(self.input_folder, filename.replace("real_rgb_image", "real_depth_image"))
#                 output_path = os.path.join(self.output_folder, filename.replace(".tif", ".npy"))

#                 rgb_image = np.array(Image.open(rgb_image_path))
#                 depth_image = np.array(Image.open(depth_image_path))

#                 # Preprocess images
#                 rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
#                 rgb_image = cv2.resize(rgb_image, params.input_size, interpolation=cv2.INTER_AREA)
#                 rgb_image = transform(rgb_image)

#                 depth_image = cv2.resize(depth_image, params.input_size, interpolation=cv2.INTER_AREA)
#                 depth_image = np.uint16(depth_image)
#                 depth_image = depth_image * 10 ** -3
#                 depth_image = (depth_image - params.depth_mean) / params.depth_std
#                 depth_image = np.expand_dims(depth_image, axis=2)
#                 depth_image = np.transpose(depth_image, (2, 0, 1))

#                 rgb_image = torch.from_numpy(np.expand_dims(rgb_image, axis=0)).double()
#                 depth_image = torch.from_numpy(np.expand_dims(depth_image, axis=0)).double()

#                 # Perform inference
#                 with torch.no_grad():
#                     output = model(rgb_image, depth_image)

#                 # Save the output
#                 output = output.squeeze().cpu().numpy()
#                 np.save(output_path, output)
#                 print(f"Inference saved for {filename}")

# if __name__ == '__main__':
#     model_path = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\model-checkpoints\best_predictor_depth.pth"
#     input_folder = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\test_data\data"
#     output_folder = r"C:\Users\tanay\Documents\Learning\Project\Autonomous-Navigation-for-Unstructured-Environments\Output"

#     inference = Inference(model_path, input_folder, output_folder)
#     inference.run()
