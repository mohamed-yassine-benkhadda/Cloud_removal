import argparse
import os
import cv2
import numpy as np
import torch

import config
from Task3_Super_resolution.SRGANPyTorch.imgproc import image_to_tensor, tensor_to_image
from Task3_Super_resolution.SRGANPyTorch.model import Generator


def main(weights_path = "Task3_Super_resolution/SRGANPyTorch/results/pretrained_models/SRResNet_x4-ImageNet-2096ee7f.pth.tar", inputs_path = "Task2_image_inpainting/lama/output/pic_mask.png", output_path = "output/picHR.png"):
    # Initialize the model
    print(os.listdir("."))
    model = Generator()
    model = model.to("cpu")
    print("Build SRGAN model successfully.")

    # Load the SRGAN model weights
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRGAN model weights `{weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    image = cv2.imread(inputs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    imagesr = np.zeros((384 * 4, 384 * 4, 3))
    for i in range(16):
        for j in range(16):
            lr_image = image[24 * i: 24 * (i+1), 24 * j: 24 * (j+1),:]
            # Convert BGR channel image format data to RGB channel image format data
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

            # Convert RGB channel image format data to Tensor channel image format data
            lr_tensor = image_to_tensor(lr_image, False, False).unsqueeze_(0)

            # Transfer Tensor channel image format data to CUDA device
            lr_tensor = lr_tensor.to("cpu")

            # Use the model to generate super-resolved images
            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            # Save image
            sr_image = tensor_to_image(sr_tensor, False, False)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            imagesr[96 * i: 96 * (i+1), 96 * j: 96 * (j+1),:] = sr_image
    cv2.imwrite(output_path, imagesr)

    print(f"SR image save to `{output_path}`")
