import os
import sys
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from scipy import ndimage
from PIL import Image, ImageEnhance
import numpy as np
import cv2
# from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from numpy import vstack
import torch.nn as nn
from torch.optim import Adam, Adagrad
from torch.nn import BCELoss
from tqdm import tqdm
# import torchvision.transforms as transforms
from torch.optim import SGD
from torch import Tensor
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from PIL import Image
# from skimage import filters
# from skimage import exposure
# import skimage
import matplotlib.pyplot as plt
import re
import random
import math
from datetime import datetime
# from unet import Unet
from Task1_Cloud_detection.data import CloudDataset, _Data


class Unet(nn.Module):
    def __init__(self, in_channels= 4, out_channels= 2):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x.float())
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand



def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def detect_cloud(img, plots = True):
    
    tensor = torch.from_numpy(img)
    print(tensor.shape)
    unet = Unet()
    unet_stat = torch.load('Task1_Cloud_detection/unet.pth', map_location=torch.device('cpu'))
    unet.load_state_dict(unet_stat)
    y = unet(tensor[None, :].float())
    print(predb_to_mask(y, 0).shape)
    pi = Image.fromarray(predb_to_mask(y, 0).numpy() * 255,'P')
    # pi.save('result.png')
    # im = Image.fromarray(predb_to_mask(y, 0))
    pi.save("Task2_image_inpainting/lama/output/pic_mask.png")
    # cv2.imwrite("Task2_image_inpainting/lama/output/pic_mask.png", np.expand_dims(predb_to_mask(y, 0), axis = 0))
    return y


