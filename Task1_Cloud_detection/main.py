import os
import sys
# from google.colab import drive
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
from sklearn.metrics import roc_curve, auc
from scipy import ndimage
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from numpy import vstack
import torch.nn as nn
from torch.optim import Adam, Adagrad
from torch.nn import BCELoss
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.optim import SGD
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from PIL import Image
from skimage import filters
from skimage import exposure
import skimage
import matplotlib.pyplot as plt
import re
import random
import math
from datetime import datetime

from .model import UNET
from .data import CloudDataset, _Data

def detect_cloud(path):
    img = cv2.imread(path)
