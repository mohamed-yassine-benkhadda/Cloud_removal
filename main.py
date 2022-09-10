from osgeo import gdal
import os
import cv2
import numpy as np
from Task1_Cloud_detection.main import detect_cloud as detect
from Task2_image_inpainting.lama.bin.predict import main as inpaint
from Task3_Super_resolution.SRGANPyTorch.inference import main as resolution

def main(path):
    img = gdal.Open(path)
    img = img.ReadAsArray()
    img4detection = img[:4,0:384,0:384]
    print(img4detection.shape)
    img4inpainting = img[[0,3,4],0:384,0:384]
    r = 255 / 7000 
    cv2.imwrite("Task2_image_inpainting/lama/output/pic.png", np.transpose(img4inpainting, (1,2,0)) * r)
    
    detect(img4detection * r)
    inpaint()
    resolution()

main("output/image0polygon1.tif")