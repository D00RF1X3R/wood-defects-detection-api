import cv2
import torch
from yolov5 import detect

detect.run(weights='models/wood-detection-model.pt',
           source='img.png',
           project='predictions/',
           conf_thres=0.6,
           exist_ok=True,
           view_img=True)
