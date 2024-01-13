import os 

import cv2

DATA_dir = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(1)