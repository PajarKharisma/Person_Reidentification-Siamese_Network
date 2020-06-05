import cv2
import imutils
import os
import numpy as np
from ..utils import MatrixOps as matOps

NUM_PARTIAL = 16
IMG_WIDTH = 64
IMG_HEIGHT = 64

def padding(img):
    rows, cols = img.shape[:2]
    border = abs(rows -cols) // 2
    if(rows > cols):
        img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT)
    else:
        img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT)
    return img

def partial_image(img):
    rows, cols = img.shape[:2]

    img_head = np.zeros(((rows//NUM_PARTIAL) * 3, cols,3),np.uint8)*255
    img_body = np.zeros(((rows//NUM_PARTIAL) * 5, cols,3),np.uint8)*255
    img_leg = np.zeros(((rows//NUM_PARTIAL) * 8, cols,3),np.uint8)*255

    for i in range(rows//NUM_PARTIAL * 3):
        for j in range(cols):
            color = img[i,j]
            img_head[i,j] = color


    for i in range(rows//NUM_PARTIAL * 5):
        for j in range(cols):
            color = img[i + (rows//NUM_PARTIAL * 3) , j]
            img_body[i,j] = color

    for i in range(rows//NUM_PARTIAL * 8):
        for j in range(cols):
            color = img[i + (rows//NUM_PARTIAL * 8), j]
            img_leg[i,j] = color
    
    img_head = cv2.resize(img_head, (IMG_WIDTH, IMG_HEIGHT))
    img_body = cv2.resize(img_body, (IMG_WIDTH, IMG_HEIGHT))
    img_leg = cv2.resize(img_leg, (IMG_WIDTH, IMG_HEIGHT))

    # img_head = matOps.resize_no_distortion(img=img_head, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)
    # img_body = matOps.resize_no_distortion(img=img_body, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)
    # img_leg = matOps.resize_no_distortion(img=img_leg, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)

    return img_head, img_body, img_leg

def create_data(images_path='', head_path='', body_path='', leg_path=''):
    
    for r, d, f in os.walk(images_path):
        for index, file in enumerate(f):
            file_name = os.path.join(r, file)
            img = cv2.imread(file_name)
            
            img_head, img_body, img_leg = partial_image(img)

            cv2.imwrite(head_path + file, img_head)
            cv2.imwrite(body_path + file, img_body)
            cv2.imwrite(leg_path + file, img_leg)
