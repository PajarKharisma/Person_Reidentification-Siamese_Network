import cv2
import imutils
import os
import numpy as np
from ..utils import MatrixOps as matOps

def padding(img):
    rows, cols = img.shape[:2]
    border = abs(rows -cols) // 2
    if(rows > cols):
        img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT)
    else:
        img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT)
    return img

def partial_image_1(img):
    NUM_PARTIAL = 16
    IMG_WIDTH = 64
    IMG_HEIGHT = 64

    rows, cols = img.shape[:2]

    part_rows = rows//NUM_PARTIAL
    img_1_rows = part_rows * 3
    img_2_rows = part_rows * 5
    img_3_rows = part_rows * 8

    img_1 = np.zeros((img_1_rows, cols,3),np.uint8)*255
    img_2 = np.zeros((img_2_rows, cols,3),np.uint8)*255
    img_3 = np.zeros((img_3_rows, cols,3),np.uint8)*255

    for i in range(img_1_rows):
        for j in range(cols):
            color = img[i,j]
            img_1[i,j] = color


    for i in range(img_2_rows):
        for j in range(cols):
            color = img[i + img_1_rows , j]
            img_2[i,j] = color

    for i in range(img_3_rows):
        for j in range(cols):
            color = img[i + img_1_rows + img_2_rows, j]
            img_3[i,j] = color
    
    img_1 = cv2.resize(img_1, (IMG_WIDTH, IMG_HEIGHT))
    img_2 = cv2.resize(img_2, (IMG_WIDTH, IMG_HEIGHT))
    img_3 = cv2.resize(img_3, (IMG_WIDTH, IMG_HEIGHT))

    # img_head = matOps.resize_no_distortion(img=img_head, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)
    # img_body = matOps.resize_no_distortion(img=img_body, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)
    # img_leg = matOps.resize_no_distortion(img=img_leg, desired_width=IMG_WIDTH, desired_height=IMG_HEIGHT)

    return (img_1, img_2, img_3)

def partial_image_2(img):
    NUM_PARTIAL = 9
    IMG_WIDTH = 64
    IMG_HEIGHT = 64

    rows, cols = img.shape[:2]

    part_rows = rows//NUM_PARTIAL
    img_1_rows = part_rows * 3
    img_2_rows = part_rows * 3
    img_3_rows = part_rows * 3

    img_1 = np.zeros((img_1_rows, cols,3),np.uint8)*255
    img_2 = np.zeros((img_2_rows, cols,3),np.uint8)*255
    img_3 = np.zeros((img_3_rows, cols,3),np.uint8)*255

    for i in range(img_1_rows):
        for j in range(cols):
            color = img[i,j]
            img_1[i,j] = color


    for i in range(img_2_rows):
        for j in range(cols):
            color = img[i + img_1_rows , j]
            img_2[i,j] = color

    for i in range(img_3_rows):
        for j in range(cols):
            color = img[i + img_1_rows + img_2_rows, j]
            img_3[i,j] = color
    
    img_1 = cv2.resize(img_1, (IMG_WIDTH, IMG_HEIGHT))
    img_2 = cv2.resize(img_2, (IMG_WIDTH, IMG_HEIGHT))
    img_3 = cv2.resize(img_3, (IMG_WIDTH, IMG_HEIGHT))

    return (img_1, img_2, img_3)

def partial_image_3(img):
    NUM_PARTIAL = 9
    IMG_WIDTH = 64
    IMG_HEIGHT = 64

    rows, cols = img.shape[:2]

    part_rows = rows//NUM_PARTIAL
    img_1_rows = part_rows * 3
    img_2_rows = part_rows * 3
    img_3_rows = part_rows * 3
    img_4_rows = part_rows * 3

    img_1 = np.zeros((img_1_rows, cols,3),np.uint8)*255
    img_2 = np.zeros((img_2_rows, cols,3),np.uint8)*255
    img_3 = np.zeros((img_3_rows, cols,3),np.uint8)*255
    img_4 = np.zeros((img_4_rows, cols,3),np.uint8)*255

    for i in range(img_1_rows):
        for j in range(cols):
            color = img[i,j]
            img_1[i,j] = color


    for i in range(img_2_rows):
        for j in range(cols):
            color = img[(i + img_1_rows) - part_rows, j]
            img_2[i,j] = color

    for i in range(img_3_rows):
        for j in range(cols):
            color = img[(i + img_1_rows + img_2_rows) - (2 * part_rows), j]
            img_3[i,j] = color

    for i in range(img_3_rows):
        for j in range(cols):
            color = img[(i + img_1_rows + img_2_rows + img_3_rows) - (3 * part_rows), j]
            img_4[i,j] = color
    
    img_1 = cv2.resize(img_1, (IMG_WIDTH, IMG_HEIGHT))
    img_2 = cv2.resize(img_2, (IMG_WIDTH, IMG_HEIGHT))
    img_3 = cv2.resize(img_3, (IMG_WIDTH, IMG_HEIGHT))
    img_4 = cv2.resize(img_4, (IMG_WIDTH, IMG_HEIGHT))

    return (img_1, img_2, img_3, img_4)

def create_data(images_path, save_path):
    for r, d, f in os.walk(images_path):
        for index, file in enumerate(f):
            file_name = os.path.join(r, file)
            img = cv2.imread(file_name)
            
            imgs = partial_image_3(img)

            for i, path in enumerate(save_path):
                # print('file saved in ' + path + file)
                cv2.imwrite(path + file, imgs[i])
