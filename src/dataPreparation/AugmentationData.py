import cv2
import random

TOP = 0
BOTTOM = 1

def create_aug_data(img, occlusion, pos):
    if pos == BOTTOM:
        img = cv2.flip(img, 0)
    
    height, width = img.shape[:2]

    for i in range(int(height * occlusion)):
        for j in range(width):
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            img[i,j] = (b, g, r)

    if pos == BOTTOM:
        img = cv2.flip(img, 0)

    return img