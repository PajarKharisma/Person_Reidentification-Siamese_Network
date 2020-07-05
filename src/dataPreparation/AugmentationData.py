import cv2

TOP = 0
BOTTOM = 1

def create_aug_data(img, pos, color, occlusion):
    if pos == BOTTOM:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    height, width = img.shape[:2]

    for i in range(int(height * occlusion)):
        for j in range(width):
            img[i,j] = color

    if pos == BOTTOM:
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img