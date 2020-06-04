import cv2
import imutils

def resize_no_distortion(img, desired_width, desired_height):
    img = imutils.resize(img, width=desired_width)
    height, width = img.shape[:2]
    if height < desired_height:
        border = (desired_height - height)//2
        img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT)
    elif height > desired_height:
        img = imutils.resize(img, height=desired_height)
        height, width = img.shape[:2]
        border = abs(desired_width - width) // 2
        img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT)

    img = cv2.resize(img, (desired_width, desired_height))

    return img