import numpy as np
import cv2

# cv2.BORDER_CONSTANT
# cv2.BORDER_REFLECT
# cv2.BORDER_REFLECT_101
# cv2.BORDER_DEFAULT
# cv2.BORDER_REPLICATE
# cv2.BORDER_WRAP

def padding(img):
    rows, cols = img.shape[:2]
    border = abs(rows -cols) // 2
    if(rows > cols):
        img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT)
    else:
        img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT)
    return img


def main():
    img = cv2.imread('img/person.jpg')
    rows, cols = img.shape[:2]
    img = cv2.resize(img, (cols*2, rows*2))
    rows *= 2
    cols *= 2

    img1 = np.zeros((rows//8, cols,3),np.uint8)*255
    img2 = np.zeros(((rows//8) * 3, cols,3),np.uint8)*255
    img3 = np.zeros(((rows//8) * 4, cols,3),np.uint8)*255

    for i in range(rows//8):
        for j in range(cols):
            color = img[i,j]
            img1[i,j] = color


    for i in range(rows//8 * 3):
        for j in range(cols):
            color = img[i + rows//8, j]
            img2[i,j] = color

    for i in range(rows//8 * 4):
        for j in range(cols):
            color = img[i + (rows//8 * 4), j]
            img3[i,j] = color


    img1 = padding(img1)
    img2 = padding(img2)
    img3 = padding(img3)

    cv2.imshow('person', img)
    cv2.imshow('kepala', img1)
    cv2.imshow('badan', img2)
    cv2.imshow('kaki', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()