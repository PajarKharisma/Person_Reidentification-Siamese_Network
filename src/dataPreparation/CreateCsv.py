import cv2
import pandas
import os

def data_contrastive(images_path='', save_path=''):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(images_path):
	    for file in f:
		    print(file)