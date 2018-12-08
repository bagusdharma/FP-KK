#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:09:54 2018

@author: mis-admin
"""

import cv2
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
import numpy as np
import time
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
import skimage.data
from skimage.color import label2rgb
import scipy
import math

DATA_DIR = os.getcwd()+"/DatasetWajah"
IMG_WIDTH=180
IMG_HEIGHT=200
radius = 3
n_points = 8 * radius

def create_class(image_name):
    word_label=image_name.split('.')
    return word_label[0]

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def load_data():
    X=[]
    y=[]
    for img in tqdm(os.listdir(DATA_DIR)):
        if(img!="Thumbs.db"):
            path=os.path.join(DATA_DIR,img)
            img_data=cv2.imread(path, cv2.IMREAD_COLOR)
            img_data=cv2.resize(img_data,(IMG_WIDTH,IMG_HEIGHT))
            img_gray=cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)            
#            img_2_d=greycomatrix(img_gray, [1], 256, symmetric=False,normed=True)
#            lbp=local_binary_pattern(img_gray, n_points, radius, method='uniform')
#            hist=scipy.stats.itemfreq(lbp)
			
            img_data_flatten=np.array(img_gray).flatten()
#			img_data_flatten=np.array(lbp).flatten()
            X.append(img_data_flatten)
            y.append(create_class(img))
#    print(hist)        
    return X,y

#def overlay_labels(image, lbp, labels):
#    mask = np.logical_or.reduce([lbp == each for each in labels])
#    return label2rgb(mask, image=image, bg_label=0, alpha=0.3)
#
#def hist(ax, lbp):
#    n_bins = int(lbp.max() + 1)
#    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range(0, n_bins),
#                   facecolor='0.5')

def main():
    X,y=load_data()
    KNN = euclideanDistance(X, y, 3)
#	KNN = euclideanDistance(X, y, 3)
    used=cross_val_score(KNN,X,y,cv=10)
    print (used)
    score=used.mean() * 100.00
    print (score)

start_time=time.time()
main()
print(time.time()-start_time)