'''
Contains modules to preprocess data

Takes the segmentation labels and reduces them in size and saves them to disk
'''

import os
import re
import gc 
import pickle as pl
import numpy as np
import cv2

# ============================================================================================
# This code takes in the dataset, reduces it's memory footprint and saves it
with open("../segmentation_dataset/labels_3000_original_1.p","rb") as f:
    labels = pl.load(f)

def process_labels(labels):
    ''' Changes label image to have a green background. Drivable areas remain red 
        and side lanes remain blue
        '''
    processed_labels = []
    for label in labels:
        label_copy = np.unpackbits(label, count=2764800).reshape((720, 1280, 3)).copy()
        label_copy[:, :, 1] = np.logical_not(np.logical_or(label_copy[:, :, 0], 
                                                           label_copy[:, :, 2]))
        processed_labels.append(label_copy)

        gc.collect()

    return processed_labels


def reduce_labels(labels):
    ''' Gets smaller size label prepresentation. Overwrites each object in label list,
        To plot in matplotlib convert label to numpy float 32 or multiply by 255
        '''
    for i, label in enumerate(labels):
        labels[i] = np.packbits(labels[i].astype(np.bool_))    

def reduce_label_data(label_path):
    with open(label_path, 'rb') as f:
        labels = pl.load(f)
    
    # reduce memory via dtype and packbits
    for i, label in enumerate(labels):
        labels[i] = np.packbits(labels[i].astype(np.bool_))

    # save to new location
    label_path_save = re.sub('\.p', '_REDUCED.p', label_path)
    with open(label_path_save, 'wb') as f_obj:
        pl.dump(labels, f_obj)


def random_rotate(image, label, angle=45):
    ''' Applys random rotation to image and label. Leaves part of the the 
        image cropped out which is desireable for this scenario since
        it further diversifies the data
        '''
    center = (image.shape[1] // 2, image.shape[0] // 2)
    angle = np.random.uniform(-angle, angle)
    rot = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_image = cv2.warpAffine(image, rot, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotated_label = cv2.warpAffine(label, rot, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image, rotated_label