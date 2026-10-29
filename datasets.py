#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:40:40 2026

@author: zhi
"""


import os
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


mean_mapping = {"colored_mnist": (0.1307, 0.1307, 0)}
std_mapping = {"colored_mnist": (0.3081, 0.3081, 0.3081),}
image_size_mapping = {"colored_mnist": 28,}


class colored_mnist(Dataset):
    
    def __init__(self, root, transform=None, protocol_idx=0, if_train=1, classes=[]):
        
        MAIN_COLOR = "green"
        INSERTED_COLOR = "red"
        
        if if_train:
            data_folder = os.path.join(root, "colorized-MNIST/training")
        else:
            data_folder = os.path.join(root, "colorized-MNIST/testing")
        
        self.dataset_in_digit = torchvision.datasets.ImageFolder(data_folder)

        self.transform = transform
        self.classes = classes
        self.images_blue, self.digit_labels_blue = [], []
        self.images_green, self.digit_labels_green = [], []
        self.images_red, self.digit_labels_red = [], []
        
        colors_map = {"blue": 0, "green": 1, "red": 2}
        colors_folders = {"blue": (self.images_blue, self.digit_labels_blue), 
                          "green": (self.images_green, self.digit_labels_green), 
                          "red": (self.images_red, self.digit_labels_red)}
        
        
        for img_path, digit_label in self.dataset_in_digit.samples:
            
            img = Image.open(img_path).convert("RGB")
            color_key = img_path.split("/")[-1]
            color_key = color_key.split("_")[0]
            
            images_folder, labels_folder = colors_folders[color_key]
            images_folder.append(img)
            labels_folder.append(digit_label)

        self.images = []
        self.labels = []

        if protocol_idx == 0:
            # Protocol 1, 0-4 -> class 0 (), 5-9 -> class 1
            color_ratio = 1.
            image_folder_main, label_folder_main = colors_folders[MAIN_COLOR]
            image_folder_inserted, label_folder_inserted = colors_folders[INSERTED_COLOR]
            self.class_zero = [0, 1, 2, 3, 4]

            num_samples_main = int(len(image_folder_main) * color_ratio)
            num_samples_inserted = int(len(image_folder_main) * (1 - color_ratio))

            self.images = image_folder_main[:num_samples_main]
            self.labels = label_folder_main[:num_samples_main]

            self.images.extend(image_folder_inserted[:num_samples_inserted])
            self.labels.extend(label_folder_inserted[:num_samples_inserted])

        elif protocol_idx == 1:
           # protocol 2, labels follow the digits, all colors exist for individual samples
           # for the sake of equal comparison with the protocol with single color, the datasets
           # is downsampled by 3
           DOWNSAMPLE_RATIO = 3
           for color, (images_color, labels_color) in colors_map.items():
               self.images.append(images_color)
               self.labels.append(labels_color)

           self.images = self.images[::DOWNSAMPLE_RATIO]
           self.labels = self.labels[::DOWNSAMPLE_RATIO]

           if len(self.classes) > 0:
               self.select_classes()

        elif protocol_idx == 2:
            # protocol 3, labels follow the digits, but the colors are the same for all samples
            TARGET_COLOR = "blue"
            images_color, labels_color = colors_map[TARGET_COLOR]
            self.images.extend(images_color)
            self.labels.extend(labels_color)

            if len(self.classes) > 0:
                self.select_classes()
            
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        
        if self.transform is not None: 
            img = self.transform(self.images[idx])
        else:
            img = self.images[idx]
            
        if self.labels[idx] in self.class_zero:
            label = 0
        else:
            label = 1
        
        return img, label


    def select_classes(self):

        self.images_new, self.labels_new = [], []
        for img, label in zip(self.images, self.labels):
            if label in self.classes:
                self.images_new.append(img)
                self.labels_new.append(label)

        self.images, self.labels = self.images_new, self.labels_new
        del self.images_blue, self.digit_labels_blue


        
            
            
if __name__ == "__main__":

    root = "../datasets/colorized-MNIST"
    