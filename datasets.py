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
    
    def __init__(self, root, transform=None, color_ratio=1., if_train=1,):
        
        MAIN_COLOR = "green"
        INSERTED_COLOR = "red"
        
        if if_train:
            data_folder = os.path.join(root, "colorized-MNIST/training")
        else:
            data_folder = os.path.join(root, "colorized-MNIST/testing")
        
        self.dataset_in_digit = torchvision.datasets.ImageFolder(data_folder)
        
        self.color_ratio = color_ratio
        self.transform = transform
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
            
        # Protocol 1, 0-4 -> class 0 (), 5-9 -> class 1
        self.images = []
        self.labels = []
        image_folder_main, label_folder_main = colors_folders[MAIN_COLOR]
        image_folder_inserted, label_folder_inserted = colors_folders[INSERTED_COLOR]
        self.class_zero = [0,1,2,3,4]
        
        num_samples_main = int(len(image_folder_main) * self.color_ratio)
        num_samples_inserted = int(len(image_folder_main) * (1-self.color_ratio))
        
        self.images = image_folder_main[:num_samples_main]
        self.labels = label_folder_main[:num_samples_main]
        
        self.images.extend(image_folder_inserted[:num_samples_inserted])
        self.labels.extend(label_folder_inserted[:num_samples_inserted])
            
        
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
        
        
            
            
if __name__ == "__main__":

    root = "../datasets/colorized-MNIST"
    