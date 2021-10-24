import torch
import clip
import cv2
import sys
import re 
import os
import os.path as osp
import matplotlib.pyplot as plt

from typing import List
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans

# Filtering functions
class DominantColors():
    '''
    Class of dominant colors of images
    '''
    def __init__(self, image, cluster=5):
        self.cluster = cluster
        self.image = image
        self.flat_image = None
        self.colors = None
        self.labels = None

    def dominant_colors(self):
        '''
        Determine dominant colors of images using Kmeans clustering algorithm

        return:
           - color
        '''
        img = self.image
        # Reshape to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # Save image after flattening
        self.flat_image = img
        
        # Using KMeans to cluster pixels
        kmeans = KMeans(n_clusters = self.cluster)
        kmeans.fit(img)
        
        # Get the colors as per dominance ordering
        self.colors = kmeans.cluster_centers_
        # Save labels
        self.labels = kmeans.labels_

        return self.colors


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def sort_list(input_list):
    '''
    Function to convert an input list into a list in ascending order based on natual keys

    params:
        - input_list: Listß
    '''
    input_list.sort(key=natural_keys)
    return input_list

def is_image(input_string):
    extension = input_string.split('.')[-1]
    if extension == 'png' or extension == 'jpg' or extension == 'jpeg' or extension == 'JPG':
        return True
    return False

def plot_figures(images: List, subplot_size=(5, 3), savefig=False, fig_name="", src_path=None):
    '''
    Function to plot a list of images

    params:
        - images: List,
            A list of images to visualize
    
    return:
       - 
    '''
    fig = plt.figure(figsize=(15, 15))
    max_len = min(len(images), subplot_size[0]*subplot_size[1])

    for cnt, data in enumerate(images[:max_len]):
        y = fig.add_subplot(subplot_size[0], subplot_size[1], cnt+1)
        img = Image.open(data)
        y.imshow(img)
        plt.title(data.split('/')[-1])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    if savefig:
        fig.savefig(osp.join(src_path, f'{fig_name}.png'))
        
        
def convert_to_concepts(image_name: str, dataset_name: str):
    '''
    Function to convert a string into the concepts having the detail of each image in VBS dataset
    
    params:
        - name: type, default=
            description
    
    return:
       - 
    '''
    if dataset_name == 'LSC':
        filename = image_name.split('/')[-2:]
        name = osp.join(filename[0], filename[1])
#         name = image_name.split('/')[-1]
        date = filename[0]
#         time = name.split('_')[-2]
        concepts = {'path': image_name, 'filename': name, 'date': date}
    elif dataset_name == 'V3C1' or dataset_name == 'V3C':
        name = image_name.split('/')[-1]
        components = name.split('_')
        dataset = None
        video = components[-3][4:]
        shot = components[-2]
        concepts = {'path': image_name, 'filename': name, 'dataset': dataset, 'video': video, 'shot': shot}
    return concepts
        

# img = '../filtering/google.jpeg'
# img = cv2.imread(img)
# #convert to RGB from BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# dominant = DominantColors(img, cluster=5)
# color = dominant.find_dominant_colors()
# print(color)
