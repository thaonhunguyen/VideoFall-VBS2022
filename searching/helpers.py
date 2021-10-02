import torch
import clip
import cv2
import sys
import os
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
    # CLUSTER = None
    # IMAGE = None
    # FLAT_IMAGE = None
    # COLORS = None
    # LABELS = None
    
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


def plot_figure(images, subplot_size=(5, 3), savefig=False, fig_name="", src_path=None):
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
        fig.savefig(os.path.join(src_path, f'{fig_name}.png'))
        
def convert_to_concepts(string):
    name = string.split('/')[-1]
    components = name.split('_')
    dataset = None
#     dataset = components[-4]
    video = components[-3][4:]
    shot = components[-2]
    return {'path': string,
            'filename': name,
            'dataset': dataset,
            'video': video,
            'shot': shot}


# img = '../filtering/google.jpeg'
# img = cv2.imread(img)
# #convert to RGB from BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# dominant = DominantColors(img, cluster=5)
# color = dominant.find_dominant_colors()
# print(color)
