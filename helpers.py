import re 
import json
import os
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import datetime

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

def sort_dict(input_dict, by_value=True, descending=False):
    index = 1
    if not by_value:
        index = 0
    return dict(sorted(input_dict.items(), key=lambda item: item[index], reverse=descending))

def plot_figures(images: List, figsize=(15, 15), subplot_size=(5, 3), savefig=False, fig_name="", src_path=None):
    '''
    Function to plot a list of images

    params:
        - images: List,
            A list of images to visualize
    
    return:
       - 
    '''
    fig = plt.figure(figsize=figsize)
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
        
        
def convert_to_concepts(image_name: str, dataset_name='V3C'):
    '''
    Function to convert a string into the concepts having the detail of each image in VBS dataset
    
    params:
        - name: type, default=
            description
    
    return:
       - 
    '''
    if dataset_name == 'V3C1' or dataset_name == 'V3C':
        name = image_name.split('/')[-1]
        components = name.split('_')
        dataset = None
        video = components[-3][4:]
        shot = components[-2]
        concepts = {'path': image_name, 'filename': name, 'dataset': dataset, 'video': video, 'shot': shot}
    return concepts
        
def save_df_to_json(data_df, filename, orient='records', indent=4):
    data = data_df.to_json(orient=orient) 
    parsed_data = json.loads(data)    
    with open(filename, 'w') as f:
        json.dump(parsed_data, f, indent=indent)

def save_list_to_csv(obj: List, filename, delimiter='\n'):
    with open(filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(obj)

def load_json(json_file):
    with open(json_file, 'r') as handle:
        text_data = handle.read()
        text_data = '[' + re.sub(r'\}\s\{', '},{', text_data) + ']'
        json_data = json.loads(text_data)
        data = json_data[0]
    return data

def resize_image(img_path, scale_percent=50, rename=False, filename=None):
    src_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    scale_percent = scale_percent

    #calculate the 50 percent of original dimensions
    width = int(src_img.shape[1] * scale_percent / 100)
    height = int(src_img.shape[0] * scale_percent / 100)
    dsize = (width, height)

    # resize image
    output = cv2.resize(src_img, dsize)
    
    if rename:
        cv2.imwrite(filename, output)
        return
        
    return output

def time_this(func):
    def calc_time(*args, **kwargs):
        before = datetime.datetime.now()
        x = func(*args, **kwargs)
        after = datetime.datetime.now()
        print("Function {} elapsed time: {}".format(func.__name__, after-before))
        return x
    return calc_time