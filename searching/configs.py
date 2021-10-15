import os
from glob import glob
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser(description='Input ')

# dataset_name = 'Flickr-8k'
dataset_name = 'V3C1'

if dataset_name == 'Flickr-8k':
    DATASET_NAME = dataset_name
    VBS_MASTER_PATH = '/home/ntnhu/projects/VBS2022'
    DATASET_MASTER_PATH = '/home/ntnhu/projects/object_detection'

    DATASET_PATH = os.path.join(DATASET_MASTER_PATH, 'dataset', DATASET_NAME)
    FEATURE_PATH = os.path.join(VBS_MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not os.path.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        
elif dataset_name == 'V3C1':
    DATASET_NAME = dataset_name
    VBS_MASTER_PATH = '/home/ntnhu/projects/VBS2022'
    # DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
    DATASET_MASTER_PATH = '/mnt/SEAGATE/root/V3C1/resized_keyframes'

    DATASET_PATH = os.path.join(DATASET_MASTER_PATH, 'resized_keyframes')
    FEATURE_PATH = os.path.join(VBS_MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not os.path.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        os.mkdir(os.path.join(FEATURE_PATH, 'ids'))
        os.mkdir(os.path.join(FEATURE_PATH, 'features'))