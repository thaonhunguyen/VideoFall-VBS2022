import os
from glob import glob
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser(description='Input ')

dataset_name = 'V3C1'
# dataset_name = 'LSC'

if dataset_name == 'LSC':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
#     DATASET_MASTER_PATH = '/home/ntnhu/projects/object_detection'

    DATASET_PATH = '/mnt/data/lsc2020' 
    FEATURE_PATH = os.path.join(MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not os.path.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)

elif dataset_name == 'V3C1':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
    # DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
    DATASET_MASTER_PATH = '/mnt/SEAGATE/root/V3C1'

    DATASET_PATH = os.path.join(DATASET_MASTER_PATH, 'resized_keyframes')
    FEATURE_PATH = os.path.join(MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not os.path.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        
elif dataset_name == 'V3C':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
    # DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
    DATASET_MASTER_PATH = '/mnt/DEAKIN/VBS2022'

    DATASET_PATH = os.path.join(DATASET_MASTER_PATH, 'resized_keyframes')
    FEATURE_PATH = os.path.join(MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not os.path.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)