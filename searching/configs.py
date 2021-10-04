import os.path as osp
from glob import glob
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser(description='Input ')

# dataset_name = 'Flicker-8k'
dataset_name = 'V3C1'

if dataset_name == 'Flicker-8k':
    DATASET_NAME = dataset_name
    VBS_MASTER_PATH = '/home/ntnhu/DCU/projects/VBS2022'
    DATASET_MASTER_PATH = '/home/ntnhu/DCU/projects/object_detection'

    DATASET_PATH = osp.join(DATASET_MASTER_PATH, 'dataset', DATASET_NAME)
    FEATURE_PATH = osp.join(VBS_MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not osp.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        
elif dataset_name == 'V3C1':
    DATASET_NAME = dataset_name
    VBS_MASTER_PATH = '/home/ntnhu/DCU/projects/VBS2022'
    DATASET_MASTER_PATH = '/mnt/SEAGATE/root/V3C1'

    DATASET_PATH = osp.join(DATASET_MASTER_PATH, 'resized_keyframes')
    FEATURE_PATH = osp.join(VBS_MASTER_PATH, 'results', f'{DATASET_NAME}-features')
    if not osp.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        os.mkdir(osp.join(FEATURE_PATH, 'ids'))
        os.mkdir(osp.join(FEATURE_PATH, 'features'))