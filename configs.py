import os
import os.path as osp
from glob import glob
from tqdm import tqdm

dataset_name = 'V3C'
# dataset_name = 'V3C'

if dataset_name == 'LSC':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
    IMAGE_NAME_PATH = osp.join(MASTER_PATH, 'results', f'{DATASET_NAME}_image_names.joblib')

    DATASET_PATH = '/mnt/data/lsc2020' 
    error_image_file = '/home/ntnhu/projects/LSC2021/LSC_clip_model/error_images.joblib'
    FEATURE_PATH = osp.join(MASTER_PATH, 'results', f'{DATASET_NAME}_features')
    if not osp.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)

elif dataset_name == 'V3C1':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
    # DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
    DATASET_MASTER_PATH = '/mnt/SEAGATE/root/V3C1'
    IMAGE_NAME_PATH = osp.join(MASTER_PATH, 'results', f'{DATASET_NAME}_image_names.joblib')

    DATASET_PATH = osp.join(DATASET_MASTER_PATH, 'resized_keyframes')
    FEATURE_PATH = osp.join(MASTER_PATH, 'results', f'{DATASET_NAME}_features')
    if not osp.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
        
elif dataset_name == 'V3C':
    DATASET_NAME = dataset_name
    MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
    # DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
    DATASET_MASTER_PATH = '/mnt/DEAKIN/VBS2022'
    METADATA_PATH = osp.join(DATASET_MASTER_PATH, 'metadata')
    IMAGE_NAME_PATH = osp.join(METADATA_PATH, f'image_names.joblib')

    KEYFRAME_PATH = osp.join(DATASET_MASTER_PATH, 'keyframes')
    FEATURE_PATH = osp.join(METADATA_PATH, f'{DATASET_NAME}_features')
    if not osp.isdir(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)