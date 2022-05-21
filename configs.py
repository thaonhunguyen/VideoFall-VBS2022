import os
import os.path as osp
from glob import glob
from tqdm import tqdm

model_name = 'L14'
# model_name = 'L14_336'
# model_name = 'B32'

DATASET_NAME = 'V3C'
MASTER_PATH = '/home/ntnhu/projects/VideoFall-VBS2022'
# DATASET_MASTER_PATH = '/home/SharedFolder/VBS2021/dataset'
DATASET_MASTER_PATH = '/mnt/deakin/VBS2022'
METADATA_PATH = osp.join(DATASET_MASTER_PATH, 'metadata')
EMBEDDING_PATH = osp.join(DATASET_MASTER_PATH, 'embedding_features')
IMAGE_NAME_PATH = osp.join(METADATA_PATH, f'image_names.joblib')

KEYFRAME_PATH = osp.join(DATASET_MASTER_PATH, 'keyframes')
if model_name == 'B32':
    FEATURE_FILENAME_PATH = osp.join(EMBEDDING_PATH, f'B32_features_512_filenames.joblib')
    FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'B32_features_512.pkl')
    FEATURE_PATH = osp.join(EMBEDDING_PATH, f'B32_features_512')
elif model_name == 'L14':
    FEATURE_FILENAME_PATH = osp.join(EMBEDDING_PATH, f'L14_features_512_filenames.joblib')
    FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'L14_features_512.pkl')
    FEATURE_PATH = osp.join(EMBEDDING_PATH, f'L14_features_512')
elif model_name == 'L14_336':
    FEATURE_FILENAME_PATH = osp.join(EMBEDDING_PATH, f'L14_336_features_128_filenames.joblib')
    FEATURE_DICT_PATH = osp.join(EMBEDDING_PATH, f'L14_336_features_128.pkl')
    FEATURE_PATH = osp.join(EMBEDDING_PATH, f'L14_336_features_128')

if not osp.isdir(FEATURE_PATH):
    os.mkdir(FEATURE_PATH)