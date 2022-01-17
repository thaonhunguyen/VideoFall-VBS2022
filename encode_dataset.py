from dataset_helpers import *
from PIL import Image

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--query', '-q', type=str, default='wedding', help='Input query for searching')
parser.add_argument('--generate_features', '-g', default=True, help='Whether you wanna generate features or not')

def main(args):
    print("Dataset name: ", DATASET_NAME)
    clip = CLIPSearchEngine(src_path=DATASET_PATH, feature_path=FEATURE_PATH, generate_features=args.generate_features)
    print(args.generate_features)
#     dataset = clip.dataset.get_file_name()
    clip.encode_dataset(entire_dataset=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

