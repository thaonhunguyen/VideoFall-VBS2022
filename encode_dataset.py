from dataset_helpers import *
from PIL import Image

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--query', '-q', type=str, default='wedding', help='Input query for searching')
parser.add_argument('--generate_features', '-g', default=True, help='Whether you want to generate features or not')
parser.add_argument('--feature_path', '-fp', help='Input the directory where you want to store the feature files')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Input batch size')

def main(args):
    print("Dataset name: ", DATASET_NAME)
    clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=args.feature_path, batch_size=args.batch_size, generate_features=args.generate_features)
    print(args.generate_features)
#     dataset = clip.dataset.get_file_name()
    clip.encode_dataset(entire_dataset=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

