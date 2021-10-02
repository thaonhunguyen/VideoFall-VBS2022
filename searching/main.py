from dataset_helpers import *
from PIL import Image

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--query', '-q', type=str, default='wedding', help='Input query for searching')

def main(args):
    data = dataset(src_path=DATASET_PATH, feature_path=FEATURE_PATH)
    data.get_file_name()
    data.preprocess_dataset(entire_dataset=False)
    print("Features: ", data.features)
    pass

if __name__ == '__main__':
    args = None
    main(args)

