import cv2
import os
import os.path as osp
import argparse
import joblib
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--src_path', '-src', type=str, default='/mnt/', help='Input source path where the dataset is stored')
parser.add_argument('--des_path', '-des', type=str, default='/mnt/DEAKIN/VBS2022/resized_keyframes', help='Input destination path where you want to store new dataset')
parser.add_argument('--scale', '-s', type=float, default=0.25, help='Input scale percentage that you want to resize the image (e.g. 0.5)')
parser.add_argument('--imwrite_', '-i', default=True)


def resize(img_path, scale=0.5, path=None, imwrite_=True):
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale)
    height = int(src.shape[0] * scale)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)
    
    name = img_path.split('/')[-1]
    filename = osp.join(path, name)
    
    if imwrite_:
        cv2.imwrite(filename, output)
        
def processing(args):
#     filename_list = glob(osp.join(args.src_path, '*/*.png'))
    filename_list = joblib.load('V3C2_diff.joblib')
    print("Number of images to process: ", len(filename_list))
    print("Processing ...")
    for filename in tqdm(filename_list):
#         print(filename)
        video_name = filename.split('/')[-1][4:9]
        video_path = osp.join('/mnt/DEAKIN/VBS2022/keyframes/', video_name)
        if not osp.isdir(video_path):
            os.mkdir(video_path)
#     for filename in tqdm(filename_list):
        resize(filename, scale=args.scale, path=video_path, imwrite_=args.imwrite_)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.imwrite_)
    processing(args)
#     print(args.src_path, args.des_path, args.scale, args.imwrite_)