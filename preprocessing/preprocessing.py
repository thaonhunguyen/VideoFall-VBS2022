import cv2
import os.path as osp
import argparse

from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--src_path', '-src', type=str, default='/mnt/', help='Input source path where the dataset is stored')
parser.add_argument('--des_path', '-des', type=str, default='/mnt/', help='Input destination path where you want to store new dataset')
parser.add_argument('--scale', '-s', type=float, default=0.5, help='Input scale percentage that you want to resize the image (e.g. 0.5)')
parser.add_argument('--imwrite_', '-i', type=bool, default=True)


def resize(img_path, scale=0.5, path=None, imwrite_=False):
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
        
def preprocessing(args):
    filename_list = glob(args.src_path)
    print(len(filename_list))
    print("Preprocessing ...")
    for filename in tqdm(filename_list[:10]):
        print(filename)
        resize(filename, scale=args.scale, path=args.des_path, imwrite_=args.imwrite_)

if __name__ == '__main__':
    args = parser.parse_args()
    preprocessing(args)
#     print(args.src_path, args.des_path, args.scale, args.imwrite_)