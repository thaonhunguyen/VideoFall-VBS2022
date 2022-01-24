import sys
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import KMeans
import utils
import importlib
import random
# np.float128 = np.longdouble
import argparse
from configs.colors import COLOR_POINTS
from tqdm import tqdm, trange
from time import sleep

parser = argparse.ArgumentParser(prog='dominant-color', description='Getting dominant colors')

# Images
parser.add_argument('--indir', default=[], action='append', help='Image Directory')
parser.add_argument('--indir-config', nargs=1, type=str, help='Image Directory Lists')
parser.add_argument('--outdir', nargs=1, default=['./outputs'], type=str, help='List of image colors in the output directory')

# Blur
parser.add_argument('--k-size', nargs=2, type=int, default=[2,2], help='Kernel size for blur')

# Resize
parser.add_argument('--target-size', nargs=2, type=int, default=[60, 60], help='Target size after resizing')

# K Mean
parser.add_argument('--cluster', nargs=1, type=int, default=10, help='Kernel size for blur')
parser.add_argument('--save-img', action='store_true', help='Save K Mean')

args = parser.parse_args()

IMAGES_DIRECTORIES = args.indir
if args.indir_config is not None:
    dirs_from_file = []
    with open(args.indir_config[0], 'r') as f:
        t = f.read()
        dirs_from_file = t.split('\n')
        dirs_from_file = list(filter(None, dirs_from_file))
    # merge 
    IMAGES_DIRECTORIES += dirs_from_file

# Blur
KSIZE = tuple(args.k_size)

# K in k-mean
CLUSTER = (args.cluster)

# Resize target size
TARGET_SIZES = tuple(args.target_size)

# save money to output directory
FLAG_SAVE_IMAGE = args.save_img
OUT_IMAGES_DIRECTORY = args.outdir[0]

# sys.exit(1)
# SHOW_SIZE = 80

def image_to_centroids(origin_image):
    resized_image = cv2.resize(origin_image, TARGET_SIZES)

    img = blurred_image = cv2.blur(resized_image, KSIZE)
    reshaped_image=img.reshape((img.shape[1]*img.shape[0],3))

    kmeans=KMeans(n_clusters=CLUSTER)
    s=kmeans.fit(reshaped_image)

    labels=kmeans.labels_
    labels=list(labels)

    centroid=kmeans.cluster_centers_
    centroid = centroid.astype(np.uint8)

    _labels, count = np.unique(np.array(labels), return_counts=True)
    count = count.tolist()

    new_img = np.asarray(list(map(lambda x: tuple(centroid[x] // 1), labels)), dtype=int)
    img = new_img.reshape((img.shape[0], img.shape[1], 3))
    new_img = img.astype(np.uint8)
    
    return {
        "labels": labels,
        "centroid": centroid,
        "count": count,
        "origin": resized_image,
        "blurred_image": blurred_image,
        "new_img": new_img,
    }

# def get_num(file):
#     m = re.search('_(\d+)_', file)
#     return int(m.group(0)[1:-1])

def validate_folder(folder_to_validate):
    IMAGES_ERROR = False
    total_images = 0
    raw_images = []
    images = []
    all_files_in_directory = os.listdir(folder_to_validate)
    # all_files_in_directory_with_num = list(map(lambda x: (get_number(x), x), dirs))
    
    for img_path in tqdm(all_files_in_directory, desc="validation", leave=False, unit_scale=True):
        if '.png' not in img_path:
            continue
        im = os.path.join(folder_to_validate, img_path)
        try:
            _img = cv2.imread(im)
            raw_images.append(_img)
            images.append(im)
        except e:
            IMAGES_ERROR = True
            raise
            # print(e)
            # print(img_path)
            break
    total_images = len(images)    
    # print(f"TOTAL is {total_images}")
    # sleep(0.01)
        
    if IMAGES_ERROR or total_images == 0:
        sys.exit(1)
        print("boom!")
    return images, raw_images, total_images

# Preparing RGB colors
# R G B => B G R
# colors = list(COLOR_POINTS.items())
colors = []
for k, v in COLOR_POINTS.items():
    for r, g, b in v:
        colors.append([k, [b, g, r]])
        
def closest_color(p1, count, result_map):
    centroid_index, centroid = p1
    # diff = list(map(lambda item: diff_weighted_srgb(centroid, item[1]), colors))
    
    # USE_MANUAL_GRAY_SCALE = False
    USE_MANUAL_GRAY_SCALE = True
    # diff = list(map(lambda item: sum([(x1 - x2) ** 2 for x1, x2 in zip(centroid, item[1])]) ** 2, colors))
    
    # cosin similarity
    diff = list(map(lambda item: dot(item[1], centroid)/(norm(item[1])*norm(centroid) + 0.00001), colors))
    # print(diff)
    
    min_value = max(diff)
    # min_value = min(diff)
    index = diff.index(min_value)
    key = colors[index][0]
    color = colors[index][1]
    
    ### is white, gray, black
    avg = np.average(centroid)
    gray_scale_diff = sum([abs(x - avg) for x in centroid])
    if USE_MANUAL_GRAY_SCALE and gray_scale_diff <= 10:
        # print('what')
        # print(avg, centroid, gray_scale_diff, [x - avg for x in centroid])
        if avg <= 25:
            key = 'black'
            color = [0, 0, 0]
        elif avg >= 240:
            key = 'white'
            color = [255, 255, 255]
        else:
            key = 'gray'
            color = [127, 127, 127]
        
    # key = str(p1[1])
    if key not in result_map:
        result_map[key] = 0
    result_map[key] += count[centroid_index]
#     key, value, count
    return key, tuple(color), tuple(centroid)
    # return key, tuple(p1[1].tolist()), tuple(centroid)

def get_unique_top_n_color(img, n = 5, return_img = False):
    result_map = dict()
    new_img = img.get('new_img')
    clone_img = np.copy(new_img)
    centroid = img.get('centroid')
    # print(centroid, new_img)
    
    color_list = list(
        set(
            tuple(map(lambda centroid: closest_color(centroid, img.get('count'), result_map)
            , enumerate(img.get('centroid'))))
        )
    )

    color_in_images = []
    FLAG = return_img
    # print(new_img)
    
    color_count = []
    for c in color_list:
        key, value, ct = c
        # print(ct)
        clone_img[np.where((new_img==list(ct)).all(axis=2))] = list(value)
        # print(np.unique(clone_img[np.where((new_img==list(ct)).all(axis=2))]))
        # print(np.unique(clone_img[np.where((new_img==list(ct)).all(axis=2))], return_counts=True))
        _img = np.zeros((300, 300, 3), np.uint8)
        _img[:] = list(value)
        color_in_images.append((_img, result_map[key]))
        color_count.append((key, result_map[key]))

    sorted_colors = sorted(color_in_images, key=lambda tup: tup[1], reverse=True)
    
    sorted_color_key = sorted(color_count, key=lambda tup: tup[1], reverse=True)
    s = set()
    for k, v in sorted_color_key:
        if len(s) == 5:
            break
        s.add(k)
    return result_map, sorted_colors, clone_img, s

def main():
    # loop every folder in the big folder
    for folder in tqdm(IMAGES_DIRECTORIES, desc="Each image dir"): 
        # validate the folder
        images, raw_images, total_img = validate_folder(folder)
        # break
        # print('\n'.join(images))
        images_with_centroids = list(map(lambda item: image_to_centroids(item), tqdm(raw_images, desc="image_to_centroid", leave=False, unit_scale=True)))
        # blurred_image = result[0].get('blurred_image')
        # new_img = result[0].get('new_img')

        # list_from_result = [(x.get('origin'), x.get('blurred_image'), x.get('new_img')) for x in tqdm(images_with_centroids, leave=False, unit_scale=True)]
        # flatten_list = [x for t in tqdm(list_from_result, leave=False, unit_scale=True) for x in t]

        # shuffled_list = list_from_result.copy()
        # random.shuffle(shuffled_list)
        # flatten_shuffled_list = [x for t in tqdm(shuffled_list, leave=False, unit_scale=True) for x in t]
        
        colors_result = []
        for image_name, the_image in tqdm(list(zip(images, images_with_centroids)), desc='Color extraction', leave=False, unit_scale=True):
            # the_image = images_with_centroids[i]
            im_kmean = the_image.get('new_img')
            result_map, sorted_colors, new_colored_img, s = get_unique_top_n_color(the_image, 5, return_img = True)
            if FLAG_SAVE_IMAGE:
                path = os.path.join(OUT_IMAGES_DIRECTORY, image_name.replace("./", ""))
                cv2.imwrite(path, im_kmean)
            colors_result.append((image_name, s))
        
        target = os.path.join(OUT_IMAGES_DIRECTORY, f'{folder}/result.txt')
        if not os.path.exists(os.path.dirname(target)):
            try:
                os.makedirs(os.path.dirname(target))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(target, 'w+') as f:
            for cr in colors_result:
                p, s = cr
                s = list(s)
                f.write(str(p) + " :: " + ",".join(s) + "\n")

    # blur
    # kmean
    # print output using tqdm

if __name__ == '__main__':
    main()