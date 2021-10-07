import cv2
import os.path as osp
from tqdm import tqdm


def resize(img_path, scale=0.5, path=None):
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale)
    height = int(src.shape[0] * scale)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)
    print(output.shape)
    
    name = img_path.split('/')[-1]
    filename = osp.join(path, name)
    print(filename)
#     return output

# def resize_images(image_paths, scale, path=None):
# #     for img in tqdm(images):
#     name = image_paths.split('/')[-1]
#     filename = osp.join(path, name)
#     print(filename)
#     output = resize(image_paths, scale)
# #         cv2.imwrite(filename)