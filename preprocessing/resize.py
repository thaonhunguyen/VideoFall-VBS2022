import cv2

def resize(img_path, scale_percent=50):
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    scale_percent = scale_percent

    #calculate the 50 percent of original dimensions
    width = int(src_img.shape[1] * scale_percent / 100)
    height = int(src_img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src_img, dsize)
    return output