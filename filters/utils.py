import matplotlib.pyplot as plt
import cv2

def show_images(images, layout=(1, 2), figsize=(12, 6), titles=[]):
    h, w = layout
    total = h * w
    total_image = len(images)
    total_title = len(titles)
    if total_image > total:
        raise Exception("Not match number!")
    fig = plt.figure(figsize=figsize)
    fig.tight_layout(pad=3.0)
    for index, im in enumerate(images):
        ax = fig.add_subplot(h, w, index + 1)
        imgplot = plt.imshow(im)
        if index < total_title:
            ax.set_title(titles[index])
        else:
            ax.set_title(str(index))
        # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    return fig