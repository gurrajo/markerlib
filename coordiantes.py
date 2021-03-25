
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2
import os

# returns the image data from a file-path


def load_img_from_file(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# input: an image to locate boxes in
# output: list of coordinates in the format [boxType, x1, y1, x2, y2]
def locate_boxes(img, amount_boxes=3, xywh_format=False):
    # TODO: use computervision model to locate boxes...

    # following code is temporary

    res = []

    (hImg, wImg) = img.shape[:2]

    for box in range(0, 3):
        boxType = int(random.randint(0, 3))
        w = random.randint(48, 256)
        h = random.randint(64, 384)
        x = random.randint(int(w/2), int(wImg - w/2 - 1))
        y = random.randint(int(h/2), int(hImg - h/2 - 1))

        if xywh_format:
            res.append([boxType, x, y, w, h])
        else:
            res.append([boxType,
                        int(x-w/2), int(y-h/2),
                        int(x+w/2), int(y+h/2)])

    print(res)
    return res


# draws the boundingboxes and displays them
# accepts coordinates in the format [boxType, x, y, w, h]
# if you want to save to file, add a path
def display_boxes(img, boxes, path=""):

    # creates a new figure (ax are like subplots in matlab)
    fig, ax = plt.subplots()

    # extract dimensions
    (hImg, wImg) = img.shape[:2]

    # display the image in the figure without the axis
    ax.imshow(img)
    ax.axis('off')

    # draws every box
    for box in boxes:
        w = box[3]
        h = box[4]
        x = int(box[1]-w/2)
        y = int(box[2]-h/2)

        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate((str(box[0]) + ", x: " + str(x) + ", y: " + str(y)),
                    (x, y-16),
                    color='w',
                    weight='bold',
                    bbox=dict(alpha=.5, fc="k"),
                    fontsize=6,
                    ha='left',
                    va='bottom')

    if path != "":
        plt.savefig(path[:-4] + "_annotated" + path[-4:], bbox_inches='tight')

    plt.show()


#base_dir = 'yolo_dataset/images/train'
#imgFile = 'IMG_3654_JPG.rf.f4311000e64ae0d17e346a006a9a3e23.jpg'

base_dir = "./"
imgFile = "IMG_3711.JPG"

path = os.path.join(base_dir, imgFile)


img = load_img_from_file(path)

boxes = locate_boxes(img, amount_boxes=3, xywh_format=True)

display_boxes(img, boxes)  # , path)
