import csv
import glob
import tag_detection
import markerlib
import numpy as np
from detect import *

# Input
start_time = time.time()
filename = 'snapshot_2021_04_29_11_41_03.jpg'
tag_type = 'aruco_4x4'
folder = 2

new_image = tag_detection.Tag(filename, tag_type, folder)
camera_orientation = 0
origin = [57.68897633, 11.97869986, 62, camera_orientation + new_image.rotation]
shelf = markerlib.Shelf(new_image.markers, origin)

# box finding code:
source_img = f'graphics/cv/{folder}/{filename}'
weights = 'ultimate_weights.pt'


# conf is the confidence threshold of the detection
# iou_threshold is the area of overlap
# device, set to '' for gpu, 'cpu' for cpu
# save_txt=True saves a text file with coordinates
# save_conf=True adds the confidence level to the coordinate output
# save_img=True saves the image with boundingboxes

# for fastest result use device='', save_txt=False, save_conf=True, save_img=False
coords = detect2(source_img, weights, conf=0.825, iou_thres=0.5, device='', save_txt=False, save_conf=True,
                 save_img=False)

h, w = new_image.image.shape[:2]
boxes = []
for line in coords.split('\n'):
    if line:
        box = markerlib.Box(line, h, w)
        boxes.append(box)
        shelf.add_box_plane(box)

errors = []
for plane in shelf.planes:
    error = np.reshape(plane.get_x_error(), (1, -1)).tolist()
    errors.extend(error[0])

    print(f'plane ID:{plane.plane_id} absolute error: {error[0]} in mm')
mean = np.mean(errors)
dev = []
print(errors)
for err in errors:
    dev.append((err-mean)**2)
print(f'average error:{mean} in mm')



# shelf information:
print(shelf)
shelf.redis_send()
shelf.disp_planes(new_image, boxes)

