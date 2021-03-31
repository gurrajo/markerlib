import csv
import glob
import tag_detection
import distance_measurement
import markerlib
import math
import time
import os
import numpy as np
from detect import *
# Input
start_time = time.time()
filename = '4_0.0_mid.jpg'
tag_type = 'aruco_4x4'


def check_detected_tags():
    images = glob.glob('graphics/images_from_tests/*.jpg')
    for counter, fname in enumerate(images):
        new_image = tag_detection.Tag(fname, tag_type)
        with open('graphics/calibration_output/detected_tags_april.csv', 'a') as file:
            writer = csv.writer(file)
            if new_image.ids is None:
                writer.writerow(f'{fname}NONE')
            else:
                writer.writerow(f"{fname}{new_image.ids}")




new_image = tag_detection.Tag(filename, tag_type)

shelf = markerlib.Shelf(new_image.markers)
# box finding code:
source_img = f'graphics/cv/{filename}'
weights = 'best.pt'


# conf is the confidence threshold of the detection
# iou_threshold is the area of overlap
# device, set to '' for gpu, 'cpu' for cpu
# save_txt=True saves a text file with coordinates
# save_conf=True adds the confidence level to the coordinate output
# save_img=True saves the image with boundingboxes

# for fastest result use device='', save_txt=False, save_conf=True, save_img=False
coords = detect2(source_img, weights, conf=0.7, iou_thres=0.45, device='', save_txt=False, save_conf=True,
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
mean = sum(errors)/12
dev = []
print(errors)
for err in errors:
    dev.append((err-mean)**2)
var = sum(dev)/mean
std = np.sqrt(var)
print(f'average absolute error:{mean} in mm')
print(f'standard deviation: {std}')


#shelf.disp_planes(new_image, boxes)
# turn coords into box objects:
#shelf.add_box_plane(box)
# shelf information:
print(shelf)

shelf.disp_planes(new_image, boxes)