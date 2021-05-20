import glob
import tag_detection
import markerlib
import os
import numpy as np
from detect import *
# Input
start_time = time.time()
tag_type = 'aruco_4x4'
folder = '4_meter_ny'

camera_orientation = 0

full_std = []
full_mean = []
full_errors_mid = np.zeros((20, 114))
full_errors_wid = np.zeros((20, 114))
boxes_on_planes = np.zeros((20, 10))
for i in range(20):
    images = glob.glob(f'graphics/{folder}/{i}/*.jpg')
    errors_mid = []
    errors_wid = []
    if i == 0:
        continue
    if i == 5:
        continue
    if i == 10:
        continue
    if i == 15:
        continue

    for j, fname in enumerate(images):
        amount_of_boxes = 0
        fname = os.path.basename(fname)
        new_image = tag_detection.Tag(fname, tag_type, i, folder)
        origin = [57.68897633, 11.97869986, 62, camera_orientation + new_image.rotation]
        if new_image.ids is None:
            print("No ids detected")
            continue
        if len(new_image.ids) < 6:
            print("5 or fewer ids detected")
            continue
        if i == 10:
            boxes_on_planes[i, j] = amount_of_boxes
            continue
        # box finding code:
        shelf = markerlib.Shelf(new_image.markers, origin)
        source_img = f'graphics/cv/{i}/{fname}'
        weights = 'ultimate_weights.pt'

        # conf is the confidence threshold of the detection
        # iou_threshold is the area of overlap
        # device, set to '' for gpu, 'cpu' for cpu
        # save_txt=True saves a text file with coordinates
        # save_conf=True adds the confidence level to the coordinate output
        # save_img=True saves the image with boundingboxes

        # for fastest result use device='', save_txt=False, save_conf=True, save_img=False
        coords = detect2(source_img, weights, conf=0.875, iou_thres=0.5, device='', save_txt=False, save_conf=True,
                         save_img=False)

        h, w = new_image.image.shape[:2]
        boxes = []

        for line in coords.split('\n'):
            if line:
                box = markerlib.Box(line, h, w)
                boxes.append(box)
                shelf.add_box_plane(box)

        for plane in shelf.planes:
            error = plane.get_x_error()
            errors_mid.extend(error[0])
            errors_wid.extend(error[1])

        for plane in shelf.planes:
            amount_of_boxes += len(plane.boxes)

        boxes_on_planes[i, j] = amount_of_boxes

    full_errors_mid[i, 0:len(errors_mid)] = errors_mid
    print(full_errors_mid)
    full_errors_wid[i, 0:len(errors_wid)] = errors_wid
    print(full_errors_wid)

for i in range(20):
    temp_err = [err for err in full_errors_mid[i] if err != 0]
    full_errors_mid[i, 0:len(temp_err)] = temp_err
    temp_err = [err for err in full_errors_wid[i] if err != 0]
    full_errors_wid[i, 0:len(temp_err)] = temp_err
print(full_errors_mid)
print(full_errors_wid)
print(boxes_on_planes)

