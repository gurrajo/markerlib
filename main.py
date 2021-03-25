import csv
import glob
import tag_detection
import distance_measurement
import markerlib
import math
import time
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


#check_detected_tags()
new_image = tag_detection.Tag(filename, tag_type)
new_image.draw_tags()
markers = [distance_measurement.get_corner_points(new_image, i) for i in range(len(new_image.ids))]
shelf = markerlib.Shelf(markers)
# box finding code:
source_img = f'graphics/calibration_output/{filename}'
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
# turn coords into box objects:
#shelf.add_box_plane(box)
# shelf information:
print(shelf)
for i in range(3):
    print(shelf.planes[i].get_plane_distance())
end_time = time.time()
# Plot plane areas:
for plane in shelf.planes:
    cv2.line(new_image.image, (math.floor(plane.x[0]), math.floor(plane.y)),
             (math.floor(plane.x[1]), math.floor(plane.y)), (255, 0, 0), 2)

# Plot boxes:
for box in boxes:
    cv2.line(new_image.image, (math.floor(box.x[0]), math.floor(box.y[0])), (math.floor(box.x[1]), math.floor(box.y[0])), (0, 255, 0), 2)
    cv2.line(new_image.image, (math.floor(box.x[1]), math.floor(box.y[0])), (math.floor(box.x[1]), math.floor(box.y[1])), (0, 255, 0), 2)
    cv2.line(new_image.image, (math.floor(box.x[1]), math.floor(box.y[1])), (math.floor(box.x[0]), math.floor(box.y[1])), (0, 255, 0), 2)
    cv2.line(new_image.image, (math.floor(box.x[0]), math.floor(box.y[1])), (math.floor(box.x[0]), math.floor(box.y[0])), (0, 255, 0), 2)
cv2.imshow('Tags', new_image.image)
cv2.imwrite("graphics/test.jpg", new_image.image)
cv2.waitKey(0)
