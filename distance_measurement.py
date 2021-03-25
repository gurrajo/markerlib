import numpy as np


def get_distance(image, tag_id):
    points = [np.array(get_center_point(image, tag_id[0])), np.array(get_center_point(image, tag_id[1]))]
    distance = np.linalg.norm(points[1][:1] - points[0][:1])
    return points, distance


def get_corner_points(image, tag_id):
    for i in range(len(image.ids)):
        current_id = image.ids[i]
        if current_id == tag_id:
            x = [image.corners[i, 0, 0, 0], image.corners[i, 0, 1, 0],
                 image.corners[i, 0, 2, 0], image.corners[i, 0, 3, 0]]

            y = [image.corners[i, 0, 0, 1], image.corners[i, 0, 1, 1],
                 image.corners[i, 0, 2, 1], image.corners[i, 0, 3, 1]]
            return [x, y, tag_id]


def get_center_point(image, tag_id):
    for counter, i in enumerate(image.ids):
        if i == tag_id:
            x = (image.corners[counter, 0, 0, 0] + image.corners[counter, 0, 1, 0] +
                 image.corners[counter, 0, 2, 0] + image.corners[counter, 0, 3, 0]) / 4

            y = (image.corners[counter, 0, 0, 1] + image.corners[counter, 0, 1, 1] +
                 image.corners[counter, 0, 2, 1] + image.corners[counter, 0, 3, 1]) / 4
            return x, y, counter