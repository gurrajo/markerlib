import numpy as np


def get_distance(image, tag_id1,tag_id2):
    point1= np.array(get_center_point(image,tag_id1))
    point1= point1[:-1]
    point2= np.array(get_center_point(image,tag_id2))
    point2 = point2[:-1]
    distance = np.linalg.norm(point1-point2)
    return distance

#Ger avstånd i mm
def get_shelf_distance_irl(image,shelf):
    if shelf == 0:
        tag1=0
        tag2=1
    elif shelf ==1:
        tag1=2
        tag2=3
    elif shelf ==2:
        tag1=4
        tag2=5
    distance = get_distance(image, tag1, tag2)
    pixel_IRL = (get_average_length(image, tag1) + get_average_length(image, tag2)) / 2
    tag_size_irl=99
    return (distance/pixel_IRL)*tag_size_irl

#Räknar ut hur många pixlar sidan på en tag är(Average)
def get_average_length(image,tag_id):
    corners = get_corner_points(image,tag_id)
    dx1 = corners[0][0]-corners[0][1]
    dy1 = corners[1][0] - corners[1][1]
    dx2 = corners[0][1] - corners[0][2]
    dy2 = corners[1][1] - corners[1][2]
    dx3 = corners[0][2] - corners[0][3]
    dy3 = corners[1][2] - corners[1][3]
    dx4 = corners[0][3] - corners[0][0]
    dy4 = corners[1][3] - corners[1][0]
    average_length = (np.sqrt(dx1**2+dy1**2)+np.sqrt(dx2**2+dy2**2)+np.sqrt(dx3**2+dy3**2)+np.sqrt(dx4**2+dy4**2))/4
    return average_length

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