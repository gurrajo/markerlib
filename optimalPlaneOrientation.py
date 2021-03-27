import tag_detection
import distance_measurement

filename = '4_0.0_mid.jpg'
tag_type = 'aruco_4x4'

new_image = tag_detection.Tag(filename, tag_type)
markers = [distance_measurement.get_corner_points(new_image, i) for i in range(len(new_image.ids))]
marker_center = []
for marker in markers:
    marker_center.append([sum((marker[0]) / 4), sum(marker[1]) / 4])

x_dif = []
x_dif.append(marker_center[1][0] - marker_center[5][0])
x_dif.append(marker_center[4][0] - marker_center[0][0])
h_b_ratio = (marker_center[4][0] - marker_center[5][0])/(marker_center[0][1]-marker_center[4][1])
return [x]