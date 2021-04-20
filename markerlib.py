import numpy as np
import math
import cv2


class Shelf:
    def __init__(self, markers):
        self.planes = [Plane(i, markers) for i in range(3)]
        self.boxes_not_on_shelf = 0

    def add_box_plane(self, box):
        for plane in self.planes:
            if plane.upper_limit < box.y[0] < plane.lower_limit:
                if plane.x[0] < box.x[0] and plane.x[1] > box.x[1]:
                    plane.add_box(box)
                    return
        self.boxes_not_on_shelf += 1

    def __str__(self):
        return "".join([str(plane) for plane in self.planes]) + f"\nboxes not on any shelf found: {self.boxes_not_on_shelf}"

    def disp_planes(self, tag,  boxes):
        # Plot plane areas:
        for plane in self.planes:
            cv2.line(tag.image,
                     (math.floor(plane.x[0]), math.floor(plane.upper_limit)),
                     (math.floor(plane.x[1]), math.floor(plane.upper_limit)),
                     (255, 0, 0), 1)
            cv2.line(tag.image,
                     (math.floor(plane.x[0]), math.floor(plane.lower_limit)),
                     (math.floor(plane.x[1]), math.floor(plane.lower_limit)),
                     (255, 0, 0), 1)

        # Plot boxes:
        for box in boxes:
            cv2.line(tag.image, (math.floor(box.x[0]), math.floor(box.y[0])),
                     (math.floor(box.x[1]), math.floor(box.y[0])), (0, 255, 0), 2)
            cv2.line(tag.image, (math.floor(box.x[1]), math.floor(box.y[0])),
                     (math.floor(box.x[1]), math.floor(box.y[1])), (0, 255, 0), 2)
            cv2.line(tag.image, (math.floor(box.x[1]), math.floor(box.y[1])),
                     (math.floor(box.x[0]), math.floor(box.y[1])), (0, 255, 0), 2)
            cv2.line(tag.image, (math.floor(box.x[0]), math.floor(box.y[1])),
                     (math.floor(box.x[0]), math.floor(box.y[0])), (0, 255, 0), 2)
        cv2.imshow('Shelf planes', tag.image)
        cv2.imwrite("graphics/test.jpg", tag.image)
        cv2.waitKey(0)


class Plane:
    def __init__(self, plane_id, markers):
        self.plane_id = plane_id
        self.boxes = []
        self.real_x = [0, 2220]
        if plane_id == 0:
            self.true_x_vals = [[222, 622], [645, 1245], [1318, 1718], [1733, 2028]]
            self.real_z = 100
            self.real_y = 0
            self.marker = [Marker(markers[0]), Marker(markers[1])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2 # assumes marker placed on edge of plane
            self.x = [self.marker[0].x[2], self.marker[1].x[3]]
            self.lower_limit = self.y + 5*(self.marker[0].y[2] - self.marker[0].y[1])/3
            self.upper_limit = self.y + 1*(self.marker[0].y[2] - self.marker[0].y[1]) / 6
            self.real_dist = 2220
        elif plane_id == 1:
            self.true_x_vals = [[97, 392], [423, 718], [744, 1144], [1231, 1526], [1544, 1839], [1864, 2159]]
            self.real_z = 200
            self.real_y = 50
            self.marker = [Marker(markers[2]), Marker(markers[3])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[0].x[2], self.marker[1].x[3]]
            self.lower_limit = self.y + 3*(self.marker[0].y[2] - self.marker[0].y[1]) / 4
            self.upper_limit = self.y - (self.marker[0].y[2] - self.marker[0].y[1]) / 2
            self.real_dist = 2225
        elif plane_id == 2:
            self.true_x_vals = [[1072, 1272]]
            self.real_z = 300
            self.real_y = 100
            self.marker = [Marker(markers[4]), Marker(markers[5])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[0].x[2], self.marker[1].x[3]]
            self.lower_limit = self.y + 3*(self.marker[0].y[2] - self.marker[0].y[1]) / 6
            self.upper_limit = self.y
            self.real_dist = 2230
        else:
            print(f"wrong plane id{plane_id}")

        self.marker_multiplier = self.optimize_marker(0.5)

    def __str__(self):
        plane_str = f"plane :{self.plane_id} real y = {self.real_y}, real z = {self.real_z}\n"
        box_str = "x value from left most tag: " + "".join([f"{self.get_box_x(box)}" for box in self.boxes])
        return plane_str + box_str + "\n"

    def add_box(self, box):
        self.boxes.append(box)

    def get_box_x(self, box):
        p0_dist = box.x[0] - self.marker[0].center_point[0], self.marker[1].center_point[0] - box.x[0]  # pixel distance for first x value to each marker
        p1_dist = box.x[1] - self.marker[0].center_point[0], self.marker[1].center_point[0] - box.x[1]  # pixel distance for second x value to each marker
        w0 = np.divide(np.array([p0_dist[1], p0_dist[0]]), self.marker[1].center_point[0] - self.marker[0].center_point[0]) # normalized distance to each marker for first x value,  used as weight
        w1 = np.divide(np.array([p1_dist[1], p1_dist[0]]), self.marker[1].center_point[0] - self.marker[0].center_point[0])

        factor_0 = self.get_average_marker_length(0) # pixel size of marker 0 (left marker when correctly oriented)
        factor_1 = self.get_average_marker_length(1) # pixel size of marker 1 (right marker when correctly oriented)
        x_values = [(self.marker_multiplier*self.marker[0].size*p0_dist[0]*2/(factor_0 + factor_1)),
                    (self.marker_multiplier*self.marker[0].size*p1_dist[0]*2/(factor_0 + factor_1))]
        return x_values

    def get_average_marker_length(self, ind):
        average_length_0 = np.sqrt(((self.marker[0].x[2] - self.marker[0].x[3])**2 + (self.marker[0].y[2] - self.marker[0].y[3])**2))
        average_length_1 = np.sqrt(((self.marker[1].x[2] - self.marker[1].x[3])**2 + (self.marker[1].y[2] - self.marker[1].y[3])**2))
        average_length = (average_length_1 + average_length_0)/2
        # dx = [self.marker[ind].x[3] - self.marker[ind].x[0]]
        # dy = [self.marker[ind].y[3] - self.marker[ind].y[0]]
        # for i in range(3):
        #     dx.append(self.marker[ind].x[i]-self.marker[ind].x[i+1])
        #     dy.append(self.marker[ind].y[i]-self.marker[ind].y[i+1])
        # average_length = sum([np.sqrt(dx[i]**2 + dy[i]**2)for i in range(4)])/4
        return average_length

    def get_plane_distance(self):
        factor_0 = self.marker[0].size*2/((self.marker[0].x[1] - self.marker[0].x[0]) + (self.marker[0].x[2] - self.marker[0].x[3]))
        factor_1 = self.marker[1].size*2/((self.marker[1].x[1] - self.marker[1].x[0]) + (self.marker[1].x[2] - self.marker[1].x[3]))
        p_dist = self.marker[1].center_point[0] - self.marker[0].center_point[0]
        distance = self.marker_multiplier*p_dist*(factor_0+factor_1)/2
        return distance

    def get_x_error(self):
        x_val = [self.get_box_x(box) for box in self.boxes]
        x_val = sorted(x_val)
        error = [np.subtract(x, true).tolist() for x, true in zip(x_val, self.true_x_vals)]
        error = [[abs(err[0]), err[1]] for err in error]
        return error

    def optimize_marker(self, marker_multiplier):
        calc_dist = 0
        while abs(calc_dist - self.real_dist) > 10:
            factor_0 = self.marker[0].size*2/((self.marker[0].x[1] - self.marker[0].x[0]) + (self.marker[0].x[2] - self.marker[0].x[3]))
            factor_1 = self.marker[1].size*2/((self.marker[1].x[1] - self.marker[1].x[0]) + (self.marker[1].x[2] - self.marker[1].x[3]))
            p_dist = self.marker[1].center_point[0] - self.marker[0].center_point[0]
            calc_dist = marker_multiplier*p_dist*(factor_0+factor_1)/2
            marker_multiplier += 1/200
            if marker_multiplier > 1.5:
                print("wrong marker multiplier probably")
                return marker_multiplier
        return marker_multiplier

class Marker:
    def __init__(self, marker):
        # marker must be: list(list(x),list(y), id)
        self.size = 99  # Marker size
        self.x = marker[0]
        self.y = marker[1]
        self.marker_id = marker[2]
        self.center_point = self.get_center_point()

    def get_center_point(self):
        x_c = sum(self.x)/4
        y_c = sum(self.y)/4
        return [x_c, y_c]


class Box:
    def __init__(self, line, h, w):
        coords = line.split()
        self.box_class = float(coords[0])
        box_x = float(coords[1]) * w
        box_y = float(coords[2]) * h
        box_w = float(coords[3]) * w
        box_h = float(coords[4]) * h
        self.conf = float(coords[5])
        self.x = [box_x-(box_w/2), box_x+(box_w/2)]
        self.y = [box_y+(box_h/2), box_y-(box_h/2)]


