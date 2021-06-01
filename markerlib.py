import numpy as np
import math
import cv2
import pymap3d as pm
import json
import redis

"""classes in this library corresponds to their real world equivalents. A shelf contains planes and a plane has boxes"""


class Shelf:
    """The class that contains planes. Only works with 3 planes"""
    def __init__(self, markers, origin):
        self.origin = origin
        self.planes = [Plane(i, markers, origin) for i in range(3)]
        self.boxes_not_on_shelf = 0

    def add_box_plane(self, box):
        """Determine plane association for a box. Checks if box borders is within x boundaries of a plane,
        check if y-values of lower rim is within limits afterwards."""
        for plane in self.planes:
            if plane.upper_limit < box.y[0] < plane.lower_limit:
                if plane.x[0] < box.x[0] and plane.x[1] > box.x[1]:
                    plane.add_box(box)
                    return
        self.boxes_not_on_shelf += 1

    def __str__(self):
        """Converts shelf information to a string"""
        return "".join([str(plane) for plane in self.planes]) + f"\nboxes not on any shelf found: {self.boxes_not_on_shelf}"

    def disp_planes(self, tag):
        """Displays a picture with lines for boxes and plane y-boundaries"""
        # Plot plane areas:
        for plane in self.planes:
            for box in plane.boxes:
                cv2.line(tag.image, (math.floor(box.x[0]), math.floor(box.y[0])),
                         (math.floor(box.x[1]), math.floor(box.y[0])), (0, 255, 0), 2)
                cv2.line(tag.image, (math.floor(box.x[1]), math.floor(box.y[0])),
                         (math.floor(box.x[1]), math.floor(box.y[1])), (0, 255, 0), 2)
                cv2.line(tag.image, (math.floor(box.x[1]), math.floor(box.y[1])),
                         (math.floor(box.x[0]), math.floor(box.y[1])), (0, 255, 0), 2)
                cv2.line(tag.image, (math.floor(box.x[0]), math.floor(box.y[1])),
                         (math.floor(box.x[0]), math.floor(box.y[0])), (0, 255, 0), 2)

            cv2.line(tag.image,
                     (math.floor(plane.x[0]), math.floor(plane.upper_limit)),
                     (math.floor(plane.x[1]), math.floor(plane.upper_limit)),
                     (255, 0, 0), 1)
            cv2.line(tag.image,
                     (math.floor(plane.x[0]), math.floor(plane.lower_limit)),
                     (math.floor(plane.x[1]), math.floor(plane.lower_limit)),
                     (255, 0, 0), 1)
        cv2.imshow('Shelf planes', tag.image)
        cv2.imwrite("graphics/temp.jpg", tag.image)  # overwritten every time
        cv2.waitKey(0)
        # Plot boxes:


    def redis_send(self):
        coordinates_shelf = []
        for i, plane in enumerate(self.planes):
            temp = []
            for box in plane.boxes:
                temp.append(plane.get_box_coordinates(box))
            coordinates_shelf.append(temp)
        # Coordinates_shelf1,2,3 are lists of arrays with coordinates for bins on respective shelf
        # Each array have the components: [ Latitud, Longitud, Altitud, Width of bin ]
        numb_of_box = len(coordinates_shelf[0]) + len(coordinates_shelf[1]) + len(coordinates_shelf[2])
        object_to_json = {
            "TotalNumOfBoxes": numb_of_box,
            "Shelf0": {
                "NumOfBoxes": len(coordinates_shelf[0]),
                "Coordinates": coordinates_shelf[0]
            },
            "Shelf1": {
                "NumOfBoxes": len(coordinates_shelf[1]),
                "Coordinates": coordinates_shelf[1]
            },
            "Shelf2": {
                "NumOfBoxes": len(coordinates_shelf[2]),
                "Coordinates": coordinates_shelf[2]
            },
        }
        print(object_to_json)
        ## Converting object to json
        # json_to_server = json.dumps(object_to_json)
        ## Redis server, change host/port/pd
        # r = redis.Redis(host='localhost', port=6379, db=0)
        ## Setting BinDetectionModel(on Redis server) to json_to_server(coordinates+number of boxes)
        # r.set("BinDetectionModel", json_to_server)

        ## To get the coordinates(json format) from Redis
        # json_from_server = json.loads(r.get("BoxModel"))
        ## Access total number of boxes
        # print(json_from_server["TotalNumOfBoxes"])
        ## Access number of boxes on shelf
        # print(json_from_server["Shelf1"]["NumOfBoxes"])
        ## Access coordinates of boxes on shelf
        # print(json_from_server["Shelf1"]["Coordinates"])


class Plane:
    """The planes of the shelf. Each plane has its own information. Hardcode when planes are measured"""
    def __init__(self, plane_id, markers, origin):
        self.plane_id = plane_id
        self.boxes = []
        self.origin = origin
        if plane_id == 0:
            self.true_x_vals_left = [200, 630, 1730]  # correct values for boxes, used for evaluation
            self.true_x_vals_right = [600, 1230, 2130]  # correct values for boxes, used for evaluation
            self.real_z = 0  # plane edge z-value
            self.real_y = 0  # plane edge y-value
            self.marker_ids = [0, 1]
            self.markers = self.get_markers(markers)
            self.marker = [Marker(markers[0]), Marker(markers[1])]  # markers associated with plane, uses the sorting to correctly assign. (should use marker id instead)
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2  # image location of plane on y axis
            self.x = [self.marker[0].center_point[0], self.marker[1].center_point[0]]  # image location of plane on x axis
            self.lower_limit = self.y + (self.marker[0].y[2] - self.marker[0].y[1]) / 2  # box association limit (y-axis)
            self.upper_limit = self.y - (self.marker[0].y[2] - self.marker[0].y[1]) / 2  # box association limit (y-axis)
            self.real_dist = 2202  # distance between centerpoints of both markers in [mm] used for distance calcualtions
        elif plane_id == 1:
            self.true_x_vals_left = [70,  405,  1205,  1520,  1835]
            self.true_x_vals_right = [365, 700, 1500, 1815, 2130]
            self.real_z = 200
            self.real_y = 50
            self.marker_ids = [2, 3]
            self.marker = [Marker(markers[2]), Marker(markers[3])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[0].center_point[0], self.marker[1].center_point[0]]
            self.lower_limit = self.y + (self.marker[0].y[2] - self.marker[0].y[1]) / 2
            self.upper_limit = self.y - (self.marker[0].y[2] - self.marker[0].y[1]) / 2
            self.real_dist = 2195
        elif plane_id == 2:
            self.true_x_vals_left = [1040, 1920]
            self.true_x_vals_right = [1240, 2120]
            self.real_z = 300
            self.real_y = 100
            self.marker_ids = [4, 5]
            self.marker = [Marker(markers[4]), Marker(markers[5])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[0].center_point[0], self.marker[1].center_point[0]]
            self.lower_limit = self.y + (self.marker[0].y[2] - self.marker[0].y[1]) / 2
            self.upper_limit = self.y - (self.marker[0].y[2] - self.marker[0].y[1]) / 2
            self.real_dist = 2185
        else:
            print(f"wrong plane id{plane_id}")

        self.marker_multiplier = self.optimize_marker()  # used for distance calculations, check function for more detail

    def __str__(self):
        """Used in shelf __str__ for shelf printout"""
        plane_str = f"plane :{self.plane_id} real y = {self.real_y}, real z = {self.real_z}\n"
        box_str = "coords: " + "".join([f"{self.get_box_coordinates(box)}" for box in self.boxes])
        return plane_str + box_str + "\n"

    def add_box(self, box):
        """called when a box is determined to be on a plane"""
        self.boxes.append(box)

    def get_box_x(self, box):
        """determine x values for a box """
        p0_dist = box.x[0] - self.marker[0].center_point[0], self.marker[1].center_point[0] - box.x[0]  # pixel distance for first x value to each marker
        p1_dist = box.x[1] - self.marker[0].center_point[0], self.marker[1].center_point[0] - box.x[1]  # pixel distance for second x value to each marker
        w0 = np.divide(np.array([p0_dist[1], p0_dist[0]]), self.marker[1].center_point[0] - self.marker[0].center_point[0]) # normalized distance to each marker for first x value,  used as weight
        w1 = np.divide(np.array([p1_dist[1], p1_dist[0]]), self.marker[1].center_point[0] - self.marker[0].center_point[0])

        factor_0 = self.get_average_marker_length()  # pixel size of marker 0 (left marker when correctly oriented)
        factor_1 = self.get_average_marker_length()  # pixel size of marker 1 (right marker when correctly oriented)
        x_values = [(self.marker_multiplier*self.marker[0].size*p0_dist[0]/(factor_0*w0[0] + factor_1*w0[1])),
                    (self.marker_multiplier*self.marker[0].size*p1_dist[0]/(factor_0*w1[0] + factor_1*w1[1]))]
        return x_values

    def get_box_coordinates(self, box):
        local_x = (self.get_box_x(box)[0] / 1e3 + self.get_box_x(box)[1] / 1e3) / 2
        local_y = self.real_y / 1e3
        local_z = self.real_z / 1e3
        width = np.abs(self.get_box_x(box)[1] / 1e3 - self.get_box_x(box)[0] / 1e3)

        east = np.cos(np.radians(self.origin[3])) * local_x + np.sin(np.radians(self.origin[3])) * local_y
        north = -np.sin(np.radians(self.origin[3])) * local_x + np.cos(np.radians(self.origin[3])) * local_y

        # Global
        lat, long, alt = pm.enu2geodetic(east, north, local_z, self.origin[0], self.origin[1], self.origin[2])
        output = [lat, long, alt, width]
        return output

    def get_average_marker_length(self):
        """Returns the average marker length in x-axis direction"""
        average_length_0 = np.sqrt(((self.marker[0].x[2] - self.marker[0].x[3])**2 + (self.marker[0].y[2] - self.marker[0].y[3])**2))
        average_length_1 = np.sqrt(((self.marker[1].x[2] - self.marker[1].x[3])**2 + (self.marker[1].y[2] - self.marker[1].y[3])**2))
        average_length = (average_length_1 + average_length_0)/2
        return average_length

    def get_plane_distance(self):
        """Returns calculated distance between marker centers"""
        factor_0 = self.marker[0].size*2/((self.marker[0].x[1] - self.marker[0].x[0]) + (self.marker[0].x[2] - self.marker[0].x[3]))
        factor_1 = self.marker[1].size*2/((self.marker[1].x[1] - self.marker[1].x[0]) + (self.marker[1].x[2] - self.marker[1].x[3]))
        p_dist = self.marker[1].center_point[0] - self.marker[0].center_point[0]
        distance = self.marker_multiplier*p_dist*(factor_0+factor_1)/2
        return distance

    def get_x_error(self):
        """Calculates error of the x values for all boxen on a plane"""
        x_val = [self.get_box_x(box) for box in self.boxes]
        x_val = sorted(x_val)
        error_mid = []
        error_wid = []
        for x in x_val:
            array1 = np.asarray(self.true_x_vals_left)
            idx_left = (np.abs(array1 - x[0])).argmin()
            array2 = np.asarray(self.true_x_vals_right)
            idx_right = (np.abs(array2 - x[1])).argmin()

            middle_calc = (x[0] + x[1]) / 2
            middle_real = (self.true_x_vals_left[idx_left] + self.true_x_vals_right[idx_right]) / 2

            width_calc = x[1] - x[0]
            width_real = self.true_x_vals_right[idx_right] - self.true_x_vals_left[idx_left]
            error_mid.append(middle_calc - middle_real)
            error_wid.append(width_calc - width_real)
        return [error_mid, error_wid]

    def get_markers(self, markers):
        for marker in markers:
            if marker[2] == self.marker_ids[0]:
                plane_markers = marker
                break

        for marker in markers:
            if marker[2] == self.marker_ids[1]:
                plane_markers.append(marker)
                break
        if len(plane_markers) < 2:
            print('marker not found')

        return plane_markers

    def optimize_marker(self):
        """Corrects the x-axis calcualtions using the already known distance between markers"""
        factor_0 = self.marker[0].size*2/((self.marker[0].x[1] - self.marker[0].x[0]) + (self.marker[0].x[2] - self.marker[0].x[3]))
        factor_1 = self.marker[1].size*2/((self.marker[1].x[1] - self.marker[1].x[0]) + (self.marker[1].x[2] - self.marker[1].x[3]))
        p_dist = self.marker[1].center_point[0] - self.marker[0].center_point[0]
        calc_dist = p_dist*(factor_0+factor_1)/2
        marker_multiplier = self.real_dist/calc_dist
        return marker_multiplier


class Marker:
    """Class for fiducial markers"""
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
    """Class for a blue box"""
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


