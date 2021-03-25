class Shelf:
    def __init__(self, markers):
        self.planes = [Plane(i, markers) for i in range(3)]
        self.boxes_not_on_shelf = 0

    def add_box_plane(self, box):
        for plane in self.planes:
            if plane.y*0.99 < box.y[0] < plane.y*1.01:
                if plane.x[0] < box.x[0] and plane.x[1] > box.x[1]:
                    plane.add_box(box)
                    return
        self.boxes_not_on_shelf += 1

    def __str__(self):
        return "".join([str(plane) for plane in self.planes]) + f"\nboxes not on any shelf found: {self.boxes_not_on_shelf}"


class Plane:
    def __init__(self, plane_id, markers):
        self.plane_id = plane_id
        self.boxes = []
        self.real_x = [0, 2000]
        if plane_id == 0:
            self.real_z = 100
            self.real_y = 0
            self.marker = [Marker(markers[0]), Marker(markers[1])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2 # assumes marker placed on edge of plane
            self.x = [self.marker[1].x[2], self.marker[0].x[3]]
        elif plane_id == 1:
            self.real_z = 200
            self.real_y = 50
            self.marker = [Marker(markers[2]), Marker(markers[3])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[1].x[2], self.marker[0].x[3]]
        elif plane_id == 2:
            self.real_z = 300
            self.real_y = 100
            self.marker = [Marker(markers[4]), Marker(markers[5])]
            self.y = (self.marker[0].y[3] + self.marker[1].y[3])/2
            self.x = [self.marker[1].x[2], self.marker[0].x[3]]
        else:
            print(f"wrong plane id{plane_id}")

    def __str__(self):
        plane_str = f"plane ID:{self.plane_id} x = {self.x} y = {self.y}, real x = {self.real_x}, real y = {self.real_y}, real z = {self.real_z}\n"
        box_str = "box x value from left tag: " + "".join([f" {self.get_box_x(box)} " for box in self.boxes])
        return plane_str + box_str + "\n"

    def add_box(self, box):
        self.boxes.append(box)

    def get_box_x(self, box):
        x_funk = (self.marker[0].size*2)/((self.marker[0].x[3]-self.marker[0].x[2]) + (self.marker[1].x[3] - self.marker[1].x[2]))  # describes x value coorelation in plane
        # x value from ref point left marker lower right corner
        return (box.x[0] - self.marker[0].x[2])*x_funk

    def get_plane_distance(self):
        factor_0 = self.marker[0].size*2/((self.marker[0].x[1] - self.marker[0].x[0]) + (self.marker[0].x[2] - self.marker[0].x[3]))
        factor_1 = self.marker[1].size*2/((self.marker[1].x[1] - self.marker[1].x[0]) + (self.marker[1].x[2] - self.marker[1].x[3]))
        p_dist = abs(self.marker[0].center_point[0] - self.marker[1].center_point[0])
        distance = p_dist*(factor_0+factor_1)/2
        return distance


class Marker:
    def __init__(self, marker):
        # marker must be: list(list(x),list(y), id)
        self.size = 100  # Marker size
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
        self.x = [box_x+box_w/2, box_x-box_w/2]
        self.y = [box_y+box_h/2, box_y-box_h/2]
        print(self.x)
        print(self.y)


