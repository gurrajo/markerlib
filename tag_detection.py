import cv2
import numpy as np
import csv


class Tag:
    """
    Image object containing tag information
    """
    def __init__(self, fname, tag_type):
        self.fname = fname
        self.image = self.image_read(f"graphics/images_from_tests/{fname}")
        self.tag_type = tag_type
        self.dict = self.get_dict()
        self.markers, self.corners, self.ids = self.detect_tags()  # necessary for rotation

        self.image = self.rotate_image()
        self.markers, self.corners, self.ids = self.detect_tags()  # after rotation re-read

    def get_dict(self):
        super().__init__()
        if self.tag_type == 'aruco_4x4':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        elif self.tag_type == 'aruco_original':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

        elif self.tag_type == 'apriltag_36h11':
            used_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
        else:
            used_dict = ""
            print("incorrect tag type")
        return used_dict

    def image_read(self, fname):
        matrix = np.array([[1239.9865705358163, 0.0, 1163.236496781095], [0.0, 1250.0790335853487, 904.7312724102765],[0.0, 0.0, 1.0]])
        distortion = np.array([-0.2990077843090973, 0.11025572359961867, -0.0006408485381324832, 0.001213706562878266, -0.021134022438474565])
        newcameramtx = np.array([[1.01386385e+03, 0.00000000e+00, 1.16515228e+03],
                                [0.00000000e+00, 1.05031275e+03, 9.05845699e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # alpha = 0.25
        image = cv2.imread(fname)
        h, w = image.shape[:2]
        dst = cv2.undistort(image, matrix, distortion, None, newcameramtx)
        return dst

    def rotate_image(self):
        # Get orientation
        (h, w) = self.image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        centerpoint2 = [sum(self.markers[2][0])/4, sum(self.markers[2][1])/4]
        centerpoint3 = [sum(self.markers[3][0])/4, sum(self.markers[3][1])/4]
        dx = centerpoint3[0] - centerpoint2[0]
        dy = centerpoint3[1] - centerpoint2[1]
        # center = (centerpoint3 + centerpoint2) / 2
        angle_degrees = np.arctan(dy / dx) * 180 / np.pi
        rot_matrix = cv2.getRotationMatrix2D((cX, cY), angle_degrees, 1)
        rot_image = cv2.warpAffine(self.image, rot_matrix, (w, h))
        if self.markers[0][0] < self.markers[1][0]:
            rotation_matrix = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
            rot_image = cv2.warpAffine(rot_image, rotation_matrix, (w, h))
        return rot_image

    def detect_tags(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)
        markers = []
        for tag_id in range(len(ids)):
            for i in range(len(ids)):
                current_id = ids[i]
                if current_id == tag_id:
                    x = [corners[i, 0, 0, 0], corners[i, 0, 1, 0],
                         corners[i, 0, 2, 0], corners[i, 0, 3, 0]]

                    y = [corners[i, 0, 0, 1], corners[i, 0, 1, 1],
                         corners[i, 0, 2, 1], corners[i, 0, 3, 1]]
                    markers.append([x, y, tag_id])
        return markers, corners, ids

    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imwrite(f'graphics/calibration_output/{self.fname}', image)
        cv2.imshow('Tags', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def optimal_plane_orientation(self, markers):
        marker_center = []
        for marker in markers:
            marker_center.append([sum(marker[0]) / 4, sum(marker[1]) / 4])

        x_dif = [marker_center[1][0] - marker_center[5][0], marker_center[4][0] - marker_center[0][0]]
        b_h_ratio = (marker_center[4][0] - marker_center[5][0])/(marker_center[0][1]-marker_center[4][1])
        return [x_dif + [b_h_ratio]]

    def re_orient_image(self, optimal_values, markers):
        rows, cols, ch = self.image.shape
        x_dif0 = optimal_values[:2]
        b_h_ratio = optimal_values[2]
        marker_center = []
        for marker in markers:
            marker_center.append([sum(marker[0]) / 4, sum(marker[1]) / 4])
        b1 = (marker_center[4][0] - marker_center[5][0])
        h1 = (marker_center[0][1] - marker_center[4][1])
        alpha = (b1/b_h_ratio) - h1
        print(alpha)
        x_dif1 = [marker_center[1][0] - marker_center[5][0], marker_center[4][0] - marker_center[0][0]]
        dif_x_dif = [x_dif0[0]-x_dif1[0], x_dif0[1]-x_dif1[1]]
        pts1 = [[marker_center[0][0] + dif_x_dif[1], marker_center[0][1] + (alpha / 2)],
                [marker_center[1][0] + dif_x_dif[0], marker_center[1][1] + (alpha / 2)],
                [marker_center[4][0] - dif_x_dif[1], marker_center[4][1] - (alpha / 2)],
                [marker_center[5][0] - dif_x_dif[0], marker_center[5][1] - (alpha / 2)]]
        pts2 = marker_center[0:2] + marker_center[4:6]
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        print(pts1)
        print(pts2)
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(self.image, M, (cols, rows))
        cv2.imshow("test", dst)
        cv2.waitKey(0)


def get_camera_matrix():
    with open('graphics/calibration_output/calibration_matrix.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        mtx = []
        dist = []

        for line_counter, line in enumerate(csv_reader):

            mtx_line = []
            if line_counter == 0:
                for element in line:
                    mtx_line.append(np.float64(element))
                mtx.append(mtx_line)

            elif line_counter == 2:
                for element in line:
                    mtx_line.append(np.float64(element))
                mtx.append(mtx_line)

            elif line_counter == 4:
                for element in line:
                    mtx_line.append(np.float64(element))
                mtx.append(mtx_line)

            elif line_counter == 6:
                for element in line:
                    dist.append(np.float64(element))
        mtx = np.array(mtx)
        dist = np.array(dist)
        return mtx, dist