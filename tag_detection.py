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
        self.ids, self.corners = self.detect_tags()  # necessary for rotation
        #self.image = self.rotate_image()
        self.ids, self.corners = self.detect_tags()  # after rotation re-read

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
        #matrix, distortion = get_camera_matrix()
        matrix = np.array([[1239.9865705358163, 0.0, 1163.236496781095],[0.0, 1250.0790335853487, 904.7312724102765],[0.0, 0.0, 1.0]])
        distortion = np.array([-0.2990077843090973,0.11025572359961867,-0.0006408485381324832,0.001213706562878266,-0.021134022438474565])

        image = cv2.imread(fname)
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 0.4, (w, h))
        dst = cv2.undistort(image, matrix, distortion, None, None)
        # The following 2 lines are used if you crop the image(Current configuration uses alpha=0.55 and does NOT crop the image)
        #x, y, w, h = roi
        #cropped_dst = dst[y:y+h,x:x+w]
        return dst

    def rotate_image(self):
        angle = 180
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_image = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rot_image

    def detect_tags(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dict, parameters=aruco_parameters)
        corners = np.array(corners)
        return ids, corners

    def draw_tags(self):
        scale = 1
        image = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        frame = cv2.aruco.drawDetectedMarkers(image, self.corners*scale, self.ids)
        cv2.imwrite(f'graphics/calibration_output/{self.fname}', image)
        cv2.imshow('Tags', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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