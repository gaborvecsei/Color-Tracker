import threading

import cv2


class Camera(object):
    def __init__(self):
        self._cam = None
        self._frame = None
        self._ret = False

        self.auto_undistortion = False
        self.__camera_matrix = None
        self.__distortion_coefficients = None

        self.__is_running = False

    def _init_camera(self):
        pass

    def start_camera(self):
        self._init_camera()
        self.__is_running = True
        threading.Thread(target=self.__update_camera, args=()).start()

    def _read_from_camera(self):
        if self._cam is None:
            raise Exception("Camera is not started!")

    def __update_camera(self):
        while True:
            if self.__is_running:
                self._ret, self._frame = self._read_from_camera()
            else:
                break

    def read(self):
        return self._ret, self._frame

    def release_camera(self):
        self.__is_running = False

    def set_calibration_matrices(self, camera_matrix, distortion_coefficients):
        self.__camera_matrix = camera_matrix
        self.__distortion_coefficients = distortion_coefficients

    def activate_auto_undistortion(self):
        self.auto_undistortion = True

    def deactivate_auto_undistortion(self):
        self.auto_undistortion = False

    def _undistort_image(self, image):
        if self.__camera_matrix is None or self.__distortion_coefficients is None:
            import warnings
            warnings.warn("Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.__camera_matrix,
                                                               self.__distortion_coefficients, (w, h),
                                                               1,
                                                               (w, h))
        undistorted = cv2.undistort(image, self.__camera_matrix, self.__distortion_coefficients, None,
                                    new_camera_matrix)
        return undistorted