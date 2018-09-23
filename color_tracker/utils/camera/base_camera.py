import threading

import cv2


class Camera(object):
    """
    Base Camera object
    """

    def __init__(self):
        self._cam = None
        self._frame = None
        self._frame_width = None
        self._frame_height = None
        self._ret = False

        self._auto_undistortion = False
        self._camera_matrix = None
        self._distortion_coefficients = None

        self._is_running = False

    def _init_camera(self):
        """
        This is the first for creating our camera
        We should override this!
        """

        pass

    def start_camera(self):
        """
        Start the running of the camera, without this we can't capture frames
        Camera runs on a separate thread so we can reach a higher FPS
        """

        self._init_camera()
        self._is_running = True
        threading.Thread(target=self._update_camera, args=()).start()

    def _read_from_camera(self):
        """
        This method is responsible for grabbing frames from the camera
        We should override this!
        """

        if self._cam is None:
            raise Exception("Camera is not started!")

    def _update_camera(self):
        """
        Grabs the frames from the camera
        """

        while True:
            if self._is_running:
                self._ret, self._frame = self._read_from_camera()
            else:
                break

    def get_frame_width_and_height(self):
        """
        Returns the width and height of the grabbed images
        :return (int int): width and height
        """

        return self._frame_width, self._frame_height

    def read(self):
        """
        With this you can grab the last frame from the camera
        :return (boolean, np.array): return value and frame
        """
        if self._is_running:
            return self._ret, self._frame
        else:
            import warnings
            warnings.warn("Camera is not started, you should start it with start_camera()")
            return False, None

    def release(self):
        """
        Stop the camera
        """

        self._is_running = False

    def is_running(self):
        return self._is_running

    def set_calibration_matrices(self, camera_matrix, distortion_coefficients):
        self._camera_matrix = camera_matrix
        self._distortion_coefficients = distortion_coefficients

    def activate_auto_undistortion(self):
        self._auto_undistortion = True

    def deactivate_auto_undistortion(self):
        self._auto_undistortion = False

    def _undistort_image(self, image):
        if self._camera_matrix is None or self._distortion_coefficients is None:
            import warnings
            warnings.warn("Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self._camera_matrix,
                                                               self._distortion_coefficients, (w, h),
                                                               1,
                                                               (w, h))
        undistorted = cv2.undistort(image, self._camera_matrix, self._distortion_coefficients, None,
                                    new_camera_matrix)
        return undistorted

    def __enter__(self):
        self.start_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
