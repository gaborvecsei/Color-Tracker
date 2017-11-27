import cv2
import numpy as np
from color_tracker.utils.camera import Camera
from color_tracker.utils import helpers


class HSVColorRangeDetector:
    """
    Just a helper to determine what kind of lower and upper HSV values you need for the tracking
    """

    def __init__(self, camera):
        assert isinstance(camera, Camera), "camera parameter is not a Camera object!"
        assert camera.is_running(), "camera is not running"

        self._camera = camera
        self._trackbars = []
        self._main_window_name = "HSV color range detector"
        cv2.namedWindow(self._main_window_name)
        self._init_trackbars()

    def _init_trackbars(self):
        trackbars_window_name = "hsv settings"
        cv2.namedWindow(trackbars_window_name)

        h_min_trackbar = _Trackbar("H min", trackbars_window_name, 0, 255)
        s_min_trackbar = _Trackbar("S min", trackbars_window_name, 0, 255)
        v_min_trackbar = _Trackbar("V min", trackbars_window_name, 0, 255)

        h_max_trackbar = _Trackbar("H max", trackbars_window_name, 255, 255)
        s_max_trackbar = _Trackbar("S max", trackbars_window_name, 255, 255)

        kernel_x = _Trackbar("kernel x", trackbars_window_name, 0, 255)
        kernel_y = _Trackbar("kernel y", trackbars_window_name, 0, 255)
        v_max_trackbar = _Trackbar("V max", trackbars_window_name, 255, 255)

        self._trackbars = [h_min_trackbar, s_min_trackbar, v_min_trackbar, h_max_trackbar, s_max_trackbar,
                           v_max_trackbar, kernel_x, kernel_y]

    def _get_trackbar_values(self):
        values = []
        for t in self._trackbars:
            value = t.get_value()
            values.append(value)
        return values

    def detect(self):
        while True:
            ret, self.frame = self._camera.read()

            if ret:
                self.frame = cv2.flip(self.frame, 1)
            else:
                continue

            img = self.frame.copy()

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            values = self._get_trackbar_values()
            h_min, s_min, v_min = values[:3]
            h_max, s_max, v_max = values[3:6]
            kernel_x, kernel_y = values[6:]

            if kernel_y < 1:
                kernel_y = 1
            if kernel_x < 1:
                kernel_x = 1

            thresh = cv2.inRange(hsv_img, (h_min, s_min, v_min), (h_max, s_max, v_max))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            preview = cv2.bitwise_and(img, img, mask=thresh)

            display_width = 360
            display_height = 240

            img_display = helpers.resize_img(img, display_width, display_height)
            thresh_display = cv2.cvtColor(helpers.resize_img(thresh, display_width, display_height),
                                          cv2.COLOR_GRAY2BGR)
            display_img_1 = np.concatenate((img_display, thresh_display), axis=1)

            preview_display = helpers.resize_img(preview, display_width, display_height)
            hsv_img_display = helpers.resize_img(hsv_img, display_width, display_height)
            display_img_2 = np.concatenate((preview_display, hsv_img_display), axis=1)

            display_img = np.concatenate((display_img_1, display_img_2), axis=0)

            cv2.imshow(self._main_window_name, display_img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        self._camera.release_camera()
        cv2.destroyAllWindows()

        upper_color = np.array([h_max, s_max, v_max])
        lower_color = np.array([h_min, s_min, v_min])

        return lower_color, upper_color, kernel


class _Trackbar(object):
    def __init__(self, name, parent_window_name, init_value=0, max_value=255):
        self.parent_window_name = parent_window_name
        self.name = name
        self.init_value = init_value
        self.max_value = max_value

        cv2.createTrackbar(self.name, self.parent_window_name, self.init_value, self.max_value, lambda x: x)

    def get_value(self):
        value = cv2.getTrackbarPos(self.name, self.parent_window_name)
        return value
