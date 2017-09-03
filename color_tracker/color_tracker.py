import math
from collections import deque
from types import FunctionType

import cv2

from utils import utils


class ColorTracker(object):
    def __init__(self, camera, max_nb_of_points=None, court_points=None, debug=True):
        super().__init__()
        self.__camera = camera
        self.__tracker_points = None
        self.__debug = debug
        self.__max_nb_of_points = max_nb_of_points
        self.__selection_points = court_points
        self.__alerted = False
        self.__alert_y = None
        self.__alert_callback_function = None
        self.__tracking_callback = None
        self.__last_detected_contour = None
        self.__last_detected_object_center = None
        self.__is_running = False

        self.__create_tracker_points_list()

    def set_alert_callback(self, alert_y, alert_callback_function):
        if not isinstance(alert_callback_function, FunctionType):
            raise Exception("alert_callback_function is not a valid Function with type: FunctionType!")
        self.__alert_y = alert_y
        self.__alert_callback_function = alert_callback_function

    def set_tracking_callback(self, tracking_callback):
        if not isinstance(tracking_callback, FunctionType):
            raise Exception("tracking_callback is not a valid Function with type: FunctionType!")
        self.__tracking_callback = tracking_callback

    def __create_tracker_points_list(self):
        if self.__max_nb_of_points:
            self.__tracker_points = deque(maxlen=self.__max_nb_of_points)
        else:
            self.__tracker_points = deque()

    def __alert_when_crossed_line(self, object_center):
        x, y = object_center
        try:
            prev_point_x, prev_point_y = self.__tracker_points[-1]
        except IndexError as e:
            return

        if prev_point_y < self.__alert_y:
            if not self.__alerted:
                if y >= self.__alert_y:
                    self.__alerted = True
                    try:
                        self.__alert_callback_function()
                    except TypeError as e:
                        print("""
                                [*] alert callback function has 0 args
                                Example:
                                    def callback():
                                        pass
                                """)
                        raise e
            else:
                if y < self.__alert_y:
                    self.__alerted = False

    def __alert_when_too_close(self, contour, alert_area, alert_callback_function):
        object_contour_area = cv2.contourArea(contour)
        if object_contour_area >= alert_area:
            if not self.__alerted:
                self.__alerted = True
                if alert_callback_function is not None:
                    alert_callback_function()
        else:
            self.__alerted = False

    def __find_and_track_object_center_point(self, contours, min_contour_area,
                                             min_point_distance, max_point_distance=math.inf):
        """
        If there was a detection it returns with True
        """

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= min_contour_area:
                # ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                self.__last_detected_contour = c
                self.__last_detected_object_center = center

                if self.__alert_y is not None and self.__alert_callback_function is not None:
                    self.__alert_when_crossed_line(object_center=center)

                try:
                    dst = utils.calculate_distance(self.__tracker_points[-1], center)
                    if max_point_distance > dst > min_point_distance:
                        self.__tracker_points.append(center)
                except IndexError as e:
                    # It happens only when the queue is empty and we need a starting point
                    self.__tracker_points.append(center)

                return True
            else:
                self.__last_detected_contour = None
                self.__last_detected_object_center = None
        return False

    def __draw_debug_things(self, debug_image):
        self.__draw_tracker_points(debug_image)

        if self.__alert_y is not None:
            h, w, c = debug_image.shape
            cv2.line(debug_image, (0, self.__alert_y), (w, self.__alert_y), (255, 0, 0), 1)

        if self.__last_detected_contour is not None:
            cv2.drawContours(debug_image, [self.__last_detected_contour], -1, (0, 255, 0), cv2.FILLED)
        if self.__last_detected_object_center is not None:
            cv2.circle(debug_image, self.__last_detected_object_center, 3, (0, 0, 255), -1)

    def clear_track_points(self):
        if len(self.__tracker_points) > 0:
            self.__create_tracker_points_list()

    def __draw_tracker_points(self, debug_image):
        if debug_image is not None:
            for i in range(1, len(self.__tracker_points)):
                if self.__tracker_points[i - 1] is None or self.__tracker_points[i] is None:
                    continue
                rectangle_offset = 4
                rectangle_pt1 = tuple(x - rectangle_offset for x in self.__tracker_points[i])
                rectangle_pt2 = tuple(x + rectangle_offset for x in self.__tracker_points[i])
                cv2.rectangle(debug_image, rectangle_pt1, rectangle_pt2, (255, 255, 255), 1)
                cv2.line(debug_image, self.__tracker_points[i - 1], self.__tracker_points[i], (255, 255, 255), 1)

    def __find_object_contours(self, image, hsv_lower_value, hsv_upper_value, kernel):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_value, hsv_upper_value)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    def stop_tracking(self):
        self.__is_running = False

    def track(self, hsv_lower_value, hsv_upper_value, kernel, min_contour_area, min_track_point_distance=20):
        self.__is_running = True

        while True:
            ret, self.frame = self.__camera.read()

            if ret:
                self.frame = cv2.flip(self.frame, 1)
            else:
                continue

            if (self.__selection_points is not None) and (self.__selection_points != []):
                self.frame = utils.crop_out_polygon_convex(self.frame, self.__selection_points)

            img = self.frame.copy()
            debug_frame = self.frame.copy()

            cnts = self.__find_object_contours(img,
                                               hsv_lower_value=hsv_lower_value,
                                               hsv_upper_value=hsv_upper_value,
                                               kernel=kernel)

            self.__find_and_track_object_center_point(contours=cnts,
                                                      min_contour_area=min_contour_area,
                                                      min_point_distance=min_track_point_distance)

            if self.__debug:
                self.__draw_debug_things(debug_frame)

            if self.__tracking_callback is not None:
                try:
                    self.__tracking_callback(self.frame, debug_frame, self.__last_detected_object_center)
                except TypeError as e:
                    print("""
                        [*] tracker callback function has 3 args: (original_frame, debug_frame, object_center)
                        Example:
                            def callback(frame, debug_frame, object_center):
                                print(object_center)
                        """)
                    raise e

            if not self.__is_running:
                break
