import math
from collections import deque
from types import FunctionType

import cv2

from color_tracker.utils import helpers


class ColorTracker(object):
    def __init__(self, camera, max_nb_of_points=None, debug=True):
        """
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param max_nb_of_points: Maxmimum number of points for storing. If it is set
        to None than it means there is no limit
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """

        super().__init__()
        self._camera = camera
        self._tracker_points = None
        self._debug = debug
        self._max_nb_of_points = max_nb_of_points
        self._selection_points = None
        self._alerted = False
        self._alert_y = None
        self._alert_callback_function = None
        self._tracking_callback = None
        self._last_detected_contour = None
        self._last_detected_object_center = None
        self._is_running = False

        self._create_tracker_points_list()

    def set_court_points(self, court_points):
        """
        Set a set of points that crops out a convex polygon from the image.
        So only on the cropped part will be detection
        :param court_points (list): list of points
        """

        self._selection_points = court_points

    def set_alert_callback(self, alert_y, alert_callback_function):
        if not isinstance(alert_callback_function, FunctionType):
            raise Exception("alert_callback_function is not a valid Function with type: FunctionType!")
        self._alert_y = alert_y
        self._alert_callback_function = alert_callback_function

    def set_tracking_callback(self, tracking_callback):
        if not isinstance(tracking_callback, FunctionType):
            raise Exception("tracking_callback is not a valid Function with type: FunctionType!")
        self._tracking_callback = tracking_callback

    def _create_tracker_points_list(self):
        """
        Initialize the tracker point list
        """

        if self._max_nb_of_points:
            self._tracker_points = deque(maxlen=self._max_nb_of_points)
        else:
            self._tracker_points = deque()

    def _alert_when_crossed_line(self, object_center):
        x, y = object_center
        try:
            prev_point_x, prev_point_y = self._tracker_points[-1]
        except IndexError as e:
            return

        if prev_point_y < self._alert_y:
            if not self._alerted:
                if y >= self._alert_y:
                    self._alerted = True
                    try:
                        self._alert_callback_function()
                    except TypeError as e:
                        print("""
                                [*] alert callback function has 0 args
                                Example:
                                    def callback():
                                        pass
                                """)
                        raise e
            else:
                if y < self._alert_y:
                    self._alerted = False

    def get_tracker_points(self):
        """
        :return (list): Returns the tracker points what were captured
        """
        return self._tracker_points

    def _find_and_track_object_center_point(self, contours, min_contour_area,
                                            min_point_distance, max_point_distance=math.inf):
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= min_contour_area:
                # ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                self._last_detected_contour = c
                self._last_detected_object_center = center

                if self._alert_y is not None and self._alert_callback_function is not None:
                    self._alert_when_crossed_line(object_center=center)

                try:
                    dst = helpers.calculate_distance(self._tracker_points[-1], center)
                    if max_point_distance > dst > min_point_distance:
                        self._tracker_points.append(center)
                except IndexError as e:
                    # It happens only when the queue is empty and we need a starting point
                    self._tracker_points.append(center)

                return True
            else:
                self._last_detected_contour = None
                self._last_detected_object_center = None
        return False

    def _draw_debug_things(self, debug_image, draw_tracker_points=True, draw_alert_line=True, draw_contour=True,
                           draw_object_center=True):
        if draw_contour:
            if self._last_detected_contour is not None:
                cv2.drawContours(debug_image, [self._last_detected_contour], -1, (0, 255, 0), cv2.FILLED)
        if draw_object_center:
            if self._last_detected_object_center is not None:
                cv2.circle(debug_image, self._last_detected_object_center, 3, (0, 0, 255), -1)
        if draw_alert_line:
            if self._alert_y is not None:
                h, w, c = debug_image.shape
                cv2.line(debug_image, (0, self._alert_y), (w, self._alert_y), (255, 0, 0), 1)
        if draw_tracker_points:
            self._draw_tracker_points(debug_image)

    def clear_track_points(self):
        """
        Delete all tracker points
        """

        if len(self._tracker_points) > 0:
            self._create_tracker_points_list()

    def _draw_tracker_points(self, debug_image):
        if debug_image is not None:
            for i in range(1, len(self._tracker_points)):
                if self._tracker_points[i - 1] is None or self._tracker_points[i] is None:
                    continue
                rectangle_offset = 4
                rectangle_pt1 = tuple(x - rectangle_offset for x in self._tracker_points[i])
                rectangle_pt2 = tuple(x + rectangle_offset for x in self._tracker_points[i])
                cv2.rectangle(debug_image, rectangle_pt1, rectangle_pt2, (255, 255, 255), 1)
                cv2.line(debug_image, self._tracker_points[i - 1], self._tracker_points[i], (255, 255, 255), 1)

    def _find_object_contours(self, image, hsv_lower_value, hsv_upper_value, kernel):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_value, hsv_upper_value)
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    def stop_tracking(self):
        """
        Stop the color tracking
        """

        self._is_running = False

    def track(self, hsv_lower_value, hsv_upper_value, min_contour_area, kernel=None, min_track_point_distance=20):
        """
        With this we can start the tracking with the given parameters
        :param hsv_lower_value: lowest acceptable hsv values
        :param hsv_upper_value: highest acceptable hsv values
        :param min_contour_area: minimum contour area for the detection. Below that the detection does not count
        :param kernel: structuring element to perform morphological operations on the mask image
        :param min_track_point_distance: minimum distance between the tracked and recognized points
        """

        self._is_running = True

        while True:
            ret, self.frame = self._camera.read()

            if ret:
                self.frame = cv2.flip(self.frame, 1)
            else:
                continue

            if (self._selection_points is not None) and (self._selection_points != []):
                self.frame = helpers.crop_out_polygon_convex(self.frame, self._selection_points)

            img = self.frame.copy()
            debug_frame = self.frame.copy()

            cnts = self._find_object_contours(img,
                                              hsv_lower_value=hsv_lower_value,
                                              hsv_upper_value=hsv_upper_value,
                                              kernel=kernel)

            self._find_and_track_object_center_point(contours=cnts,
                                                     min_contour_area=min_contour_area,
                                                     min_point_distance=min_track_point_distance)

            if self._debug:
                self._draw_debug_things(debug_frame, draw_contour=False)

            if self._tracking_callback is not None:
                try:
                    self._tracking_callback(self.frame, debug_frame, self._last_detected_object_center)
                except TypeError as e:
                    print("""
                        [*] tracker callback function has 3 args: (original_frame, debug_frame, object_center)
                        Example:
                            def callback(frame, debug_frame, object_center):
                                print(object_center)
                        """)
                    raise e

            if not self._is_running:
                break
