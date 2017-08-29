import math
from collections import deque
from types import FunctionType

import cv2

import utils


class ColorTracker(object):
    def __init__(self, camera, max_nb_of_points=None, court_points=None, debug=True):
        self.__camera = camera
        self.__frame = None
        self.__tracker_points = None
        self.__debug = debug
        self.__max_nb_of_points = max_nb_of_points
        self.__selection_points = court_points
        self.__alerted = False
        self.__create_tracker_points_list()

    def __create_tracker_points_list(self):
        if self.__max_nb_of_points:
            self.__tracker_points = deque(maxlen=self.__max_nb_of_points)
        else:
            self.__tracker_points = deque()

    def __alert_when_crossed_line(self, object_center, alert_y, alert_callback_function, img_to_draw_on=None):
        if not isinstance(alert_callback_function, FunctionType):
            raise Exception("alert_function is not a valid Function with type: FunctionType!")

        x, y = object_center
        if not self.__alerted:
            if y >= alert_y:
                self.__alerted = True
                alert_callback_function(object_center)
        else:
            if y < alert_y:
                self.__alerted = False

        if self.__debug:
            if img_to_draw_on is not None:
                h, w, c = img_to_draw_on.shape
                cv2.line(img_to_draw_on, (0, alert_y), (w, alert_y), (255, 0, 0), 1)

    def __find_and_track_object_center_point(self, contours, min_contour_area, alert_y, alert_callback_function,
                                             min_point_distance=10, max_point_distance=math.inf,
                                             img_to_draw_on=None):
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= min_contour_area:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if alert_y is not None:
                    self.__alert_when_crossed_line(center, alert_y, alert_callback_function, img_to_draw_on)

                if self.__debug:
                    if img_to_draw_on is not None:
                        cv2.drawContours(img_to_draw_on, [c], -1, (0, 255, 0), cv2.FILLED)
                        cv2.circle(img_to_draw_on, (int(x), int(y)), int(radius), (0, 255, 255), 1)
                        cv2.circle(img_to_draw_on, center, 3, (0, 0, 255), -1)

                try:
                    dst = utils.calculate_distance(self.__tracker_points[-1], center)
                    if max_point_distance > dst > min_point_distance:
                        self.__tracker_points.append(center)
                except IndexError as e:
                    # It happens only when the queue is empty and we need a starting point
                    self.__tracker_points.append(center)

                return True, center
        return False, None

    def __draw_tracker_points(self, img_to_draw_on=None):
        if self.__debug:
            if img_to_draw_on is not None:
                for i in range(1, len(self.__tracker_points)):
                    if self.__tracker_points[i - 1] is None or self.__tracker_points[i] is None:
                        continue

                    rectangle_offset = 4
                    rectangle_pt1 = tuple(x - rectangle_offset for x in self.__tracker_points[i])
                    rectangle_pt2 = tuple(x + rectangle_offset for x in self.__tracker_points[i])
                    cv2.rectangle(img_to_draw_on, rectangle_pt1, rectangle_pt2, (255, 255, 255), 1)
                    cv2.line(img_to_draw_on, self.__tracker_points[i - 1], self.__tracker_points[i], (255, 255, 255), 1)

    def track(self, hsv_lower_value, hsv_upper_value, min_contour_area=10000, kernel=None, tracking_callback=None,
              alert_y=None,
              alert_callback_function=lambda x: x):
        while True:
            ret, self.__frame = self.__camera.read()

            if ret:
                self.__frame = cv2.flip(self.__frame, 1)
            else:
                continue

            if (self.__selection_points is not None) and (self.__selection_points != []):
                self.__frame = utils.crop_out_polygon_convex(self.__frame, self.__selection_points)

            img = self.__frame.copy()
            debug_frame = self.__frame.copy()

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower_value, hsv_upper_value)
            if kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            ret, obj_center = self.__find_and_track_object_center_point(cnts, min_contour_area, alert_y,
                                                                        img_to_draw_on=debug_frame,
                                                                        alert_callback_function=alert_callback_function)
            if tracking_callback is not None:
                tracking_callback(obj_center)

            self.__draw_tracker_points(debug_frame)

            cv2.imshow("color tracker", debug_frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key & 0XFF == ord("c"):
                if len(self.__tracker_points) > 0:
                    self.__create_tracker_points_list()
