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
        self._tracking_callback = None
        self._last_detected_object_contour = None
        self._last_detected_object_center = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None

        self._create_tracker_points_list()

    def set_court_points(self, court_points):
        """
        Set a set of points that crops out a convex polygon from the image.
        So only on the cropped part will be detection
        :param court_points (list): list of points
        """

        self._selection_points = court_points

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

    def get_tracker_points(self):
        """
        :return (list): Returns the tracker points what were captured
        """
        return self._tracker_points

    def _add_new_tracker_point(self, min_point_distance, max_point_distance):
        try:
            dst = helpers.calculate_distance(self._tracker_points[-1], self._last_detected_object_center)
            if max_point_distance > dst > min_point_distance:
                self._tracker_points.append(self._last_detected_object_center)
        except IndexError:
            # It happens only when the queue is empty and we need a starting point
            self._tracker_points.append(self._last_detected_object_center)

    def _find_and_track_object_center_point(self, contours, min_contour_area,
                                            min_point_distance, max_point_distance):

        self._last_detected_object_contour = helpers.get_largest_contour(contours, min_contour_area)

        if self._last_detected_object_contour is not None:
            self._last_detected_object_center = helpers.get_contour_center(self._last_detected_object_contour)

            self._add_new_tracker_point(min_point_distance, max_point_distance)
        else:
            self._last_detected_object_center = None

    def _draw_debug_things(self, draw_tracker_points=True, draw_contour=True,
                           draw_object_center=True):
        if draw_contour:
            if self._last_detected_object_contour is not None:
                cv2.drawContours(self._debug_frame, [self._last_detected_object_contour], -1, (0, 255, 0), cv2.FILLED)
        if draw_object_center:
            if self._last_detected_object_center is not None:
                cv2.circle(self._debug_frame, self._last_detected_object_center, 3, (0, 0, 255), -1)
        if draw_tracker_points:
            self._draw_tracker_points(self._debug_frame)

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

    def stop_tracking(self):
        """
        Stop the color tracking
        """

        self._is_running = False

    def _read_from_camera(self):
        ret, self._frame = self._camera.read()

        if ret:
            self._frame = cv2.flip(self._frame, 1)
        else:
            import warnings
            warnings.warn("There is no camera feed!")

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
            self._read_from_camera()

            if (self._selection_points is not None) and (self._selection_points != []):
                self._frame = helpers.crop_out_polygon_convex(self._frame, self._selection_points)

            img = self._frame.copy()
            self._debug_frame = self._frame.copy()

            cnts = helpers.find_object_contours(image=img,
                                                hsv_lower_value=hsv_lower_value,
                                                hsv_upper_value=hsv_upper_value,
                                                kernel=kernel)

            self._find_and_track_object_center_point(contours=cnts,
                                                     min_contour_area=min_contour_area,
                                                     min_point_distance=min_track_point_distance,
                                                     max_point_distance=math.inf)

            if self._debug:
                self._draw_debug_things(draw_contour=False)

            if self._tracking_callback is not None:
                try:
                    self._tracking_callback()
                except TypeError:
                    import warnings
                    warnings.warn(
                        "Tracker callback function is not working because of wrong arguments! It takes zero arguments")

            if not self._is_running:
                break

    def get_debug_image(self):
        if self._debug:
            return self._debug_frame
        else:
            import warnings
            warnings.warn("Debugging is not enabled so there is no debug frame")

    def get_frame(self):
        return self._frame

    def get_last_object_center(self):
        return self._last_detected_object_center
