import warnings
from collections import deque
from typing import Union, List, Callable

import cv2
import numpy as np
from color_tracker.utils.tracker_object import TrackedObject

from color_tracker.utils import helpers, visualize
from color_tracker.utils.camera import Camera


class ColorTracker(object):
    def __init__(self, camera: Union[Camera, cv2.VideoCapture], max_nb_of_points: int = None, debug: bool = True):
        """
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param max_nb_of_points: Maxmimum number of points for storing. If it is set
        to None than it means there is no limit
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """

        super().__init__()
        self._camera = camera
        self._debug = debug
        self._selection_points = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self._frame_preprocessor = None

        self._tracked_object = TrackedObject(0)

        self._tracking_callback = None

    @property
    def tracked_object(self) -> TrackedObject:
        return self._tracked_object

    @property
    def frame(self):
        return self._frame

    @property
    def debug_frame(self):
        if self._debug:
            return self._debug_frame
        else:
            warnings.warn("Debugging is not enabled so there is no debug frame")
        return None

    def set_frame_preprocessor(self, preprocessor_func):
        self._frame_preprocessor = preprocessor_func

    def set_court_points(self, court_points):
        """
        Set a set of points that crops out a convex polygon from the image.
        So only on the cropped part will be detection
        :param court_points (list): list of points
        """

        self._selection_points = court_points

    def set_tracking_callback(self, tracking_callback: Callable[["ColorTracker"], None]):
        self._tracking_callback = tracking_callback

    def _add_new_tracker_point(self, point, min_point_distance: float, max_point_distance: float):
        try:
            dst = helpers.calculate_distance(self._tracked_object.tracked_points[-1], point)
            if max_point_distance > dst > min_point_distance:
                self._tracked_object.add_point(point)
        except IndexError:
            # It happens only when the queue is empty and we need a starting point
            self._tracked_object.add_point(point)

    def stop_tracking(self):
        """
        Stop the color tracking
        """

        self._is_running = False

    @staticmethod
    def _read_from_camera(camera, horizontal_flip: bool) -> np.ndarray:
        ret, frame = camera.read()

        if ret:
            if horizontal_flip:
                frame = cv2.flip(frame, 1)
        else:
            warnings.warn("There is no camera feed!")

        return frame

    def track(self, hsv_lower_value: Union[np.ndarray, List[int]], hsv_upper_value: Union[np.ndarray, List[int]],
              min_contour_area: Union[float, int], kernel: np.ndarray = None, min_track_point_distance: int = 20,
              horizontal_flip: bool = True):
        """
        With this we can start the tracking with the given parameters
        :param horizontal_flip: Flip input image horizontally
        :param hsv_lower_value: lowest acceptable hsv values
        :param hsv_upper_value: highest acceptable hsv values
        :param min_contour_area: minimum contour area for the detection. Below that the detection does not count
        :param kernel: structuring element to perform morphological operations on the mask image
        :param min_track_point_distance: minimum distance between the tracked and recognized points
        """

        self._is_running = True

        while True:
            self._frame = self._read_from_camera(self._camera, horizontal_flip)

            if self._frame_preprocessor is not None:
                self._frame = self._frame_preprocessor(self._frame)

            if (self._selection_points is not None) and (len(self._selection_points) > 0):
                self._frame = helpers.crop_out_polygon_convex(self._frame, self._selection_points)

            contours = helpers.find_object_contours(image=self._frame,
                                                    hsv_lower_value=hsv_lower_value,
                                                    hsv_upper_value=hsv_upper_value,
                                                    kernel=kernel)

            contours = helpers.filter_contours_by_area(contours, min_contour_area)
            contours = helpers.sort_contours_by_area(contours)
            object_centers = helpers.get_contour_centers(contours)

            self._tracked_object._last_object_contours = contours
            self._tracked_object._last_bounding_boxes = helpers.get_bbox_for_contours(contours)
            self._tracked_object._last_object_centers = object_centers

            if len(object_centers) > 0:
                self._add_new_tracker_point(object_centers[0], min_track_point_distance, np.inf)

            if self._debug:
                self._debug_frame = self._frame.copy()
                self._debug_frame = visualize.draw_debug_for_object(self._debug_frame, self._tracked_object)

            if self._tracking_callback is not None:
                self._tracking_callback(self)

            if not self._is_running:
                break
