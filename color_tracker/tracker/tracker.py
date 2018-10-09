import warnings
from typing import Union, List, Callable

import cv2
import numpy as np

from color_tracker.utils import helpers, visualize
from color_tracker.utils.camera import Camera
from color_tracker.utils.tracker_object import TrackedObject


class ColorTracker(object):
    def __init__(self, max_nb_of_objects: int = None,
                 max_nb_of_points: int = None, debug: bool = True):
        """
        :param max_nb_of_points: Maxmimum number of points for storing. If it is set
        to None than it means there is no limit
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """

        super().__init__()
        self._debug = debug
        self._max_nb_of_objects = max_nb_of_objects
        self._max_nb_of_points = max_nb_of_points
        self._debug_colors = visualize.random_colors(max_nb_of_objects)
        self._selection_points = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self._frame_preprocessor = None

        self._tracked_objects = []
        self._tracked_object_id_count = 0

        self._tracking_callback = None

    @property
    def tracked_objects(self) -> List[TrackedObject]:
        return self._tracked_objects

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
            raise ValueError("There is no camera feed")

        return frame

    def _init_new_tracked_object(self, obj_center):
        tracked_obj = TrackedObject(self._tracked_object_id_count, self._max_nb_of_points)
        tracked_obj.add_point(obj_center)
        self._tracked_object_id_count += 1
        self._tracked_objects.append(tracked_obj)

    def track(self, camera: Union[Camera, cv2.VideoCapture], hsv_lower_value: Union[np.ndarray, List[int]],
              hsv_upper_value: Union[np.ndarray, List[int]], min_contour_area: Union[float, int] = 0,
              kernel: np.ndarray = None, horizontal_flip: bool = True, max_track_point_distance: int = 100,
              max_skipped_frames: int = 24):
        """
        With this we can start the tracking with the given parameters
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param max_skipped_frames: An object can be hidden for this many frames, after that it will be counted as a new
        :param max_track_point_distance: maximum distance between tracking points
        :param horizontal_flip: Flip input image horizontally
        :param hsv_lower_value: lowest acceptable hsv values
        :param hsv_upper_value: highest acceptable hsv values
        :param min_contour_area: minimum contour area for the detection. Below that the detection does not count
        :param kernel: structuring element to perform morphological operations on the mask image
        """

        self._is_running = True

        while True:
            self._frame = self._read_from_camera(camera, horizontal_flip)

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
            if self._max_nb_of_objects is not None and self._max_nb_of_objects > 0:
                contours = contours[:self._max_nb_of_objects]
            bboxes = helpers.get_bbox_for_contours(contours)
            object_centers = helpers.get_contour_centers(contours)

            # Init the list of tracked objects if it's empty
            if len(self._tracked_objects) == 0:
                for obj_center in object_centers:
                    self._init_new_tracked_object(obj_center)

            # Constructing cost matrix (matrix with the distances from points to other points)
            cost_mtx = helpers.calculate_distance_mtx(self._tracked_objects, object_centers)

            # Solve assignment problem
            assignment = helpers.solve_assignment(cost_mtx)

            # Refine assignment list and objects's skipped frames
            for i in range(len(assignment)):
                if assignment[i] != -1:
                    if cost_mtx[i][assignment[i]] > max_track_point_distance:
                        assignment[i] = -1
                else:
                    self._tracked_objects[i].skipped_frames += 1

            # Remove tracked object if the object skipped to many frames, so it was not detected
            helpers.remove_object_if_too_many_frames_skipped(self._tracked_objects, assignment, max_skipped_frames)

            # Check for new objects and initialize them
            un_assigned_detections = [i for i in range(len(object_centers)) if i not in assignment]
            if len(un_assigned_detections) != 0:
                if len(self._tracked_objects) < self._max_nb_of_objects:
                    for i in un_assigned_detections:
                        self._init_new_tracked_object(object_centers[i])

            # Refresh tracked objects (reset "skipped frames" counter and add new object center to the queue)
            for i in range(len(assignment)):
                if assignment[i] != -1:
                    self._tracked_objects[i].skipped_frames = 0
                    self._tracked_objects[i].add_point(object_centers[assignment[i]])

                    if len(contours) > i:
                        self._tracked_objects[i].last_object_contour = contours[i]
                        self._tracked_objects[i].last_bbox = bboxes[i]

            if self._debug:
                self._debug_frame = self._frame.copy()
                for i, tracked_obj in enumerate(self._tracked_objects):
                    self._debug_frame = visualize.draw_debug_frame_for_object(self._debug_frame,
                                                                              tracked_obj,
                                                                              self._debug_colors[i])

            if self._tracking_callback is not None:
                self._tracking_callback(self)

            if not self._is_running:
                break
