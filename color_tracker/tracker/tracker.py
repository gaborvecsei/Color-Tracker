import warnings
from collections import deque
from typing import Union, List, Callable

import cv2
import numpy as np
from scipy import optimize

from color_tracker.utils.tracker_object import TrackedObject

from color_tracker.utils import helpers, visualize
from color_tracker.utils.camera import Camera


class ColorTracker(object):
    def __init__(self, camera: Union[Camera, cv2.VideoCapture], max_nb_of_points: int = None, debug: bool = True):
        super().__init__()
        self._camera = camera
        self._debug = debug
        self._selection_points = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self._frame_preprocessor = None

        self._tracked_objects = []
        self._tracked_object_id_count = 0
        self._max_nb_of_points = max_nb_of_points

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

    # def _add_new_tracker_point(self, point, min_point_distance: float, max_point_distance: float):
    #     try:
    #         dst = helpers.calculate_distance(self._tracked_object.tracked_points[-1], point)
    #         if max_point_distance > dst > min_point_distance:
    #             self._tracked_object.add_point(point)
    #     except IndexError:
    #         # It happens only when the queue is empty and we need a starting point
    #         self._tracked_object.add_point(point)

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
              min_contour_area: Union[float, int], kernel: np.ndarray = None,
              min_track_point_distance: int = 20, max_track_point_distance: int = 150, max_skipped_frames: int = 20,
              max_nb_of_objects: int = 2,
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
            if len(contours) > max_nb_of_objects:
                contours = contours[:max_nb_of_objects]
            object_centers = helpers.get_contour_centers(contours)

            if len(self._tracked_objects) == 0:
                for d in object_centers:
                    track = TrackedObject(self._tracked_object_id_count, d, self._max_nb_of_points)
                    self._tracked_object_id_count += 1
                    self._tracked_objects.append(track)

            cost = np.zeros(shape=(len(self._tracked_objects), len(object_centers)))

            for i, to in enumerate(self._tracked_objects):
                for j, oc in enumerate(object_centers):
                    diff = to.prediction - oc
                    distance = np.sqrt(diff[0][0] ** 2 + diff[1][0] ** 2)
                    cost[i][j] = distance

            cost = 0.5 * cost

            assignment = [-1 for i in range(len(self._tracked_objects))]
            row_index, column_index = optimize.linear_sum_assignment(cost)
            for i in range(len(row_index)):
                assignment[row_index[i]] = column_index[i]

            un_assigned_tracks = []
            for i in range(len(assignment)):
                if assignment[i] != -1:
                    if cost[i][assignment[i]] > max_track_point_distance:
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                else:
                    self._tracked_objects[i].skipped_frames += 1

            for i, to in enumerate(self._tracked_objects):
                if to.skipped_frames > max_skipped_frames:
                    del self._tracked_objects[i]
                    del assignment[i]

            un_assigned_detections = [i for i in range(len(object_centers)) if i not in assignment]

            if len(un_assigned_detections) != 0:
                for uad in un_assigned_detections:
                    track = TrackedObject(self._tracked_object_id_count, object_centers[uad], self._max_nb_of_points)
                    self._tracked_object_id_count += 1
                    self._tracked_objects.append(track)

            for i in range(len(assignment)):
                self._tracked_objects[i].KF.predict()

                if assignment[i] != -1:
                    self._tracked_objects[i].skipped_frames = 0
                    self._tracked_objects[i].prediction = self._tracked_objects[i].KF.correct(
                        object_centers[assignment[i]], 1)
                    try:
                        self._tracked_objects[i].last_detected_contour = contours[i]
                    except Exception:
                        pass
                else:
                    self._tracked_objects[i].prediction = self._tracked_objects[i].KF.correct(
                        np.array([[0], [0]]), 0)

                self._tracked_objects[i].tracked_points.append(self._tracked_objects[i].prediction)
                self._tracked_objects[i].KF.lastResult = self._tracked_objects[i].prediction

            for o, c, ce in zip(self.tracked_objects, contours, object_centers):
                o._last_object_contours = [c]
                o._last_bounding_boxes = helpers.get_bbox_for_contours([c])
                o._last_object_centers = [ce]

            # if len(object_centers) > 0:
            #     self._add_new_tracker_point(object_centers[0], min_track_point_distance, np.inf)

            if self._debug:
                self._debug_frame = self._frame.copy()
                # if len(self._tracked_objects) > 0:
                #     for o in self._tracked_objects:
                #         self._debug_frame = visualize.draw_debug_for_object(self._debug_frame, o)
                self._debug_frame = self._draw_debug_things_for_multi_tracking(self._debug_frame)

            if self._tracking_callback is not None:
                self._tracking_callback(self)

            if not self._is_running:
                break

    def _draw_debug_things_for_multi_tracking(self, debug_frame, draw_tracker_points=True, draw_contour=True,
                                              draw_object_center=True, draw_bounding_box=True):
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

        for to in self._tracked_objects:
            tmp_color_index = to._id % len(color_list)

            if draw_contour:
                if to.last_object_contours is not None:
                    cv2.drawContours(debug_frame, to.last_object_contours, -1, (255, 255, 0), cv2.FILLED)

            if draw_tracker_points:
                if len(to.tracked_points) > 1:
                    for j in range(len(to.tracked_points) - 1):
                        x1 = to.tracked_points[j][0][0]
                        y1 = to.tracked_points[j][1][0]
                        x2 = to.tracked_points[j + 1][0][0]
                        y2 = to.tracked_points[j + 1][1][0]
                        cv2.line(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 color_list[tmp_color_index], 2)

            if draw_object_center:
                if len(to.tracked_points) > 0:
                    x = to.tracked_points[-1][0][0]
                    y = to.tracked_points[-1][1][0]
                    cv2.circle(debug_frame, (int(x), int(y)), 4, (0, 0, 255), cv2.FILLED)

        return debug_frame
