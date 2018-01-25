import math
from collections import deque
from types import FunctionType
import numpy as np
import cv2
from scipy import optimize

from color_tracker.tracker.tracked_object import TrackedObject
from color_tracker.utils import helpers

_RGB_TYPE = "rgb"
_BGR_TYPE = "bgr"
_GRAY_TYPE = "gray"

_ACCEPTED_IMAGE_TYPES = [_RGB_TYPE, _BGR_TYPE, _GRAY_TYPE]


class ColorTracker(object):
    def __init__(self, camera, debug=True):
        """
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """

        super().__init__()
        self._camera = camera
        self._tracked_objects = []
        self._debug = debug
        self._selection_points = None
        self._tracking_callback = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self._track_id_count = 0
        self._frame_preprocessor = None

    def set_frame_preprocessor(self, preprocessor_func):
        self._frame_preprocessor = preprocessor_func

    def set_court_points(self, court_points):
        """
        Set a set of points that crops out a convex polygon from the image.
        So only on the cropped part will be detection
        :param court_points: list of points
        """

        self._selection_points = court_points

    def set_tracking_callback(self, tracking_callback):
        if not isinstance(tracking_callback, FunctionType):
            raise Exception("tracking_callback is not a valid Function with type: FunctionType!")
        self._tracking_callback = tracking_callback

    def get_tracked_objects(self):
        return self._tracked_objects

    def _draw_debug_things(self, draw_tracker_points=True, draw_contour=True,
                           draw_object_center=True, draw_bounding_box=True):

        for to in self._tracked_objects:

            tmp_color_index = to.track_id % len(helpers.color_list)

            if draw_contour:
                if to.last_detected_contour is not None:
                    cv2.drawContours(self._debug_frame, [to.last_detected_contour], -1, (255, 255, 0), cv2.FILLED)

            if draw_bounding_box:
                if to.last_detected_contour is not None:
                    bbox = helpers.get_bounding_box_for_contour(to.last_detected_contour)
                    cv2.rectangle(self._debug_frame, bbox[0], bbox[1], tmp_color_index, 2)

            if draw_tracker_points:
                if len(to.trace) > 1:
                    for j in range(len(to.trace) - 1):
                        x1 = to.trace[j][0][0]
                        y1 = to.trace[j][1][0]
                        x2 = to.trace[j + 1][0][0]
                        y2 = to.trace[j + 1][1][0]
                        cv2.line(self._debug_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 helpers.color_list[tmp_color_index], 2)

            if draw_object_center:
                if len(to.trace) > 0:
                    x = to.trace[-1][0][0]
                    y = to.trace[-1][1][0]
                    cv2.circle(self._debug_frame, (int(x), int(y)), 4, (0, 0, 255), cv2.FILLED)

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

    def track(self, hsv_lower_value, hsv_upper_value, min_contour_area=0, max_number_of_points=300,
              max_nb_of_objects=10, maximum_distance_between_points=200, max_frames_to_skip=10, kernel=None,
              input_image_type="bgr"):
        """
        With this we can start the tracking with the given parameters
        :param max_nb_of_objects:
        :param max_number_of_points: Maximum number of points
        :param max_frames_to_skip:
        :param maximum_distance_between_points:
        :param input_image_type: Type of the input image (color ordering). The standard is BGR because of OpenCV.
        That is the default image ordering but if you use a different type you have to set it here.
        (For example when you use a different input source or you used some preprocessing on the input image)
        :param hsv_lower_value: lowest acceptable hsv values
        :param hsv_upper_value: highest acceptable hsv values
        :param min_contour_area: minimum contour area for the detection. Below that the detection does not count
        :param kernel: structuring element to perform morphological operations on the mask image
        """

        self._is_running = True

        while True:
            self._read_from_camera()

            if self._frame_preprocessor is not None:
                self._frame = self._frame_preprocessor(self._frame)

            self._check_and_fix_image_type(input_image_type=input_image_type)

            if (self._selection_points is not None) and (self._selection_points != []):
                self._frame = helpers.crop_out_polygon_convex(self._frame, self._selection_points)

            img = self._frame.copy()
            self._debug_frame = self._frame.copy()

            contours = helpers.find_object_contours(image=img,
                                                    hsv_lower_value=hsv_lower_value,
                                                    hsv_upper_value=hsv_upper_value,
                                                    kernel=kernel,
                                                    max_nb_of_objects=max_nb_of_objects,
                                                    min_contour_area=min_contour_area)

            object_centers = helpers.get_contour_centers(contours)

            self._kalman_filter_multi_object_tracking(object_centers, contours, maximum_distance_between_points,
                                                      max_frames_to_skip, max_number_of_points)

            if self._debug:
                self._draw_debug_things()

            if self._tracking_callback is not None:
                try:
                    self._tracking_callback()
                except TypeError:
                    import warnings
                    warnings.warn(
                        "Tracker callback function is not working because of wrong arguments! It takes zero arguments")

            if not self._is_running:
                break

    def _kalman_filter_multi_object_tracking(self, object_centers, contours, maximum_distance_between_points,
                                             max_frames_to_skip,
                                             max_number_of_points):
        if len(self._tracked_objects) == 0:
            for d in object_centers:
                track = TrackedObject(self._track_id_count, d)
                self._track_id_count += 1
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
                if cost[i][assignment[i]] > maximum_distance_between_points:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                self._tracked_objects[i].skipped_frames += 1

        for i, to in enumerate(self._tracked_objects):
            if to.skipped_frames > max_frames_to_skip:
                del self._tracked_objects[i]
                del assignment[i]

        un_assigned_detections = [i for i in range(len(object_centers)) if i not in assignment]

        if len(un_assigned_detections) != 0:
            for uad in un_assigned_detections:
                track = TrackedObject(self._track_id_count, object_centers[uad])
                self._track_id_count += 1
                self._tracked_objects.append(track)

        for i in range(len(assignment)):
            self._tracked_objects[i].KF.predict()

            if assignment[i] != -1:
                self._tracked_objects[i].skipped_frames = 0
                self._tracked_objects[i].prediction = self._tracked_objects[i].KF.correct(
                    object_centers[assignment[i]], 1)
                self._tracked_objects[i].last_detected_contour = contours[i]
            else:
                self._tracked_objects[i].prediction = self._tracked_objects[i].KF.correct(
                    np.array([[0], [0]]), 0)

            # TODO: use deque inside TrackedObject
            if len(self._tracked_objects[i].trace) > max_number_of_points:
                self._tracked_objects[i].trace = self._tracked_objects[i].trace[-max_number_of_points:]

            self._tracked_objects[i].trace.append(self._tracked_objects[i].prediction)
            self._tracked_objects[i].KF.lastResult = self._tracked_objects[i].prediction

    def _check_and_fix_image_type(self, input_image_type="bgr"):
        input_image_type = input_image_type.lower()

        if input_image_type not in _ACCEPTED_IMAGE_TYPES:
            raise ValueError(
                "Image type: {0} is not in accepted types: {1}".format(input_image_type, _ACCEPTED_IMAGE_TYPES))

        try:
            if input_image_type == "rgb":
                self._frame = cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)
            elif input_image_type == "gray":
                self._frame = cv2.cvtColor(self._frame, cv2.COLOR_GRAY2BGR)
        except cv2.error as e:
            print("Could not convert to BGR image format. Maybe you should define another input_image_type")
            raise

    def get_debug_image(self):
        if self._debug:
            return self._debug_frame
        else:
            import warnings
            warnings.warn("Debugging is not enabled so there is no debug frame")

    def get_frame(self):
        return self._frame
