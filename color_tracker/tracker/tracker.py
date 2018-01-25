import math
from collections import deque
from types import FunctionType
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from color_tracker.tracker.tracked_object import TrackedObject
from color_tracker.utils import helpers

_RGB_TYPE = "rgb"
_BGR_TYPE = "bgr"
_GRAY_TYPE = "gray"

_ACCEPTED_IMAGE_TYPES = [_RGB_TYPE, _BGR_TYPE, _GRAY_TYPE]


class ColorTracker(object):
    def __init__(self, camera, dist_thresh, max_frames_to_skip, max_trace_length, max_nb_of_points=None, debug=True):
        """
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param max_nb_of_points: Maximum number of points for storing. If it is set
        to None than it means there is no limit
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """

        super().__init__()
        self._camera = camera
        self.tracks = []
        self._debug = debug
        self._max_nb_of_objects = max_nb_of_points
        self._selection_points = None
        self._tracking_callback = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.trackIdCount = 0
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                             (0, 255, 255), (255, 0, 255), (255, 127, 255),
                             (127, 0, 255), (127, 0, 127)]

        self._frame_preprocessor = None

    def set_frame_preprocessor(self, preprocessor_func):
        self._frame_preprocessor = preprocessor_func

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

    def get_tracked_objects(self):
        return self._tracked_objects

    def _add_new_tracked_object(self):
        # TODO
        pass

    def _draw_debug_things(self, draw_tracker_points=True, draw_contour=True,
                           draw_object_center=True, draw_boundin_box=True):
        # TODO
        pass

    def _draw_tracker_points(self, debug_image):
        # TODO
        pass

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

    def track(self, hsv_lower_value, hsv_upper_value, min_contour_area=0, input_image_type="bgr", kernel=None):
        """
        With this we can start the tracking with the given parameters
        :param input_image_type: Type of the input image (color ordering). The standard is BGR because of OpenCV.
        That is the default image ordering but if you use a different type you have to set it here.
        (For example when you use a different input source or you used some preprocessing on the input image)
        :param hsv_lower_value: lowest acceptable hsv values
        :param hsv_upper_value: highest acceptable hsv values
        :param min_contour_area: minimum contour area for the detection. Below that the detection does not count
        :param kernel: structuring element to perform morphological operations on the mask image
        :param min_track_point_distance: minimum distance between the tracked and recognized points
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
                                                    min_contour_area=min_contour_area)

            detections = helpers.get_contour_centers(contours)

            ###############################################################

            if (len(self.tracks) == 0):
                for i in range(len(detections)):
                    track = TrackedObject(self.trackIdCount, detections[i])
                    self.trackIdCount += 1
                    self.tracks.append(track)

                # Calculate cost using sum of square distance between
                # predicted vs detected centroids
            N = len(self.tracks)
            M = len(detections)
            cost = np.zeros(shape=(N, M))  # Cost matrix
            for i in range(len(self.tracks)):
                for j in range(len(detections)):
                    try:
                        diff = self.tracks[i].prediction - detections[j]
                        distance = np.sqrt(diff[0][0] * diff[0][0] +
                                           diff[1][0] * diff[1][0])
                        cost[i][j] = distance
                    except:
                        pass

            # Let's average the squared ERROR
            cost = (0.5) * cost
            # Using Hungarian Algorithm assign the correct detected measurements
            # to predicted tracks
            assignment = []
            for _ in range(N):
                assignment.append(-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]

            # Identify tracks with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check for cost distance threshold.
                    # If cost is very high then un_assign (delete) the track
                    if (cost[i][assignment[i]] > self.dist_thresh):
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                    pass
                else:
                    self.tracks[i].skipped_frames += 1

            # If tracks are not detected for long time, remove them
            del_tracks = []
            for i in range(len(self.tracks)):
                if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                    del_tracks.append(i)
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                for id in del_tracks:
                    if id < len(self.tracks):
                        del self.tracks[id]
                        del assignment[id]

            # Now look for un_assigned detects
            un_assigned_detects = []
            for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

            # Start new tracks
            if (len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    track = TrackedObject(self.trackIdCount, detections[un_assigned_detects[i]])
                    self.trackIdCount += 1
                    self.tracks.append(track)

            # Update KalmanFilter state, lastResults and tracks trace
            for i in range(len(assignment)):
                self.tracks[i].KF.predict()

                if (assignment[i] != -1):
                    self.tracks[i].skipped_frames = 0
                    self.tracks[i].prediction = self.tracks[i].KF.correct(
                        detections[assignment[i]], 1)
                else:
                    self.tracks[i].prediction = self.tracks[i].KF.correct(
                        np.array([[0], [0]]), 0)

                if (len(self.tracks[i].trace) > self.max_trace_length):
                    for j in range(len(self.tracks[i].trace) -
                                   self.max_trace_length):
                        del self.tracks[i].trace[j]

                self.tracks[i].trace.append(self.tracks[i].prediction)
                self.tracks[i].KF.lastResult = self.tracks[i].prediction

            ###############################################################

            for i in range(len(self.tracks)):
                if (len(self.tracks[i].trace) > 1):
                    for j in range(len(self.tracks[i].trace) - 1):
                        # Draw trace line
                        x1 = self.tracks[i].trace[j][0][0]
                        y1 = self.tracks[i].trace[j][1][0]
                        x2 = self.tracks[i].trace[j + 1][0][0]
                        y2 = self.tracks[i].trace[j + 1][1][0]
                        clr = self.tracks[i].track_id % 9
                        cv2.line(self._debug_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 self.track_colors[clr], 2)

            ########################################x
            if self._tracking_callback is not None:
                try:
                    self._tracking_callback()
                except TypeError:
                    import warnings
                    warnings.warn(
                        "Tracker callback function is not working because of wrong arguments! It takes zero arguments")

            if not self._is_running:
                break

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
