import collections
import numpy as np

from color_tracker.utils import kalman_filter


class TrackedObject:
    def __init__(self, id: int, prediction: np.ndarray, max_nb_of_points: int = 20):
        self._id = id
        self._max_nb_of_points = max_nb_of_points
        self._tracked_points = collections.deque(maxlen=max_nb_of_points)

        self.KF = kalman_filter.KalmanFilter()
        self.prediction = prediction
        self.skipped_frames = 0

        self._last_object_contours = None
        self._last_object_centers = None
        self._last_bounding_boxes = None

    @property
    def tracked_points(self):
        return self._tracked_points

    @property
    def last_object_contours(self):
        return self._last_object_contours

    @property
    def last_object_centers(self):
        return self._last_object_centers

    @property
    def last_object_center(self):
        if len(self.last_object_centers) > 0:
            return self.last_object_centers[0]
        return None

    @property
    def last_bboxes(self):
        return self._last_bounding_boxes

    def add_point(self, point):
        self._tracked_points.append(point)


