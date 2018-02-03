from color_tracker.utils.kalman_filter import KalmanFilter
import numpy as np


class SimpleObject(object):
    def __init__(self, track_id=0):
        self.track_id = track_id
        self.trace = []
        self.last_detected_contour = None

    def get_trace(self):
        tmp_point_list = np.reshape(self.trace, (np.asarray(self.trace).shape[0], 2))
        return [tuple(x) for x in tmp_point_list]


class MultiTrackedObject(SimpleObject):
    def __init__(self, track_id, prediction):
        super().__init__(track_id)
        self.KF = KalmanFilter()
        self.prediction = np.asarray(prediction)
        self.skipped_frames = 0
