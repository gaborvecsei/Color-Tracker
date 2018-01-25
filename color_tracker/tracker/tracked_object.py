from color_tracker.utils.kalman_filter import KalmanFilter
import numpy as np


class TrackedObject(object):
    def __init__(self, track_id, prediction):
        self.track_id = track_id  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []