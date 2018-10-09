import collections


class TrackedObject:
    def __init__(self, id: int, max_nb_of_points: int = None):
        self._id = id
        self._tracked_points = collections.deque(maxlen=max_nb_of_points)
        self._skipped_frames = 0

        self._last_object_contour = None
        self._last_bounding_box = None

    @property
    def id(self):
        return self._id

    @property
    def skipped_frames(self):
        return self._skipped_frames

    @skipped_frames.setter
    def skipped_frames(self, value):
        self._skipped_frames = value

    @property
    def tracked_points(self):
        return self._tracked_points

    @property
    def last_point(self):
        return self.tracked_points[-1]

    @property
    def last_object_contour(self):
        return self._last_object_contour

    @last_object_contour.setter
    def last_object_contour(self, value):
        self._last_object_contour = value

    @property
    def last_bbox(self):
        return self._last_bounding_box

    @last_bbox.setter
    def last_bbox(self, value):
        self._last_bounding_box = value

    def add_point(self, point):
        self._tracked_points.append(point)
