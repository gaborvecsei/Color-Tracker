import cv2
from color_tracker.utils.camera.camera import Camera
from color_tracker.utils.helpers import overrides


class WebCamera(Camera):
    """
    Simple Webcamera
    """

    def __init__(self, video_src=0):
        """
        :param video_src (int): camera source code (it should be 0 or 1)
        """

        super().__init__()
        self._video_src = video_src

    @overrides(Camera)
    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self._video_src)
        self._ret, self._frame = self._cam.read()
        self._frame_height, self._frame_width, c = self._frame.shape
        return self._ret

    @overrides(Camera)
    def _read_from_camera(self):
        super()._read_from_camera()
        self._ret, self._frame = self._cam.read()
        if self._ret:
            if self.auto_undistortion:
                self._frame = self._undistort_image(self._frame)
            return True, self._frame
        else:
            return False, None

    @overrides(Camera)
    def release_camera(self):
        super().release_camera()
        self._cam.release()
