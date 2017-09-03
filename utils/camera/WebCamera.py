import cv2

from utils.camera.Camera import Camera
from utils.utils import overrides


class WebCamera(Camera):
    def __init__(self, video_src=0):
        super().__init__()
        self.__video_src = video_src

    @overrides(Camera)
    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self.__video_src)
        self._ret, self._frame = self._cam.read()
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
