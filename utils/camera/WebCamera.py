import cv2

from utils import overrides
from utils.camera.Camera import Camera


class WebCamera(Camera):
    def __init__(self, video_src=0):
        super().__init__()
        self.__video_src = video_src

    @overrides(Camera)
    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self.__video_src)
        ret, self._frame = self._cam.read()
        return ret

    @overrides(Camera)
    def read(self):
        super().read()
        ret, self._frame = self._cam.read()
        if ret:
            if self.auto_undistortion:
                self._frame = self._undistort_image(self._frame)
            return True, self._frame
        else:
            return False, None

    @overrides(Camera)
    def release_camera(self):
        super().release_camera()
        self._cam.release()
