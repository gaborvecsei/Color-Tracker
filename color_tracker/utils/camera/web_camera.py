import cv2

from color_tracker.utils.camera.base_camera import Camera


class WebCamera(Camera):
    """
    Simple Webcamera
    """

    def __init__(self, video_src=0):
        """
        :param video_src (int): camera source code. It can be an integer or the name of the video file.
        """

        super().__init__()
        self._video_src = video_src

    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self._video_src)
        self._ret, self._frame = self._cam.read()
        if not self._ret:
            raise Exception("No camera feed")
        self._frame_height, self._frame_width, c = self._frame.shape
        return self._ret

    def _read_from_camera(self):
        super()._read_from_camera()
        self._ret, self._frame = self._cam.read()
        if self._ret:
            if self._auto_undistortion:
                self._frame = self._undistort_image(self._frame)
            return True, self._frame
        else:
            return False, None

    def release_camera(self):
        super().release_camera()
        self._cam.release()