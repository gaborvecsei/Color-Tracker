import cv2

from ColorTracker import ColorTracker
from utils import WebCamera

webcam = WebCamera(video_src=0)
webcam.start_camera()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
alert_callback_function = lambda x: print("Crossed the line! at position: {0}".format(x))
tracking_callback = lambda x: print(x)

tracker = ColorTracker(webcam, 20, debug=True)
tracker.track((0, 100, 100),
              (10, 255, 255),
              min_contour_area=10,
              kernel=kernel,
              tracking_callback=tracking_callback,
              alert_y=320,
              alert_callback_function=alert_callback_function)

webcam.release_camera()
cv2.destroyAllWindows()
