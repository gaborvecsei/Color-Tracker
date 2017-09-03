# Color Tracker

Simple color tracking with OpenCV 3.

It also has an "alert callback function" when the object crosses a line. You can set this line with `alert_y` and
the callback function with `alert_callback_function`.

## Sample

TODO: sample image

## Setup

You will need:

- Python 3
- OpenCV 3
- Numpy

## Run

`python app.py`

## Basic Usage

There are 2 callbacks:

- *tracking callback*: called at every frame
- *alert callback*: called only when it crossed the "alert line"

```python
import cv2

from color_tracker import ColorTracker
from utils import WebCamera


def tracking_callback(frame, debug_frame, object_center):
    cv2.imshow("original frame", frame)
    cv2.imshow("debug frame", debug_frame)
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()
    print("Object center: {0}".format(object_center))


def alert_callback():
    print("Crossed the line!")


if __name__ == "__main__":
    webcam = WebCamera(video_src=0)
    webcam.start_camera()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    tracker = ColorTracker(camera=webcam, max_nb_of_points=20, debug=True)

    tracker.set_tracking_callback(tracking_callback=tracking_callback)
    tracker.set_alert_callback(350, alert_callback)

    tracker.track(hsv_lower_value=(0, 100, 100),
                  hsv_upper_value=(10, 255, 255),
                  min_contour_area=1000,
                  kernel=kernel)

    webcam.release_camera()
```

## About

Gábor Vecsei

- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
- vecseigabor.x@gmail.com