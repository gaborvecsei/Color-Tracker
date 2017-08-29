# Color Tracker

Simple color tracking with OpenCV 3.

It also has a "callback function" when the object crosses a line. You can set this line with `alert_y` and
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

- tracking callback: called every time the object's position changes
- alert callback: called only when it crossed the "alert line"

The two callback are the same with *1 parameter* which is the *object's center coordinates*.

```python
from ColorTracker import ColorTracker
from utils import WebCamera


webcam = WebCamera(video_src=0)
webcam.start_camera()

alert_callback_function = lambda x: print("Crossed the line! at position: {0}".format(x))
tracking_callback = lambda x: print("Current position: {0}".format(x))

tracker = ColorTracker(webcam, 20, debug=True)
tracker.track((0, 100, 100),
              (10, 255, 255),
              min_contour_area=1000,
              tracking_callback=tracking_callback,
              alert_y=320,
              alert_callback_function=alert_callback_function)

webcam.release_camera()
```

## TODOs

- Cleaner code

## About

Gábor Vecsei

- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
- vecseigabor.x@gmail.com