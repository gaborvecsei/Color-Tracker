[![Codacy Badge](https://api.codacy.com/project/badge/Grade/67f0a9e168b3457385f2f7fcd09a9afa)](https://www.codacy.com/app/vecseigabor.x/Color-Tracker?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gaborvecsei/Color-Tracker&amp;utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/color-tracker.svg)](https://badge.fury.io/py/color-tracker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Color Tracker

Easy to use color tracking package for object tracking based on colors :art:.

## Samples

![yellow cruiser sample gif](art/yellow_cruiser.gif)

<img  width="400" src="art/ball_tracking.gif" />

## Install

- Python3
- OpenCV>=3
- NumPy

```
pip install color-tracker
```

## Basic Usage

There is a tracking **callback**:

- **tracking callback**: called at every frame of the tracking

You can find sample scripts at the `examples` folder

```python
import cv2
import color_tracker


def tracking_callback(tracker):
    frame = tracker.get_frame()
    debug_frame = tracker.get_debug_image()
    object_center = tracker.get_last_object_center()

    cv2.imshow("original frame", frame)
    cv2.imshow("debug frame", debug_frame)
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()
    print("Object center: {0}".format(object_center))


webcam = color_tracker.WebCamera(video_src=0)
webcam.start_camera()

tracker = color_tracker.ColorTracker(camera=webcam, max_nb_of_points=20, debug=True)
tracker.set_tracking_callback(tracking_callback=tracking_callback)
tracker.track(hsv_lower_value=(0, 100, 100),
              hsv_upper_value=(10, 255, 255),
              min_contour_area=1000)

webcam.release()
```

## Color Range Detection

With this tool you can easily determine the necessary *HSV* color values and kernel sizes for you app

(You can find this also in the `examples` folder)

```python
import color_tracker

cam = color_tracker.WebCamera(video_src=0)
cam.start_camera()

detector = color_tracker.HSVColorRangeDetector(camera=cam)
lower, upper, kernel = detector.detect()

print("Lower HSV color is: {0}".format(lower))
print("Upper HSV color is: {0}".format(upper))
print("Kernel shape is: {0}".format(kernel.shape))
```

## About

Gábor Vecsei

- [Website](https://gaborvecsei.com)
- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
