[![Codacy Badge](https://api.codacy.com/project/badge/Grade/67f0a9e168b3457385f2f7fcd09a9afa)](https://www.codacy.com/app/vecseigabor.x/Color-Tracker?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gaborvecsei/Color-Tracker&amp;utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/color-tracker.svg)](https://badge.fury.io/py/color-tracker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Color Tracker

Easy to use color tracking package for object tracking based on colors :art:

## Samples

<p align="center">
<img src="art/yellow_cruiser.gif" width="600" alt="yellow-cruiser"></a><br/>
<img src="art/ball_tracking.gif" width="600" alt="yellow-cruiser"></a><br/>
</p>

## Install

- Python3
- OpenCV>=3
- NumPy

```
pip install color-tracker
```

## Usage

- Basic usage:

``` python
import cv2
import color_tracker


def tracker_callback(t: color_tracker.ColorTracker):
    cv2.imshow("debug", t.debug_frame)
    cv2.waitKey(1)


tracker = color_tracker.ColorTracker(max_nb_of_objects=1, max_nb_of_points=20, debug=True)
tracker.set_tracking_callback(tracker_callback)

with color_tracker.WebCamera() as cam:
    # Define your custom Lower and Upper HSV values
    tracker.track(cam, [155, 103, 82], [178, 255, 255], max_skipped_frames=24)
```

Check out the [examples folder](examples), or go straight to the [sample tracking app](examples/tracking.py) which is an extended
version of the script above

## Color Range Detection

This is a tool which you can use to easily determine the necessary *HSV* color values and kernel sizes for you app

You can find [the code here](examples/hsv_color_detector.py)

## About

Gábor Vecsei

- [Website](https://gaborvecsei.com)
- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
