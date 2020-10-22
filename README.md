[![Codacy Badge](https://api.codacy.com/project/badge/Grade/67f0a9e168b3457385f2f7fcd09a9afa)](https://www.codacy.com/app/vecseigabor.x/Color-Tracker?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gaborvecsei/Color-Tracker&amp;utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/color-tracker.svg)](https://badge.fury.io/py/color-tracker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3](https://img.shields.io/badge/Python-3-brightgreen.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/101786270.svg)](https://zenodo.org/badge/latestdoi/101786270)


# Color Tracker - Multi Object Tracker

Easy to use **multi object tracking** package based on colors :art:

<img src="art/yellow_cruiser.gif" width="400" alt="yellow-cruiser"></a> <img src="art/ball_tracking.gif" width="400" alt="ball-tracking"></a>

## Install

```
pip install color-tracker
```

```
pip install git+https://github.com/gaborvecsei/Color-Tracker.git
```

## Object Tracker

- Check out the **[examples folder](examples)**, or go straight to the **[sample tracking app](examples/tracking.py)** which is an extended version of the script below.
This script tracks the red-ish objects, if you'd like to track another color, then start with the `hsv_color_detector.py` script 
    ``` python
    $ python examples/tracking.py --help
  
  
    usage: tracking.py [-h] [-low LOW LOW LOW] [-high HIGH HIGH HIGH]
                   [-c CONTOUR_AREA] [-v]

    optional arguments:
      -h, --help            show this help message and exit
      -low LOW LOW LOW, --low LOW LOW LOW
                            Lower value for the HSV range. Default = 155, 103, 82
      -high HIGH HIGH HIGH, --high HIGH HIGH HIGH
                            Higher value for the HSV range. Default = 178, 255,
                            255
      -c CONTOUR_AREA, --contour-area CONTOUR_AREA
                            Minimum object contour area. This controls how small
                            objects should be detected. Default = 2500
      -v, --verbose
    ```
- Simple script:

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

## Color Range Detection

This is a tool which you can use to easily determine the necessary *HSV* color values and kernel sizes for you app

You can find **[the HSV Color Detector code here](examples/hsv_color_detector.py)**

``` python
python examples/hsv_color_detector.py
```

## Donate :coffee:

If you feel like it is a **useful package** and it **saved you time and effor**, then you can donate a coffe for me, so I can keep on staying awake for days :smiley: 

<a href='https://ko-fi.com/A0A5KN4E' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi5.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

## About

Gábor Vecsei

- [Website](https://gaborvecsei.com)
- [Personal Blog](https://gaborvecsei.com)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)

```
@misc{vecsei2018colortracker,
      doi = {10.5281/ZENODO.4097717},
      howpublished={\url{https://github.com/gaborvecsei/Color-Tracker}},
      author = {Gabor Vecsei},
      title = {Color Tracker - Multi Object Tracker},
      year = {2018},
      copyright = {MIT License}
}
```
