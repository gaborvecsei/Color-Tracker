# Color Tracker

Color tracking module with OpenCV 3.

It also has an "alert callback function" when the object crosses a line. You can set this line with `alert_y` and
the callback function with `alert_callback_function`.

## Sample

TODO: sample gif

## Setup & install

You will need:

- Python 3
- OpenCV 3
- Numpy

Install:

`python setup.py install`

## Basic Usage

There are 2 callbacks:

- *tracking callback*: called at every frame
- *alert callback*: called only when it crossed the "alert line"

You can find sample scripts at the `Examples` folder

## Color Range Detection

Just run this little script and you can get the necessary information for the color detection.

(You can find this also in the `Examples` folder)

```python
import color_tracker

cam = color_tracker.WebCamera(video_src=0)
cam.start_camera()

detector = color_tracker.HSVColorRangeDetector(camera=cam)
lower, upper, kernel = detector.detect()

print("Lower HSV color is: {0}".format(lower))
print("Upper HSV color is: {0}".format(upper))
print("Kernel is: {0}".format(kernel))
```

## About

Gábor Vecsei

- [Personal Blog](https://gaborvecsei.wordpress.com/)
- [LinkedIn](https://www.linkedin.com/in/gaborvecsei)
- [Twitter](https://twitter.com/GAwesomeBE)
- [Github](https://github.com/gaborvecsei)
- vecseigabor.x@gmail.com