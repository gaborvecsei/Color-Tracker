import argparse
from functools import partial

import cv2

import color_tracker

# You can determine these values with the HSVColorRangeDetector()
HSV_LOWER_VALUE = [155, 103, 82]
HSV_UPPER_VALUE = [178, 255, 255]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-low", "--low", nargs=3, type=int, default=HSV_LOWER_VALUE,
                        help="Lower value for the HSV range. Default = 155, 103, 82")
    parser.add_argument("-high", "--high", nargs=3, type=int, default=HSV_UPPER_VALUE,
                        help="Higher value for the HSV range. Default = 178, 255, 255")
    parser.add_argument("-c", "--contour-area", type=float, default=2500,
                        help="Minimum object contour area. This controls how small objects should be detected. Default = 2500")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    return args


def tracking_callback(tracker: color_tracker.ColorTracker, verbose: bool = True):
    # Visualizing the original frame and the debugger frame
    cv2.imshow("original frame", tracker.frame)
    cv2.imshow("debug frame", tracker.debug_frame)

    # Stop the script when we press ESC
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()

    if verbose:
        for obj in tracker.tracked_objects:
            print("Object {0} center {1}".format(obj.id, obj.last_point))


def main():
    args = get_args()

    # Creating a kernel for the morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Init the ColorTracker object
    tracker = color_tracker.ColorTracker(max_nb_of_objects=5, max_nb_of_points=20, debug=True)

    # Setting a callback which is called at every iteration
    callback = partial(tracking_callback, verbose=args.verbose)
    tracker.set_tracking_callback(tracking_callback=callback)

    # Start tracking with a camera
    with color_tracker.WebCamera(video_src=0) as webcam:
        # Start the actual tracking of the object
        tracker.track(webcam,
                      hsv_lower_value=args.low,
                      hsv_upper_value=args.high,
                      min_contour_area=args.contour_area,
                      kernel=kernel)


if __name__ == "__main__":
    main()
