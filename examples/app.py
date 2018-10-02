import cv2

import color_tracker

# You can determine these values with the HSVColorRangeDetector()
HSV_LOWER_VALUE = [155, 103, 82]
HSV_UPPER_VALUE = [178, 255, 255]


def tracking_callback(tracker: color_tracker.ColorTracker):
    # Visualizing the original frame and the debugger frame
    cv2.imshow("original frame", tracker.frame)
    cv2.imshow("debug frame", tracker.debug_frame)

    # Stop the script when we press ESC
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()

    for obj in tracker.tracked_objects:
        print("Object {0} center {1}".format(obj.id, obj.last_point))


# Init the webcamera
# You can use the Camera class which is only a wrapper around the original
# OpenCV VideoCapture object. Of course because of that you can use the original VideoCapture class like:
# webcam = cv2.VideoCapture(0), (then you won't need to call the start_camera() function)

webcam = color_tracker.WebCamera(video_src=0)
webcam.start_camera()

# Creating a kernel for the morphology operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Init the ColorTracker object
tracker = color_tracker.ColorTracker(max_nb_of_objects=5, max_nb_of_points=20, debug=True)

# Setting a callback which is called at every iteration
tracker.set_tracking_callback(tracking_callback=tracking_callback)

# Start the actual tracking of the object
tracker.track(webcam,
              hsv_lower_value=HSV_LOWER_VALUE,
              hsv_upper_value=HSV_UPPER_VALUE,
              min_contour_area=2500,
              kernel=kernel)

webcam.release()
