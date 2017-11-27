import cv2

import color_tracker


def tracking_callback():
    frame = tracker.get_frame()
    debug_frame = tracker.get_debug_image()
    object_center = tracker.get_last_object_center()

    cv2.imshow("original frame", frame)
    cv2.imshow("debug frame", debug_frame)
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()
    print("Object center: {0}".format(object_center))


if __name__ == "__main__":
    webcam = color_tracker.WebCamera(video_src=0)
    webcam.start_camera()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    tracker = color_tracker.ColorTracker(camera=webcam, max_nb_of_points=20, debug=True)

    tracker.set_tracking_callback(tracking_callback=tracking_callback)

    tracker.track(hsv_lower_value=(0, 100, 100),
                  hsv_upper_value=(10, 255, 255),
                  min_contour_area=1000,
                  kernel=kernel,
                  input_image_type="bgr")

    webcam.release_camera()
