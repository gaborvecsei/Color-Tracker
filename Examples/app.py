import cv2

import color_tracker


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
    webcam = color_tracker.WebCamera(video_src=0)
    webcam.start_camera()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    tracker = color_tracker.ColorTracker(camera=webcam, max_nb_of_points=20, debug=True)

    tracker.set_tracking_callback(tracking_callback=tracking_callback)
    tracker.set_alert_callback(350, alert_callback)

    tracker.track(hsv_lower_value=(0, 100, 100),
                  hsv_upper_value=(10, 255, 255),
                  min_contour_area=1000,
                  kernel=kernel)

    webcam.release_camera()
