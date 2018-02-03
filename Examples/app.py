import cv2

import color_tracker


def tracking_callback():
    frame = tracker.get_frame()
    debug_frame = tracker.get_debug_image()

    cv2.imshow("original frame", frame)
    cv2.imshow("debug frame", debug_frame)
    key = cv2.waitKey(1)
    if key == 27:
        tracker.stop_tracking()


if __name__ == "__main__":
    webcam = color_tracker.WebCamera(video_src=0)
    webcam.start_camera()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    tracker = color_tracker.ColorTracker(camera=webcam, debug=True)

    tracker.set_tracking_callback(tracking_callback=tracking_callback)

    tracker.track(hsv_lower_value=(94, 173, 80),
                  hsv_upper_value=(128, 255, 255),
                  kernel=kernel,
                  min_contour_area=1000,
                  max_nb_of_objects=3,
                  max_number_of_points=100,
                  max_frames_to_skip=30,
                  maximum_distance_between_points=160)

    webcam.release_camera()
