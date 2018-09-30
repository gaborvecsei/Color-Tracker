import colorsys

import cv2
import numpy as np
from color_tracker.utils.tracker_object import TrackedObject


def random_colors(nb_of_colors: int, brightness: float = 1.0):
    hsv = [(i / nb_of_colors, 1, brightness) for i in range(nb_of_colors)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    colors = colors * 255
    colors = colors.astype(np.uint8)
    np.random.shuffle(colors)
    return colors


def draw_tracker_points(points, debug_image):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        rectangle_offset = 4
        rectangle_pt1 = tuple(x - rectangle_offset for x in points[i])
        rectangle_pt2 = tuple(x + rectangle_offset for x in points[i])
        cv2.rectangle(debug_image, rectangle_pt1, rectangle_pt2, (255, 255, 255), 1)
        cv2.line(debug_image, tuple(points[i - 1]), tuple(points[i]), (255, 255, 255), 1)
    return debug_image


def draw_debug_for_object(debug_frame, tracked_object: TrackedObject):
    contours = tracked_object.last_object_contours
    bboxes = tracked_object.last_bboxes
    points = tracked_object.tracked_points

    if contours is not None:
        for c in contours:
            cv2.drawContours(debug_frame, [c], -1, (0, 255, 0), cv2.FILLED)

    if bboxes is not None:
        for b in bboxes:
            x1, y1, x2, y2 = b
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    if points is not None and len(points) > 0:
        try:
            cv2.circle(debug_frame, tuple(points[0].astype(int)), 3, (0, 0, 255), -1)
        except:
            pass
        draw_tracker_points(points, debug_frame)

    return debug_frame
