import colorsys
import random
from typing import Tuple

import cv2

from color_tracker.utils.tracker_object import TrackedObject


def random_colors(nb_of_colors: int, brightness: float = 1.0):
    hsv = [(i / nb_of_colors, 1, brightness) for i in range(nb_of_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # note: we need to use list here with values [0, 255] as python built in scalar types,
    # because OpenCV functions can't get numpy dtypes for color
    colors = [list(map(lambda x: int(x * 255), c)) for c in colors]
    random.shuffle(colors)
    return colors


def draw_tracker_points(points, debug_image, color: Tuple[int, int, int] = (255, 255, 255)):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        rectangle_offset = 4
        rectangle_pt1 = tuple(x - rectangle_offset for x in points[i])
        rectangle_pt2 = tuple(x + rectangle_offset for x in points[i])
        cv2.rectangle(debug_image, rectangle_pt1, rectangle_pt2, color, 1)
        cv2.line(debug_image, tuple(points[i - 1]), tuple(points[i]), color, 1)
    return debug_image


def draw_debug_frame_for_object(debug_frame, tracked_object: TrackedObject, color: Tuple[int, int, int] = (255, 255, 255)):
    # contour = tracked_object.last_object_contour
    bbox = tracked_object.last_bbox
    points = tracked_object.tracked_points

    # if contour is not None:
    #     cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), cv2.FILLED)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(debug_frame, "Id {0}".format(tracked_object.id), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))

    if points is not None and len(points) > 0:
        draw_tracker_points(points, debug_frame, color)
        cv2.circle(debug_frame, tuple(points[-1]), 3, (0, 0, 255), -1)

    return debug_frame
