from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy import optimize

from color_tracker.utils.tracker_object import TrackedObject


def crop_out_polygon_convex(image: np.ndarray, point_array: np.ndarray) -> np.ndarray:
    """
    Crops out a convex polygon given from a list of points from an image
    :param image: Opencv BGR image
    :param point_array: list of points that defines a convex polygon
    :return: Cropped out image
    """

    point_array = np.reshape(cv2.convexHull(point_array), point_array.shape)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([point_array], dtype=np.int32)
    ignore_mask_color = (255, 255, 255)
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def resize_img(image: np.ndarray, min_width: int, min_height: int) -> np.ndarray:
    """
    Resize the image with keeping the aspect ratio.
    :param image: image
    :param min_width: minimum width of the image
    :param min_height: minimum height of the image
    :return: resized image
    """

    h, w = image.shape[:2]

    new_w = w
    new_h = h

    if w > min_width:
        new_w = min_width
        new_h = int(h * (float(new_w) / w))

    h, w = (new_h, new_w)
    if h > min_height:
        new_h = min_height
        new_w = int(w * (float(new_h) / h))

    return cv2.resize(image, (new_w, new_h))


def sort_contours_by_area(contours: np.ndarray, descending: bool = True) -> np.ndarray:
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=descending)
    return contours


def filter_contours_by_area(contours: np.ndarray, min_area: float = 0, max_area: float = np.inf) -> np.ndarray:
    if len(contours) == 0:
        return np.array([])

    def _keep_contour(c):
        area = cv2.contourArea(c)
        if area <= min_area:
            return False
        if area >= max_area:
            return False
        return True

    return np.array(list(filter(_keep_contour, contours)))


def get_contour_centers(contours: np.ndarray) -> np.ndarray:
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """

    if len(contours) == 0:
        return np.array([])

    # ((x, y), radius) = cv2.minEnclosingCircle(c)
    centers = np.zeros((len(contours), 2), dtype=np.int16)
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centers[i] = center
    return centers


def find_object_contours(image: np.ndarray, hsv_lower_value: Union[Tuple[int], List[int]],
                         hsv_upper_value: Union[Tuple[int], List[int]], kernel: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(hsv_lower_value), tuple(hsv_upper_value))
    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def get_bbox_for_contours(contours: np.ndarray) -> np.ndarray:
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return np.array(bboxes)


def calculate_distance_mtx(tracked_objects: List[TrackedObject], points: np.ndarray) -> np.ndarray:
    # (nb_tracked_objects, nb_current_detected_points)
    cost_mtx = np.zeros((len(tracked_objects), len(points)))
    for i, tracked_obj in enumerate(tracked_objects):
        for j, point in enumerate(points):
            diff = tracked_obj.last_point - point
            distance = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            cost_mtx[i][j] = distance
    return cost_mtx


def solve_assignment(cost_mtx: np.ndarray) -> List[int]:
    nb_tracked_objects, nb_detected_obj_centers = cost_mtx.shape
    assignment = [-1] * nb_tracked_objects
    row_index, column_index = optimize.linear_sum_assignment(cost_mtx)
    for i in range(len(row_index)):
        assignment[row_index[i]] = column_index[i]
    return assignment


def remove_object_if_too_many_frames_skipped(tracked_objects: List[TrackedObject], assignment: List[int],
                                             max_skipped_frames: int):
    for i, tracked_obj in enumerate(tracked_objects):
        if tracked_obj.skipped_frames > max_skipped_frames:
            del tracked_objects[i]
            del assignment[i]
