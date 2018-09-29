from typing import List, Tuple, Union

import cv2
import numpy as np


def calculate_distance(pt1, pt2):
    """
    Calculate the distance between two points
    :param pt1: tuple , 2D point
    :param pt2:  tuple, 2D point
    :return: distance between given points
    """

    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def crop_out_polygon_convex(image, point_array):
    """
    Crops out a convex polygon given from a list of points from an image
    :param image: Opencv BGR image
    :param point_array: list of points that defines a convex polygon
    :return: Cropped out image
    """

    point_array = np.array(point_array)
    point_array = np.reshape(cv2.convexHull(point_array), point_array.shape)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([point_array], dtype=np.int32)
    channel_count = get_image_channel(image)
    ignore_mask_color = (255,) * channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_image_channel(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        raise Exception("Image needs to have a shape like: (height, width, channels)")
    return c


def resize_img(image, min_width, min_height):
    """
    Resize the image with keeping the aspect ratio.
    min_width and min_height defines a bounding box
    :param image:
    :param min_width:
    :param min_height:
    :return: resized image
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        raise Exception("Something is wrong with the image dimensions")

    new_w = w
    new_h = h

    if w > min_width:
        new_w = min_width
        new_h = int(h * (float(new_w) / w))

    h, w, c = (new_h, new_w, c)
    if h > min_height:
        new_h = min_height
        new_w = int(w * (float(new_h) / h))

    resized = cv2.resize(image, (new_w, new_h))

    return resized


def sort_contours_by_area(contours, descending: bool = True):
    if len(contours) > 0:
        # sort in a descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=descending)
    return contours


def filter_contours_by_area(contours, min_area: float = 0, max_area: float = np.inf):
    if len(contours) == 0:
        return []

    def _keep_contour(c):
        area = cv2.contourArea(c)
        if area <= min_area:
            return False
        if area >= max_area:
            return False
        return True

    return np.array(list(filter(_keep_contour, contours)))


def get_largest_contour(contours, min_contour_area):
    """
    Find the largest contour in a set of detected contours
    :param contours: detected contours
    :param min_contour_area: only above this threshold it counts as a valid contour
    :return: contour
    """

    if len(contours) > 0:
        c = sort_contours_by_area(contours, True)[0]
        if cv2.contourArea(c) > min_contour_area:
            return c
    return None


def get_contour_centers(contours):
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """

    if len(contours) == 0:
        return []

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


def get_bbox_for_contours(contours):
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return np.array(bboxes)
