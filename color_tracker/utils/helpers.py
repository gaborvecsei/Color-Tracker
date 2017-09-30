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
        raise Exception("Image is not RGB image")
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
